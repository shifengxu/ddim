import logging
import os
import time
import torch
import torch.utils.data as data
from torch import optim
from torch.backends import cudnn

from datasets import get_dataset, data_transform
from functions import get_optimizer
from functions.losses import noise_estimation_loss2
from models.diffusion import Model
from models.ema import EMAHelper
from models.sam import SAM, disable_running_stats, enable_running_stats
from runners.diffusion import Diffusion
import utils


class DiffusionTrainingSam(Diffusion):
    """
    Diffusion training with SAM (Sharpness-Aware Minimization)
    """
    def __init__(self, args, config, device=None, output_ab=False):
        super().__init__(args, config, device, output_ab=output_ab)
        self.sam_flag = args.sam_flag
        self.model = None
        self.m_ema = None   # model for EMA
        self.optimizer = None
        self.optim_sam = None   # SAM optimizer
        self.scheduler = None
        self.ema_helper = None
        self.ema_flag = args.ema_flag
        self.ema_rate = args.ema_rate
        self.ema_start_epoch = args.ema_start_epoch
        logging.info(f"DiffusionTrainingSam()")
        logging.info(f"  sam_flag   : {self.sam_flag}")
        logging.info(f"  ema_flag   : {self.ema_flag}")
        logging.info(f"  ema_rate   : {self.ema_rate}")
        logging.info(f"  ema_s_epoch: {self.ema_start_epoch}")

        # statistic related data
        self.test_avg_arr = []  # test dataset loss avg array
        self.test_ema_arr = []  # test dataset loss avg array on self.m_ema (EMA model)
        self.ts_arr_arr = [[], [], []]  # array of array, for time-step array when testing

    def get_data_loaders(self):
        args, config = self.args, self.config
        batch_size = args.batch_size or config.training.batch_size
        dataset, test_dataset = get_dataset(args, config)
        train_loader = data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=config.data.num_workers,
        )
        logging.info(f"train dataset and data loader:")
        logging.info(f"  root       : {dataset.root}")
        logging.info(f"  split      : {dataset.split}") if hasattr(dataset, 'split') else None
        logging.info(f"  len        : {len(dataset)}")
        logging.info(f"  batch_cnt  : {len(train_loader)}")
        logging.info(f"  batch_size : {batch_size}")
        logging.info(f"  shuffle    : True")
        logging.info(f"  num_workers: {config.data.num_workers}")

        test_loader = data.DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=config.data.num_workers,
        )
        logging.info(f"test dataset and loader:")
        logging.info(f"  root          : {test_dataset.root}")
        logging.info(f"  len           : {len(test_dataset)}")
        logging.info(f"  test_data_dir : {args.test_data_dir}")
        logging.info(f"  batch_cnt     : {len(test_loader)}")
        logging.info(f"  batch_size    : {batch_size}")
        logging.info(f"  shuffle       : False")
        logging.info(f"  num_workers   : {config.data.num_workers}")
        return train_loader, test_loader

    def init_models(self):
        args, config = self.args, self.config
        in_channels = args.model_in_channels
        out_channels = args.model_in_channels
        resolution = args.data_resolution
        self.model = Model(config, in_channels=in_channels, out_channels=out_channels, resolution=resolution)
        self.m_ema = Model(config, in_channels=in_channels, out_channels=out_channels, resolution=resolution)
        self.m_ema.eval()  # only used for evaluation
        cnt, str_cnt = utils.count_parameters(self.model, log_fn=None)

        logging.info(f"DiffusionTrainingSam.model: ===================")
        logging.info(f"  model name : {type(self.model).__name__}")
        logging.info(f"  param size : {cnt} => {str_cnt}")
        logging.info(f"  model.to({self.device})")
        self.model.to(self.device)
        self.m_ema.to(self.device)
        self.optimizer = get_optimizer(self.config, self.model.parameters(), self.args.lr)
        kwargs = {"lr"          : self.args.lr,
                  "weight_decay": config.optim.weight_decay,
                  "betas"       : (config.optim.beta1, 0.999),
                  "amsgrad"     : config.optim.amsgrad,
                  "eps"         : config.optim.eps
                  }
        self.optim_sam = SAM(self.model.parameters(), optim.Adam, rho=0.05, adaptive=False, **kwargs)
        e_cnt = self.args.n_epochs
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=e_cnt)
        logging.info(f"DiffusionTrainingSam.optimizer:  ===================")
        logging.info(f"  optimizer   : {type(self.optimizer).__name__}")
        logging.info(f"  lr          : {self.args.lr}")
        logging.info(f"  sam_flag    : {self.args.sam_flag}")
        logging.info(f"  sam.rho     : {self.optim_sam.rho}")
        logging.info(f"  sam.adaptive: {self.optim_sam.adaptive}")

        if self.ema_flag:
            self.ema_helper = EMAHelper(mu=self.ema_rate)
            logging.info(f"  ema_helper: EMAHelper(mu={self.ema_rate})")
        else:
            self.ema_helper = None
            logging.info(f"  ema_helper: None")

        start_epoch = 0
        if self.args.resume_training:
            ckpt_path = self.args.resume_ckpt
            logging.info(f"resume_training: {self.args.resume_training}")
            logging.info(f"resume_ckpt    : {ckpt_path}")
            states = torch.load(ckpt_path, map_location=self.device)
            start_epoch = self.load_model(states)
        if len(args.gpu_ids) > 1:
            logging.info(f"torch.nn.DataParallel(device_ids={args.gpu_ids})")
            self.model = torch.nn.DataParallel(self.model, device_ids=args.gpu_ids)
            self.m_ema = torch.nn.DataParallel(self.m_ema, device_ids=args.gpu_ids)
        cudnn.benchmark = True
        return e_cnt, start_epoch

    def train(self):
        args, config = self.args, self.config
        train_loader, test_loader = self.get_data_loaders() # data loaders
        e_cnt, start_epoch = self.init_models()             # models, optimizer and others
        log_interval = args.log_interval
        test_per_epoch = args.test_per_epoch
        b_cnt = len(train_loader)               # batch count
        eb_cnt = (e_cnt - start_epoch) * b_cnt  # epoch * batch
        loss_avg_arr = []
        data_start = time.time()
        self.model.train()
        logging.info(f"DiffusionTrainingSam::train()")
        logging.info(f"  log_interval  : {log_interval}")
        logging.info(f"  start_epoch   : {start_epoch}")
        logging.info(f"  epoch_cnt     : {e_cnt}")
        logging.info(f"  test_per_epoch: {test_per_epoch}")
        for epoch in range(start_epoch, e_cnt):
            lr = self.scheduler.get_last_lr()[0]
            msg = f"lr={lr:8.7f}; ts=[{self.ts_low}, {self.ts_high}];"
            if self.ema_flag: msg += f" ema_start_epoch={self.ema_start_epoch}, ema_rate={self.ema_rate}"
            logging.info(f"Epoch {epoch}/{e_cnt} ---------- {msg}")
            if self.ema_flag and self.ema_start_epoch == epoch:
                logging.info(f"EMA register...")
                self.ema_helper.register(self.model)
            loss_ttl, loss_cnt, ema_cnt = 0., 0, 0
            for i, (x, y) in enumerate(train_loader):
                x = x.to(self.device)
                x = data_transform(self.config, x)
                e = torch.randn_like(x)
                loss, xt, upd_flag = self.train_model(x, e, epoch)
                loss_ttl += loss.item()
                loss_cnt += 1
                ema_cnt += upd_flag

                if i % log_interval == 0 or i == b_cnt - 1:
                    elp, eta = utils.get_time_ttl_and_eta(data_start, (epoch-start_epoch)*b_cnt+i, eb_cnt)
                    logging.info(f"E{epoch}.B{i:03d}/{b_cnt} loss:{loss.item():8.4f}; elp:{elp}, eta:{eta}")
            # for loader
            loss_avg = loss_ttl / loss_cnt
            logging.info(f"E:{epoch}/{e_cnt}: avg_loss:{loss_avg:8.4f} . ema_cnt:{ema_cnt}")
            loss_avg_arr.append(loss_avg)
            # self.scheduler.step()
            if epoch % 100 == 0 or epoch == e_cnt - 1:
                self.save_model(epoch)
            if test_per_epoch > 0 and (epoch % test_per_epoch == 0 or epoch == e_cnt - 1):
                self.test_and_save_result(epoch, test_loader)
        # for epoch
        utils.output_list(loss_avg_arr, 'loss_avg')
        utils.output_list(self.test_avg_arr, 'test_avg')
        utils.output_list(self.test_ema_arr, 'test_ema')

    def train_model(self, x_0_batch, epsilon, epoch):
        config = self.config
        b_sz = x_0_batch.size(0)  # batch size
        once_log = lambda msg: None
        if not self._ts_log_flag:
            self._ts_log_flag = True
            once_log = lambda msg: logging.info(f"train_model() {msg}")
        # smart_log: only log once during the training process.
        once_log(f"torch.randint({self.ts_low}, {self.ts_high}, ({b_sz},), {self.device})")
        t = torch.randint(low=self.ts_low, high=self.ts_high, size=(b_sz,), device=self.device)
        if self.sam_flag:
            # first forward-backward step
            once_log(f"first forward-backward step")
            loss, xt = noise_estimation_loss2(self.model, x_0_batch, t, epsilon, self.alphas_cumprod)
            self.optimizer.zero_grad()
            loss.backward()
            self.optim_sam.first_step(zero_grad=True)
            # second forward-backward step
            once_log(f"second forward-backward step")
            once_log(f"  disable_running_stats(self.model)")
            disable_running_stats(self.model)
            loss, xt = noise_estimation_loss2(self.model, x_0_batch, t, epsilon, self.alphas_cumprod)
            loss.backward()
            self.optim_sam.second_step(zero_grad=True)
            once_log(f"  enable_running_stats(self.model)")
            enable_running_stats(self.model)
        else:
            loss, xt = noise_estimation_loss2(self.model, x_0_batch, t, epsilon, self.alphas_cumprod)
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), config.optim.grad_clip)
            self.optimizer.step()

        ema_update_flag = 0
        if self.ema_flag and epoch >= self.ema_start_epoch:
            ema_update_flag = self.ema_helper.update(self.model)
        return loss, xt, ema_update_flag

    def test_and_save_result(self, epoch, test_loader):
        ts_msg = f"ts_arr_arr[{len(self.ts_arr_arr)}, {len(self.ts_arr_arr[0])}]"
        logging.info(f"E{epoch}. calculate loss on test dataset...{ts_msg}")
        self.model.eval()
        test_loss_avg = self.get_avg_loss(self.model, test_loader, self.ts_arr_arr)
        self.model.train()
        logging.info(f"E{epoch}. test_loss_avg:{test_loss_avg:8.4f}")
        self.test_avg_arr.append(test_loss_avg)
        if self.ema_flag and epoch >= self.ema_start_epoch:
            self.ema_helper.ema(self.m_ema)
            test_ema_avg = self.get_avg_loss(self.m_ema, test_loader, self.ts_arr_arr)
            self.test_ema_arr.append(test_ema_avg)
            logging.info(f"E{epoch}. test_ema_avg :{test_ema_avg:8.4f}")

    def get_avg_loss(self, model, test_loader, ts_arr_arr):
        """
        By ts_arr_arr, we can make sure that the testing process is deterministic.
        The first testing round will generate the timesteps. and the consequence
        testing rounds will reuse those timesteps.
        :param model:
        :param test_loader:
        :param ts_arr_arr: timestep array
        :return:
        """
        loss_ttl = 0.
        loss_cnt = 0
        with torch.no_grad():
            for ts_arr in ts_arr_arr:  # run multiple rounds, then get more accurate avg loss
                for i, (x, y) in enumerate(test_loader):
                    x = x.to(self.device)
                    x = data_transform(self.config, x)
                    e = torch.randn_like(x)

                    b_sz = x.size(0)  # batch size
                    if len(ts_arr) > i:
                        t = ts_arr[i]
                    else:
                        t = torch.randint(low=self.ts_low, high=self.ts_high, size=(b_sz,), device=self.device)
                        ts_arr.append(t)
                    loss, xt = noise_estimation_loss2(model, x, t, e, self.alphas_cumprod)
                    loss_ttl += loss.item()
                    loss_cnt += 1
                # for loader
            # for idx of arr
        # with
        loss_avg = loss_ttl / loss_cnt
        return loss_avg

    def load_model(self, states):
        self.model.load_state_dict(states['model'])
        op_st = states['optimizer']
        op_st["param_groups"][0]["eps"] = self.config.optim.eps
        self.optimizer.load_state_dict(op_st)
        self.scheduler.load_state_dict(states['scheduler'])
        start_epoch = states['cur_epoch']
        logging.info(f"  load_model_dict()...")
        logging.info(f"  resume optimizer  : eps={op_st['param_groups'][0]['eps']}")
        logging.info(f"  resume scheduler  : lr={self.scheduler.get_last_lr()[0]:8.6f}")
        logging.info(f"  resume start_epoch: {start_epoch}")
        if self.ema_flag:
            self.ema_helper.load_state_dict(states['ema_helper'])
            logging.info(f"  resume ema_helper : mu={self.ema_helper.mu:8.6f}")
        return start_epoch

    def save_model(self, e_idx):
        real_model = self.model
        if isinstance(real_model, torch.nn.DataParallel):
            real_model = real_model.module
        states = {
            'model'         : real_model.state_dict(),
            'optimizer'     : self.optimizer.state_dict(),
            'scheduler'     : self.scheduler.state_dict(),
            'beta_schedule' : self.beta_schedule,
            'cur_epoch'     : e_idx
        }
        if self.ema_flag and self.ema_helper:
            states['ema_helper'] = self.ema_helper.state_dict()

        fpath = os.path.join(self.args.log_path, f"ckpt_E{e_idx:04d}.pth")
        logging.info(f"save ckpt dict: {fpath}")
        torch.save(states, fpath)
        fpath = os.path.join(self.args.log_path, f"ckpt_{self.ts_low:03d}-{self.ts_high:03d}.pth")
        logging.info(f"save ckpt dict: {fpath}")
        torch.save(states, fpath)

# class
