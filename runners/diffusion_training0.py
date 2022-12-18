import logging
import os
import time
import torch
import utils
import torch.utils.data as data

from torch.backends import cudnn
from datasets import get_dataset, data_transform
from functions import get_optimizer
from functions.losses import x0_estimation_loss2
from models.diffusion import Model
from models.ema import EMAHelper
from runners.diffusion import Diffusion
from utils import count_parameters


class DiffusionTraining0(Diffusion):
    """
    predict x0
    """
    def __init__(self, args, config, device=None):
        super().__init__(args, config, device)
        self.model_x0 = None
        self.optim_x0 = None
        self.sched_x0 = None
        self.ema_x0 = None
        self.ema_flag = args.ema_flag
        self.ema_rate = args.ema_rate or config.model.ema_rate
        self.ema_start_epoch = args.ema_start_epoch

    def train(self):
        args, config = self.args, self.config
        dataset, test_dataset = get_dataset(args, config)
        logging.info(f"train dataset:")
        logging.info(f"  root : {dataset.root}")
        logging.info(f"  split: {dataset.split}") if hasattr(dataset, 'split') else None
        logging.info(f"  len  : {len(dataset)}")
        logging.info(f"test dataset:")
        logging.info(f"  root : {test_dataset.root}")
        logging.info(f"  len  : {len(test_dataset)}")
        batch_size = args.batch_size or config.training.batch_size
        train_loader = data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=config.data.num_workers,
        )
        logging.info(f"train_data_loader:")
        logging.info(f"  batch_cnt  : {len(train_loader)}")
        logging.info(f"  batch_size : {batch_size}")
        logging.info(f"  shuffle    : True")
        logging.info(f"  num_workers: {config.data.num_workers}")
        test_per_epoch = args.test_per_epoch
        test_loader = data.DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=config.data.num_workers,
        )
        logging.info(f"test_data_loader:")
        logging.info(f"  test_per_epoch: {test_per_epoch}")
        logging.info(f"  test_data_dir : {args.test_data_dir}")
        logging.info(f"  batch_cnt     : {len(test_loader)}")
        logging.info(f"  batch_size    : {batch_size}")
        logging.info(f"  shuffle       : False")
        logging.info(f"  num_workers   : {config.data.num_workers}")
        in_channels = args.model_in_channels
        out_channels = args.model_in_channels
        resolution = args.data_resolution
        self.model_x0 = Model(config, in_channels=in_channels, out_channels=out_channels, resolution=resolution)
        cnt, str_cnt = count_parameters(self.model_x0, log_fn=None)
        model_name = type(self.model_x0).__name__

        logging.info(f"model: {model_name} ===================")
        logging.info(f"  config type: {config.model.type}")
        logging.info(f"  param size : {cnt} => {str_cnt}")
        logging.info(f"  ema_flag   : {self.ema_flag}")
        logging.info(f"  ema_rate   : {self.ema_rate}")
        logging.info(f"  ema_start_epoch: {self.ema_start_epoch}")
        logging.info(f"  stack_sz   : {self.model_x0.stack_sz}") if model_name == 'ModelStack' else None
        logging.info(f"  ts_cnt     : {self.model_x0.ts_cnt}") if model_name == 'ModelStack' else None
        logging.info(f"  brick_cvg  : {self.model_x0.brick_cvg}") if model_name == 'ModelStack' else None
        logging.info(f"  model.to({self.device})")
        self.model_x0.to(self.device)
        if self.args.n_epochs:
            e_cnt = self.args.n_epochs
        else:
            e_cnt = self.config.training.n_epochs
        self.optim_x0 = get_optimizer(self.config, self.model_x0.parameters(), self.args.lr)
        self.sched_x0 = torch.optim.lr_scheduler.CosineAnnealingLR(self.optim_x0, T_max=e_cnt)
        logging.info(f"  optimizer  : {type(self.optim_x0).__name__}")
        logging.info(f"  lr         : {self.args.lr}")

        if self.ema_flag:
            self.ema_x0 = EMAHelper(mu=self.ema_rate)
            logging.info(f"  ema_helper: EMAHelper(mu={self.ema_rate})")
        else:
            self.ema_x0 = None
            logging.info(f"  ema_helper: None")

        start_epoch = 0
        if self.args.resume_training:
            ckpt_path = self.args.resume_ckpt
            logging.info(f"resume_training: {self.args.resume_training}")
            logging.info(f"resume_ckpt0   : {ckpt_path}")
            states = torch.load(ckpt_path, map_location=self.device)
            start_epoch = self.load_model(states)
        logging.info(f"  torch.nn.DataParallel(device_ids={args.gpu_ids})")
        self.model_x0 = torch.nn.DataParallel(self.model_x0, device_ids=args.gpu_ids)
        cudnn.benchmark = True

        b_cnt = len(train_loader)
        eb_cnt = (e_cnt - start_epoch) * b_cnt  # epoch * batch
        loss_x0_avg_arr = []
        test_x0_avg_arr = []  # test dataset loss avg array by model_x0
        data_start = time.time()
        self.model_x0.train()
        log_interval = 1 if model_name == 'ModelStack' else 10
        test_ts_arr = []  # time-step array for testing
        logging.info(f"log_interval: {log_interval}")
        logging.info(f"start_epoch : {start_epoch}")
        logging.info(f"epoch_cnt   : {e_cnt}")
        logging.info(f"b_cnt       : {b_cnt}")
        for epoch in range(start_epoch, e_cnt):
            lr = self.sched_x0.get_last_lr()[0]
            msg = f"lr={lr:8.7f}; ts=[{self.ts_low}, {self.ts_high}];"
            if self.ema_flag: msg += f" ema_start_epoch={self.ema_start_epoch}, ema_rate={self.ema_rate}"
            logging.info(f"Epoch {epoch}/{e_cnt} ---------- {msg}")
            if self.ema_flag and self.ema_start_epoch == epoch:
                logging.info(f"EMA register...")
                self.ema_x0.register(self.model_x0)
            loss_x0_ttl = 0.
            loss_cnt = 0
            ema_cnt = 0
            for i, (x, y) in enumerate(train_loader):
                x = x.to(self.device)
                x = data_transform(self.config, x)
                if i % log_interval == 0 or i == b_cnt - 1:
                    var0, mean0 = torch.var_mean(x)
                e = torch.randn_like(x)

                # antithetic sampling
                loss_x0, xt, upd_flag = self.train_model(x, e, epoch)
                loss_x0_ttl += loss_x0.item()
                loss_cnt += 1
                ema_cnt += upd_flag

                if i % log_interval == 0 or i == b_cnt - 1:
                    elp, eta = utils.get_time_ttl_and_eta(data_start, (epoch-start_epoch) * b_cnt + i, eb_cnt)
                    var, mean = torch.var_mean(xt)
                    logging.info(f"E{epoch}.B{i:03d}/{b_cnt} loss:{loss_x0.item():7.3f};"
                                 f" x0:{mean0:6.3f} {var0:5.3f}; xt:{mean:6.3f} {var:5.3f};"
                                 f" elp:{elp}, eta:{eta}")

            # for loader
            loss_x0_avg = loss_x0_ttl / loss_cnt
            logging.info(f"E:{epoch}/{e_cnt}: avg_loss:{loss_x0_avg:8.4f}. ema_cnt:{ema_cnt}")
            loss_x0_avg_arr.append(loss_x0_avg)
            # self.sched_x0.step()
            if epoch % 100 == 0 or epoch == e_cnt - 1:
                self.save_model(epoch)
            if test_per_epoch > 0 and (epoch % test_per_epoch == 0 or epoch == e_cnt - 1):
                logging.info(f"E{epoch}. calculate loss on test dataset...test_ts_arr:{len(test_ts_arr)}")
                self.model_x0.eval()
                test_x0_avg = self.get_avg_loss(self.model_x0, test_loader, test_ts_arr)
                self.model_x0.train()
                logging.info(f"E{epoch}. test_loss_avg: {test_x0_avg:8.4f}")
                test_x0_avg_arr.append(test_x0_avg)
        # for epoch
        utils.output_list(loss_x0_avg_arr, 'loss_x0_avg')
        utils.output_list(test_x0_avg_arr, 'test_x0_avg')
    # train(self)

    def get_avg_loss(self, model, test_loader, test_ts_arr):
        """
        By test_ts_arr, we can make sure that the testing process is deterministic.
        The first testing round will generate the timesteps. and the consequence
        testing rounds will reuse those timesteps.
        :param model:
        :param test_loader:
        :param test_ts_arr: timestep array
        :return:
        """
        loss_ttl = 0.
        loss_cnt = 0
        with torch.no_grad():
            for i, (x, y) in enumerate(test_loader):
                x = x.to(self.device)
                x = data_transform(self.config, x)
                e = torch.randn_like(x)

                b_sz = x.size(0)  # batch size
                if len(test_ts_arr) > i:
                    t = test_ts_arr[i]
                else:
                    t = torch.randint(low=self.ts_low, high=self.ts_high, size=(b_sz,), device=self.device)
                    test_ts_arr.append(t)
                loss, xt = x0_estimation_loss2(model, x, t, e, self.alphas_cumprod)
                loss_ttl += loss.item()
                loss_cnt += 1
            # for loader
        # with
        loss_avg = loss_ttl / loss_cnt
        return loss_avg

    def train_model(self, x, epsilon, epoch):
        """
        train model
        :param x: input clean image
        :param epsilon: epsilon, distribution N(0, 1)
        :param epoch:
        :return:
        """
        config = self.config
        b_sz = x.size(0)  # batch size
        # t = torch.randint(high=self.num_timesteps, size=(b_sz // 2 + 1,), device=self.device)
        # t = torch.cat([t, self.num_timesteps - t - 1], dim=0)[:b_sz]
        if not self._ts_log_flag:
            self._ts_log_flag = True
            logging.info(f"train_model() timestep: torch.randint(low={self.ts_low}, "
                         f"high={self.ts_high}, size=({b_sz},), device={self.device})")
        t = torch.randint(low=self.ts_low, high=self.ts_high, size=(b_sz,), device=self.device)

        loss_x0, xt = x0_estimation_loss2(self.model_x0, x, t, epsilon, self.alphas_cumprod)
        self.optim_x0.zero_grad()
        loss_x0.backward()
        torch.nn.utils.clip_grad_norm_(self.model_x0.parameters(), config.optim.grad_clip)
        self.optim_x0.step()

        ema_update_flag = 0
        if self.ema_flag and epoch >= self.ema_start_epoch:
            ema_update_flag = self.ema_x0.update(self.model_x0)
        return loss_x0, xt, ema_update_flag

    def load_model(self, states):
        self.model_x0.load_state_dict(states['model_x0'])
        op_x0 = states['optim_x0']
        op_x0["param_groups"][0]["eps"] = self.config.optim.eps
        self.optim_x0.load_state_dict(op_x0)
        self.sched_x0.load_state_dict(states['sched_x0'])
        start_epoch = states['cur_epoch']
        logging.info(f"  load_model_dict()...")
        logging.info(f"  resume model_x0")
        logging.info(f"  resume optim_x0   : eps={op_x0['param_groups'][0]['eps']}")
        logging.info(f"  resume sched_x0   : lr={self.sched_x0.get_last_lr()[0]:8.6f}")
        logging.info(f"  resume start_epoch: {start_epoch}")
        if self.ema_flag:
            self.ema_x0.load_state_dict(states['ema_x0'])
            logging.info(f"  resume ema_x0     : mu={self.ema_x0.mu:8.6f}")
        return start_epoch

    def save_model(self, e_idx):
        real_model_x0 = self.model_x0
        if isinstance(real_model_x0, torch.nn.DataParallel):
            real_model_x0 = real_model_x0.module
        states = {
            'model_x0' : real_model_x0.state_dict(),
            'optim_x0' : self.optim_x0.state_dict(),
            'sched_x0' : self.sched_x0.state_dict(),
            'beta_schedule': self.beta_schedule,
            'cur_epoch': e_idx
        }
        if self.ema_flag and self.ema_x0:
            states['ema_x0'] = self.ema_x0.state_dict()

        fpath = os.path.join(self.args.log_path, f"ckpt_E{e_idx:04d}.pth")
        logging.info(f"save ckpt dict: {fpath}")
        torch.save(states, fpath)
        fpath = os.path.join(self.args.log_path, f"ckpt_{self.ts_low:03d}-{self.ts_high:03d}.pth")
        logging.info(f"save ckpt dict: {fpath}")
        torch.save(states, fpath)

# class
