import logging
import os
import time
import torch
import utils
import torch.utils.data as data

from torch.backends import cudnn
from datasets import get_dataset, data_transform
from functions import get_optimizer
from functions.losses import noise_estimation_loss2, x0_estimation_loss2
from models.diffusion import Model
from models.ema import EMAHelper
from runners.diffusion import Diffusion
from utils import count_parameters


class DiffusionTraining2(Diffusion):
    """
    Use 2 models to train and sample. The first model predicts epsilon, and the second
    predicts the original image x0.
    We call the first model model_et, and "et" means epsilon_t; and call the second
    model model_x0.
    The reason is: when timestep is near T (such as 1000), x_t is similar with epsilon_t,
    and model_et works well. This is confirmed by existing experiment. But on the other
    hand, when timestep is near 0, x_t is similar with x0. Therefore, we add model_x0 to
    predict x0. And hopefully, model_x0 will work better than model_et in such scenario.
    """
    def __init__(self, args, config, device=None):
        super().__init__(args, config, device)
        self.model_et = None
        self.model_x0 = None
        self.optim_et = None
        self.optim_x0 = None
        self.sched_et = None
        self.sched_x0 = None
        self.ema_et = None
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
        self.model_et = Model(config, in_channels=in_channels, out_channels=out_channels, resolution=resolution)
        self.model_x0 = Model(config, in_channels=in_channels, out_channels=out_channels, resolution=resolution)
        cnt, str_cnt = count_parameters(self.model_et, log_fn=None)
        model_name = type(self.model_et).__name__

        logging.info(f"model: {model_name} ===================")
        logging.info(f"  config type: {config.model.type}")
        logging.info(f"  param size : {cnt} => {str_cnt}")
        logging.info(f"  ema_flag   : {self.ema_flag}")
        logging.info(f"  ema_rate   : {self.ema_rate}")
        logging.info(f"  ema_start_epoch: {self.ema_start_epoch}")
        logging.info(f"  stack_sz   : {self.model_et.stack_sz}") if model_name == 'ModelStack' else None
        logging.info(f"  ts_cnt     : {self.model_et.ts_cnt}") if model_name == 'ModelStack' else None
        logging.info(f"  brick_cvg  : {self.model_et.brick_cvg}") if model_name == 'ModelStack' else None
        logging.info(f"  model.to({self.device})")
        self.model_et.to(self.device)
        self.model_x0.to(self.device)
        if self.args.n_epochs:
            e_cnt = self.args.n_epochs
        else:
            e_cnt = self.config.training.n_epochs
        self.optim_et = get_optimizer(self.config, self.model_et.parameters(), self.args.lr)
        self.optim_x0 = get_optimizer(self.config, self.model_x0.parameters(), self.args.lr)
        self.sched_et = torch.optim.lr_scheduler.CosineAnnealingLR(self.optim_et, T_max=e_cnt)
        self.sched_x0 = torch.optim.lr_scheduler.CosineAnnealingLR(self.optim_x0, T_max=e_cnt)
        logging.info(f"  optimizer  : {type(self.optim_et).__name__}")
        logging.info(f"  lr         : {self.args.lr}")

        if self.ema_flag:
            self.ema_et = EMAHelper(mu=self.ema_rate)
            self.ema_x0 = EMAHelper(mu=self.ema_rate)
            logging.info(f"  ema_helper: EMAHelper(mu={self.ema_rate})")
        else:
            self.ema_et = None
            self.ema_x0 = None
            logging.info(f"  ema_helper: None")

        start_epoch = 0
        if self.args.resume_training:
            ckpt_path = self.args.resume_ckpt
            logging.info(f"resume_training: {self.args.resume_training}")
            logging.info(f"resume_ckpt2   : {ckpt_path}")
            states = torch.load(ckpt_path, map_location=self.device)
            start_epoch = self.load_model(states)
        logging.info(f"  torch.nn.DataParallel(device_ids={args.gpu_ids})")
        self.model_et = torch.nn.DataParallel(self.model_et, device_ids=args.gpu_ids)
        self.model_x0 = torch.nn.DataParallel(self.model_x0, device_ids=args.gpu_ids)
        cudnn.benchmark = True

        b_cnt = len(train_loader)
        eb_cnt = (e_cnt - start_epoch) * b_cnt  # epoch * batch
        loss_et_avg_arr = []
        loss_x0_avg_arr = []
        test_et_avg_arr = []  # test dataset loss avg array by model_et
        test_x0_avg_arr = []  # test dataset loss avg array by model_x0
        data_start = time.time()
        self.model_et.train()
        self.model_x0.train()
        log_interval = 1 if model_name == 'ModelStack' else 10
        test_ts_arr = []  # time-step array for testing
        logging.info(f"log_interval: {log_interval}")
        logging.info(f"start_epoch : {start_epoch}")
        logging.info(f"epoch_cnt   : {e_cnt}")
        logging.info(f"b_cnt       : {b_cnt}")
        for epoch in range(start_epoch, e_cnt):
            lr = self.sched_et.get_last_lr()[0]
            msg = f"lr={lr:8.7f}; ts=[{self.ts_low}, {self.ts_high}];"
            if self.ema_flag: msg += f" ema_start_epoch={self.ema_start_epoch}, ema_rate={self.ema_rate}"
            logging.info(f"Epoch {epoch}/{e_cnt} ---------- {msg}")
            if self.ema_flag and self.ema_start_epoch == epoch:
                logging.info(f"EMA register...")
                self.ema_et.register(self.model_et)
                self.ema_x0.register(self.model_x0)
            loss_et_ttl = 0.
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
                loss_et, loss_x0, xt, upd_flag = self.train_model(x, e, epoch)
                loss_et_ttl += loss_et.item()
                loss_x0_ttl += loss_x0.item()
                loss_cnt += 1
                ema_cnt += upd_flag

                if i % log_interval == 0 or i == b_cnt - 1:
                    elp, eta = utils.get_time_ttl_and_eta(data_start, (epoch-start_epoch) * b_cnt + i, eb_cnt)
                    var, mean = torch.var_mean(xt)
                    logging.info(f"E{epoch}.B{i:03d}/{b_cnt} loss:{loss_et.item():7.3f} {loss_x0.item():7.3f};"
                                 f" x0:{mean0:6.3f} {var0:5.3f}; xt:{mean:6.3f} {var:5.3f};"
                                 f" elp:{elp}, eta:{eta}")

            # for loader
            loss_et_avg = loss_et_ttl / loss_cnt
            loss_x0_avg = loss_x0_ttl / loss_cnt
            logging.info(f"E:{epoch}/{e_cnt}: avg_loss:{loss_et_avg:8.4f} {loss_x0_avg:8.4f}. ema_cnt:{ema_cnt}")
            loss_et_avg_arr.append(loss_et_avg)
            loss_x0_avg_arr.append(loss_x0_avg)
            # self.sched_et.step()
            # self.sched_x0.step()
            if epoch % 100 == 0 or epoch == e_cnt - 1:
                self.save_model(epoch)
            if test_per_epoch > 0 and (epoch % test_per_epoch == 0 or epoch == e_cnt - 1):
                logging.info(f"E{epoch}. calculate loss on test dataset...test_ts_arr:{len(test_ts_arr)}")
                self.model_et.eval()
                self.model_x0.eval()
                test_et_avg = self.get_avg_loss(self.model_et, test_loader, test_ts_arr, noise_estimation_loss2)
                test_x0_avg = self.get_avg_loss(self.model_x0, test_loader, test_ts_arr, x0_estimation_loss2)
                self.model_et.train()
                self.model_x0.train()
                logging.info(f"E{epoch}. test_loss_avg:{test_et_avg:8.4f} {test_x0_avg:8.4f}")
                test_et_avg_arr.append(test_et_avg)
                test_x0_avg_arr.append(test_x0_avg)
        # for epoch
        utils.output_list(loss_et_avg_arr, 'loss_et_avg')
        utils.output_list(loss_x0_avg_arr, 'loss_x0_avg')
        utils.output_list(test_et_avg_arr, 'test_et_avg')
        utils.output_list(test_x0_avg_arr, 'test_x0_avg')
    # train(self)

    def get_avg_loss(self, model, test_loader, test_ts_arr, loss_fn):
        """
        By test_ts_arr, we can make sure that the testing process is deterministic.
        The first testing round will generate the timesteps. and the consequence
        testing rounds will reuse those timesteps.
        :param model:
        :param test_loader:
        :param test_ts_arr: timestep array
        :param loss_fn
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
                loss, xt = loss_fn(model, x, t, e, self.alphas_cumprod)
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

        loss_et, xt = noise_estimation_loss2(self.model_et, x, t, epsilon, self.alphas_cumprod)
        self.optim_et.zero_grad()
        loss_et.backward()
        torch.nn.utils.clip_grad_norm_(self.model_et.parameters(), config.optim.grad_clip)
        self.optim_et.step()

        loss_x0, _ = x0_estimation_loss2(self.model_x0, x, t, epsilon, self.alphas_cumprod)
        self.optim_x0.zero_grad()
        loss_x0.backward()
        torch.nn.utils.clip_grad_norm_(self.model_x0.parameters(), config.optim.grad_clip)
        self.optim_x0.step()

        ema_update_flag = 0
        if self.ema_flag and epoch >= self.ema_start_epoch:
            ema_update_flag = self.ema_et.update(self.model_et)
            self.ema_x0.update(self.model_x0)
        return loss_et, loss_x0, xt, ema_update_flag

    def load_model(self, states):
        self.model_et.load_state_dict(states['model'])
        self.model_x0.load_state_dict(states['model_x0'])
        op_et = states['optimizer']
        op_et["param_groups"][0]["eps"] = self.config.optim.eps
        self.optim_et.load_state_dict(op_et)
        op_x0 = states['optim_x0']
        op_x0["param_groups"][0]["eps"] = self.config.optim.eps
        self.optim_x0.load_state_dict(op_x0)
        self.sched_et.load_state_dict(states['scheduler'])
        self.sched_x0.load_state_dict(states['sched_x0'])
        start_epoch = states['cur_epoch']
        logging.info(f"  load_model_dict()...")
        logging.info(f"  resume model_et")
        logging.info(f"  resume model_x0")
        logging.info(f"  resume optim_et   : eps={op_et['param_groups'][0]['eps']}")
        logging.info(f"  resume optim_x0   : eps={op_x0['param_groups'][0]['eps']}")
        logging.info(f"  resume sched_et   : lr={self.sched_et.get_last_lr()[0]:8.6f}")
        logging.info(f"  resume sched_x0   : lr={self.sched_x0.get_last_lr()[0]:8.6f}")
        logging.info(f"  resume start_epoch: {start_epoch}")
        if self.ema_flag:
            self.ema_et.load_state_dict(states['ema_helper'])
            self.ema_x0.load_state_dict(states['ema_x0'])
            logging.info(f"  resume ema_et     : mu={self.ema_et.mu:8.6f}")
            logging.info(f"  resume ema_x0     : mu={self.ema_x0.mu:8.6f}")
        return start_epoch

    def save_model(self, e_idx):
        real_model_et = self.model_et
        if isinstance(real_model_et, torch.nn.DataParallel):
            real_model_et = real_model_et.module
        real_model_x0 = self.model_x0
        if isinstance(real_model_x0, torch.nn.DataParallel):
            real_model_x0 = real_model_x0.module
        states = {
            'model'    : real_model_et.state_dict(),  # use old name; then compatible in diffusion_training.py
            'model_x0' : real_model_x0.state_dict(),
            'optimizer': self.optim_et.state_dict(),
            'optim_x0' : self.optim_x0.state_dict(),
            'scheduler': self.sched_et.state_dict(),
            'sched_x0' : self.sched_x0.state_dict(),
            'beta_schedule': self.beta_schedule,
            'cur_epoch': e_idx
        }
        if self.ema_flag and self.ema_et:
            states['ema_helper'] = self.ema_et.state_dict()
        if self.ema_flag and self.ema_x0:
            states['ema_x0'] = self.ema_x0.state_dict()

        fpath = os.path.join(self.args.log_path, f"ckpt2_E{e_idx:04d}.pth")
        logging.info(f"save ckpt2 dict: {fpath}")
        torch.save(states, fpath)
        fpath = os.path.join(self.args.log_path, f"ckpt2_{self.ts_low:03d}-{self.ts_high:03d}.pth")
        logging.info(f"save ckpt2 dict: {fpath}")
        torch.save(states, fpath)

# class
