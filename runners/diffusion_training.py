import logging
import os
import time

import torch
from torch.backends import cudnn

import utils
from datasets import get_dataset, data_transform
from functions import get_optimizer
from functions.losses import noise_estimation_loss2
from models.diffusion import Model
from models.ema import EMAHelper
from runners.diffusion import Diffusion

import torch.utils.data as data

from utils import count_parameters


class DiffusionTraining(Diffusion):
    def __init__(self, args, config, device=None):
        super().__init__(args, config, device)
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.ema_helper = None
        self.ema_flag = args.ema_flag
        self.ema_rate = args.ema_rate or config.model.ema_rate
        self.ema_start_epoch = args.ema_start_epoch

    def train(self):
        args, config = self.args, self.config
        tb_logger = self.config.tb_logger
        dataset, test_dataset = get_dataset(args, config)
        logging.info(f"dataset:")
        logging.info(f"  root : {dataset.root}")
        logging.info(f"  split: {dataset.split}") if hasattr(dataset, 'split') else None
        logging.info(f"  len  : {len(dataset)}")
        logging.info(f"data_loader:")
        logging.info(f"  batch_size : {config.training.batch_size}")
        logging.info(f"  shuffle    : True")
        logging.info(f"  num_workers: {config.data.num_workers}")
        train_loader = data.DataLoader(
            dataset,
            batch_size=config.training.batch_size,
            shuffle=True,
            num_workers=config.data.num_workers,
        )
        in_channels = args.model_in_channels
        out_channels = args.model_in_channels
        resolution = args.data_resolution
        self.model = Model(config, in_channels=in_channels, out_channels=out_channels, resolution=resolution)
        # model = ModelStack(config)
        cnt, str_cnt = count_parameters(self.model, log_fn=None)
        model_name = type(self.model).__name__

        logging.info(f"model: {model_name} ===================")
        logging.info(f"  config type: {config.model.type}")
        logging.info(f"  param size : {cnt} => {str_cnt}")
        logging.info(f"  ema_flag   : {self.ema_flag}")
        logging.info(f"  ema_rate   : {self.ema_rate}")
        logging.info(f"  ema_start_epoch: {self.ema_start_epoch}")
        logging.info(f"  stack_sz   : {self.model.stack_sz}") if model_name == 'ModelStack' else None
        logging.info(f"  ts_cnt     : {self.model.ts_cnt}") if model_name == 'ModelStack' else None
        logging.info(f"  brick_cvg  : {self.model.brick_cvg}") if model_name == 'ModelStack' else None
        logging.info(f"  model.to({self.device})")
        if self.args.n_epochs:
            e_cnt = self.args.n_epochs
        else:
            e_cnt = self.config.training.n_epochs
        self.model.to(self.device)
        self.optimizer = get_optimizer(self.config, self.model.parameters(), self.args.lr)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=e_cnt)

        if self.ema_flag:
            self.ema_helper = EMAHelper(mu=self.ema_rate)
            logging.info(f"  ema_helper: EMAHelper(mu={self.ema_rate})")
        else:
            self.ema_helper = None
            logging.info(f"  ema_helper: None")

        start_epoch = 0
        if self.args.resume_training:
            ckpt_path = os.path.join(self.args.log_path, "ckpt.pth")
            logging.info(f"resume_training: {ckpt_path}")
            states = torch.load(ckpt_path, map_location=self.device)
            start_epoch = self.load_model(states)
        logging.info(f"  torch.nn.DataParallel(device_ids={args.gpu_ids})")
        self.model = torch.nn.DataParallel(self.model, device_ids=args.gpu_ids)
        cudnn.benchmark = True

        b_cnt = len(train_loader)
        eb_cnt = e_cnt * b_cnt  # epoch * batch
        loss_avg_arr = []
        data_start = time.time()
        self.model.train()
        log_interval = 1 if model_name == 'ModelStack' else 10
        logging.info(f"log_interval: {log_interval}")
        logging.info(f"start_epoch : {start_epoch}")
        logging.info(f"epoch_cnt   : {e_cnt}")
        for epoch in range(start_epoch, e_cnt):
            lr = self.scheduler.get_last_lr()[0]
            msg = f"lr={lr:8.7f}; ts=[{self.ts_low}, {self.ts_high}];"
            if self.ema_flag: msg += f" ema_start_epoch={self.ema_start_epoch}, ema_rate={self.ema_rate}"
            logging.info(f"Epoch {epoch}/{e_cnt} ---------- {msg}")
            if self.ema_flag and self.ema_start_epoch == epoch:
                logging.info(f"EMA register...")
                self.ema_helper.register(self.model)
            loss_ttl = 0.
            loss_cnt = 0
            for i, (x, y) in enumerate(train_loader):
                x = x.to(self.device)
                x = data_transform(self.config, x)
                if i % log_interval == 0 or i == b_cnt - 1:
                    var0, mean0 = torch.var_mean(x)
                e = torch.randn_like(x)

                # antithetic sampling
                if model_name == 'ModelStack':
                    loss, xt = self.train_model_stack(x, e)
                else:
                    loss, xt = self.train_model(x, e, epoch)
                loss_ttl += loss.item()
                loss_cnt += 1

                if i % log_interval == 0 or i == b_cnt - 1:
                    elp, eta = utils.get_time_ttl_and_eta(data_start, epoch * b_cnt + i, eb_cnt)
                    var, mean = torch.var_mean(xt)
                    logging.info(f"E{epoch}.B{i:03d}/{b_cnt} loss:{loss.item():8.4f};"
                                 f" x0:{mean0:7.4f} {var0:6.4f}; xt:{mean:7.4f} {var:6.4f};"
                                 f" elp:{elp}, eta:{eta}")

            # for loader
            loss_avg = loss_ttl / loss_cnt
            logging.info(f"E:{epoch}/{e_cnt}: avg_loss:{loss_avg:8.4f}")
            loss_avg_arr.append(loss_avg)
            # self.scheduler.step()
            if epoch % 20 == 0 or epoch == e_cnt - 1:
                self.save_model(epoch)
        # for epoch
        utils.output_list(loss_avg_arr, 'loss_avg')
    # train(self)

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
        loss, xt = noise_estimation_loss2(self.model, x, t, epsilon, self.alphas_cumprod)
        self.optimizer.zero_grad()
        loss.backward()
        try:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), config.optim.grad_clip)
        except Exception:
            pass
        self.optimizer.step()

        if self.ema_flag and epoch >= self.ema_start_epoch:
            self.ema_helper.update(self.model)
            self.ema_helper.ema(self.model)
        return loss, xt

    def train_model_stack(self, x, e):
        config = self.config
        b_sz = x.size(0)  # batch size
        stack_sz = config.model.stack_size if hasattr(config.model, 'stack_size') else 5
        ts_cnt = self.num_timesteps
        brick_cvg = ts_cnt // stack_sz
        total_loss = 0.
        for k in range(stack_sz):
            low = brick_cvg * k
            high = brick_cvg * (k + 1)
            if high > ts_cnt: high = ts_cnt
            t = torch.randint(low=low, high=high, size=(b_sz,), device=self.device)
            if not self._ts_log_flag:
                self._ts_log_flag = True
                logging.info(f"train_model_stack() timestep: torch.randint(low={self.ts_low}, "
                             f"high={self.ts_high}, size=({b_sz},), device={self.device})")
            loss, xt = noise_estimation_loss2(self.model, x, t, e, self.alphas_cumprod)
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), config.optim.grad_clip)
            self.optimizer.step()
            if self.ema_flag:
                self.ema_helper.update(self.model)
            total_loss += loss
        avg_loss = total_loss / stack_sz
        return avg_loss, xt

    def load_model(self, states):
        if isinstance(states, dict):
            # If states is a dict, use another function.
            # This is for backward-compatibility. As previously, the states is a list ([]).
            # And that was not convenient for adding or dropping items.
            # Therefore, we change to use dict.
            return self.load_model_dict(states)
        self.model.load_state_dict(states[0])

        states[1]["param_groups"][0]["eps"] = self.config.optim.eps
        self.optimizer.load_state_dict(states[1])
        start_epoch = states[2]
        logging.info(f"  resume optimizer  : eps={states[1]['param_groups'][0]['eps']}")
        logging.info(f"  resume start_epoch: {start_epoch}")
        if self.ema_flag:
            self.ema_helper.load_state_dict(states[-1])
            logging.info(f"  resume ema_helper : ...")
        return start_epoch

    def load_model_dict(self, states):
        self.model.load_state_dict(states['model'])
        op_st = states['optimizer']
        op_st["param_groups"][0]["eps"] = self.config.optim.eps
        self.optimizer.load_state_dict(op_st)
        self.scheduler.load_state_dict(states['scheduler'])
        start_epoch = states['cur_epoch']
        logging.info(f"  load_model_dict()...")
        logging.info(f"  resume optimizer  : eps={op_st['param_groups'][0]['eps']}")
        logging.info(f"  resume scheduler  : lr={self.scheduler.get_last_lr():8.6f}")
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
            'model': real_model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'beta_schedule': self.beta_schedule,
            'beta_cos_expo': self.beta_cos_expo,
            'cur_epoch': e_idx
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
