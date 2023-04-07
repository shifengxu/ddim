import logging
import os
import random
import time

import torch
from torch.backends import cudnn

import utils
from datasets import get_dataset, data_transform
from functions import get_optimizer
from models.diffusion import Model
from models.ema import EMAHelper
from runners.diffusion import Diffusion

import torch.utils.data as data


class DiffusionTrainingFast(Diffusion):
    def __init__(self, args, config, device=None):
        super().__init__(args, config, device, output_ab=False)
        self.model = None
        self.m_ema = None   # model for EMA
        self.ab_list = args.ab_list
        self.optimizer = None
        self.scheduler = None
        self.ema_helper = None
        self.ema_flag = args.ema_flag
        self.ema_rate = args.ema_rate
        self.ema_start_epoch = args.ema_start_epoch
        logging.info(f"DiffusionTrainingFast() ===================")
        logging.info(f"  ab_list        : {self.ab_list}")
        logging.info(f"  config type    : {config.model.type}")
        logging.info(f"  ema_flag       : {self.ema_flag}")
        logging.info(f"  ema_rate       : {self.ema_rate}")
        logging.info(f"  ema_start_epoch: {self.ema_start_epoch}")
        if len(self.ab_list) == 0:
            raise ValueError(f"ab_list cannot be empty.")

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
        in_ch  = args.model_in_channels
        out_ch = args.model_in_channels
        res    = args.data_resolution
        ts_type = 'continuous'
        self.model = Model(config, in_channels=in_ch, out_channels=out_ch, resolution=res, ts_type=ts_type)
        self.m_ema = Model(config, in_channels=in_ch, out_channels=out_ch, resolution=res, ts_type=ts_type)
        self.m_ema.eval()  # only used for evaluation

        if self.args.n_epochs:
            e_cnt = self.args.n_epochs
        else:
            e_cnt = self.config.training.n_epochs
        logging.info(f"DiffusionTrainingFast::init_models() ===================")
        logging.info(f"  model.to({self.device})")
        self.model.to(self.device)
        self.m_ema.to(self.device)
        self.optimizer = get_optimizer(self.config, self.model.parameters(), self.args.lr)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=e_cnt)
        logging.info(f"  optimizer: {type(self.optimizer).__name__} ===================")
        logging.info(f"    lr: {self.args.lr}")

        if self.ema_flag:
            self.ema_helper = EMAHelper(mu=self.ema_rate)
            logging.info(f"  ema_helper: EMAHelper(mu={self.ema_rate})")
        else:
            self.ema_helper = None
            logging.info(f"  ema_helper: None")

        start_epoch = 0
        if self.args.resume_training:
            ckpt_path = self.args.resume_ckpt
            logging.info(f"  resume_training: {self.args.resume_training}")
            logging.info(f"  resume_ckpt    : {ckpt_path}")
            states = torch.load(ckpt_path, map_location=self.device)
            start_epoch = self.load_model(states)
        logging.info(f"  torch.nn.DataParallel(device_ids={args.gpu_ids})")
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
        logging.info(f"log_interval  : {log_interval}")
        logging.info(f"start_epoch   : {start_epoch}")
        logging.info(f"epoch_cnt     : {e_cnt}")
        logging.info(f"test_per_epoch: {test_per_epoch}")
        logging.info(f"save_per_epoch: {args.save_per_epoch}")
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

                # antithetic sampling
                loss, upd_flag = self.train_model(x, e, epoch, i)
                loss_ttl += loss.item()
                loss_cnt += 1
                ema_cnt += upd_flag

                if i % log_interval == 0 or i == b_cnt - 1:
                    elp, eta = utils.get_time_ttl_and_eta(data_start, (epoch-start_epoch) * b_cnt + i, eb_cnt)
                    logging.info(f"E{epoch}.B{i:03d}/{b_cnt} loss:{loss.item():.8f}; elp:{elp}, eta:{eta}")
            # for loader
            loss_avg = loss_ttl / loss_cnt
            logging.info(f"E:{epoch}/{e_cnt}: avg_loss:{loss_avg:.8f} . ema_cnt:{ema_cnt}")
            loss_avg_arr.append(loss_avg)
            # self.scheduler.step()
            if epoch % args.save_per_epoch == 0 or epoch == e_cnt - 1:
                self.save_model(epoch)
            if test_per_epoch > 0 and (epoch % test_per_epoch == 0 or epoch == e_cnt - 1):
                self.test_and_save_result(epoch, test_loader)
        # for epoch
        utils.output_list(loss_avg_arr, 'loss_avg', ftm_str="{:10.8f}")
        utils.output_list(self.test_avg_arr, 'test_vanilla', ftm_str="{:10.8f}")
        utils.output_list(self.test_ema_arr, 'test_ema', ftm_str="{:10.8f}")

    # train(self)

    def train_model(self, x, epsilon, epoch, b_idx):
        b_sz = x.size(0)  # batch size
        if epoch == 0 and b_idx == 0:
            logging.info(f"DiffusionTrainingFast::train_model() ab = random.choice({self.ab_list})")
        ab = random.choice(self.ab_list)
        ab = torch.ones(size=(b_sz,), device=self.device) * ab
        loss = self.estimate_loss(self.model, x, ab, epsilon)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.optim.grad_clip)
        self.optimizer.step()

        ema_update_flag = 0
        if self.ema_flag and epoch >= self.ema_start_epoch:
            ema_update_flag = self.ema_helper.update(self.model)
        return loss, ema_update_flag

    @staticmethod
    def estimate_loss(model, x0: torch.Tensor, ab: torch.LongTensor, e: torch.Tensor):
        """
        Assume batch size is 250; and image size is 32*32.
        :param model:
        :param x0:  shape [250, 3, 32, 32]  x0
        :param ab:  shape [250]             alpha_bar
        :param e:   shape [250, 3, 32, 32]  epsilon
        :return:
        """
        at = ab.view(-1, 1, 1, 1)  # alpha_bar_t
        x = x0 * at.sqrt() + e * (1.0 - at).sqrt()
        output = model(x, ab)
        return (e - output).square().sum(dim=(1, 2, 3)).mean(dim=0)

    def test_and_save_result(self, epoch, test_loader):
        ts_msg = f"ts_arr_arr[{len(self.ts_arr_arr)}, {len(self.ts_arr_arr[0])}]"
        logging.info(f"E{epoch}. calculate loss on test dataset...{ts_msg}")
        self.model.eval()
        test_loss_avg = self.get_avg_loss(self.model, test_loader)
        self.model.train()
        logging.info(f"E{epoch}. test_loss_avg:{test_loss_avg:.8f}")
        self.test_avg_arr.append(test_loss_avg)
        if self.ema_flag and epoch >= self.ema_start_epoch:
            self.ema_helper.ema(self.m_ema)
            test_ema_avg = self.get_avg_loss(self.m_ema, test_loader)
            self.test_ema_arr.append(test_ema_avg)
            logging.info(f"E{epoch}. test_ema_avg :{test_ema_avg:.8f}")

    def get_avg_loss(self, model, data_loader):
        """
        By ts_arr_arr, we can make sure that the testing process is deterministic.
        The first testing round will generate the timesteps. and the consequence
        testing rounds will reuse those timesteps.
        :param model:
        :param data_loader:
        :return:
        """
        loss_ttl = 0.
        loss_cnt = 0
        with torch.no_grad():
            for ts_arr in self.ts_arr_arr:  # run multiple rounds, then get more accurate avg loss
                for i, (x, y) in enumerate(data_loader):
                    x = x.to(self.device)
                    x = data_transform(self.config, x)
                    e = torch.randn_like(x)

                    b_sz = x.size(0)  # batch size
                    if len(ts_arr) > i:
                        t = ts_arr[i]
                    else:
                        t = random.choice(self.ab_list)
                        ts_arr.append(t)
                    t = torch.ones(size=(b_sz,), device=self.device) * t
                    loss = self.estimate_loss(model, x, t, e)
                    loss_ttl += loss.item()
                    loss_cnt += 1
                # for loader
            # for idx of arr
        # with
        loss_avg = loss_ttl / loss_cnt
        return loss_avg

    def load_model(self, states):
        self.model.load_state_dict(states['model'])
        self.m_ema.load_state_dict(states['m_ema'])
        op_st = states['optimizer']
        op_st["param_groups"][0]["eps"] = self.config.optim.eps
        self.optimizer.load_state_dict(op_st)
        self.scheduler.load_state_dict(states['scheduler'])
        start_epoch = states['cur_epoch']
        self.ab_list = states['ab_list']
        logging.info(f"  load_model()...")
        logging.info(f"  resume optimizer  : eps={op_st['param_groups'][0]['eps']}")
        logging.info(f"  resume scheduler  : lr={self.scheduler.get_last_lr()[0]:8.6f}")
        logging.info(f"  resume start_epoch: {start_epoch}")
        logging.info(f"  resume ab_list    : {self.ab_list}")
        if self.ema_flag:
            self.ema_helper.register(self.model)
            self.ema_helper.load_state_dict(states['ema_helper'])
            logging.info(f"  resume ema_helper : mu={self.ema_helper.mu:8.6f}")
        return start_epoch

    def save_model(self, e_idx):
        real_model = self.model
        if isinstance(real_model, torch.nn.DataParallel):
            real_model = real_model.module
        real_m_ema = self.m_ema
        if isinstance(real_m_ema, torch.nn.DataParallel):
            real_m_ema = real_m_ema.module
        states = {
            'model': real_model.state_dict(),
            'm_ema': real_m_ema.state_dict(),
            'ab_list': self.ab_list,
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'beta_schedule': self.beta_schedule,
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
