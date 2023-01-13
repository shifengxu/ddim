import os
import time
import torch
from torch.backends import cudnn

import utils
from albar.albar_model import AlbarModel
from datasets import get_dataset, data_transform
from functions import get_optimizer
from models.ema import EMAHelper
from runners.diffusion import Diffusion

import torch.utils.data as data
log_fn = utils.log_info

class AlbarTraining(Diffusion):
    def __init__(self, args, config, device=None):
        super().__init__(args, config, device)
        self.model = None
        self.m_ema = None   # model for EMA
        self.optimizer = None
        self.scheduler = None
        self.ema_helper = None
        self.ema_flag = args.ema_flag
        self.ema_rate = args.ema_rate or config.model.ema_rate
        self.ema_start_epoch = args.ema_start_epoch

    def train(self):
        args, config = self.args, self.config
        train_loader, test_loader = self.get_data_loaders()

        ar = args.albar_range
        self.model = AlbarModel(config, albar_range=ar, log_fn=log_fn)
        self.m_ema = AlbarModel(config, albar_range=ar, log_fn=log_fn)
        self.m_ema.eval()  # only used for evaluation
        log_fn(f"model: {type(self.model).__name__} ===================")
        log_fn(f"  config type: {config.model.type}")
        log_fn(f"  ema_flag   : {self.ema_flag}")
        log_fn(f"  ema_rate   : {self.ema_rate}")
        log_fn(f"  ema_start_epoch: {self.ema_start_epoch}")
        log_fn(f"  model.to({self.device})")
        self.model.to(self.device)
        self.m_ema.to(self.device)
        test_per_epoch = args.test_per_epoch
        e_cnt = self.args.n_epochs or self.config.training.n_epochs
        self.optimizer = get_optimizer(self.config, self.model.parameters(), self.args.lr)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=e_cnt)
        log_fn(f"optimizer: {type(self.optimizer).__name__} ===================")
        log_fn(f"  lr: {self.args.lr}")

        if self.ema_flag:
            self.ema_helper = EMAHelper(mu=self.ema_rate)
            log_fn(f"  ema_helper: EMAHelper(mu={self.ema_rate})")
        else:
            self.ema_helper = None
            log_fn(f"  ema_helper: None")

        start_epoch = 0
        if self.args.resume_training:
            ckpt_path = self.args.resume_ckpt
            log_fn(f"resume_training: {self.args.resume_training}")
            log_fn(f"resume_ckpt    : {ckpt_path}")
            states = torch.load(ckpt_path, map_location=self.device)
            start_epoch = self.load_model(states)
        log_fn(f"  torch.nn.DataParallel(device_ids={args.gpu_ids})")
        self.model = torch.nn.DataParallel(self.model, device_ids=args.gpu_ids)
        self.m_ema = torch.nn.DataParallel(self.m_ema, device_ids=args.gpu_ids)
        cudnn.benchmark = True

        b_cnt = len(train_loader)
        eb_cnt = (e_cnt - start_epoch) * b_cnt  # epoch * batch
        loss_avg_arr = []
        test_avg_arr = []  # test dataset loss avg array
        test_ema_arr = []  # test dataset loss avg array on self.m_ema (EMA model)
        data_start = time.time()
        self.model.train()
        log_interval = 10
        albar_arr_arr = [[], [], []]  # array of array, for albar array when testing
        log_fn(f"log_interval: {log_interval}")
        log_fn(f"start_epoch : {start_epoch}")
        log_fn(f"epoch_cnt   : {e_cnt}")
        for epoch in range(start_epoch, e_cnt):
            lr = self.scheduler.get_last_lr()[0]
            msg = f"lr={lr:8.7f}; ts=[{self.ts_low}, {self.ts_high}];"
            if self.ema_flag: msg += f" ema_start_epoch={self.ema_start_epoch}, ema_rate={self.ema_rate}"
            log_fn(f"Epoch {epoch}/{e_cnt} ---------- {msg}")
            if self.ema_flag and self.ema_start_epoch == epoch:
                log_fn(f"EMA register...")
                self.ema_helper.register(self.model)
            loss_ttl = 0.
            loss_cnt = 0
            ema_cnt = 0
            for i, (x, y) in enumerate(train_loader):
                x = x.to(self.device)
                x = data_transform(self.config, x)
                e = torch.randn_like(x)

                # antithetic sampling
                loss, upd_flag = self.train_model(x, e, epoch)
                loss_ttl += loss.item()
                loss_cnt += 1
                ema_cnt += upd_flag

                if i % log_interval == 0 or i == b_cnt - 1:
                    elp, eta = utils.get_time_ttl_and_eta(data_start, (epoch-start_epoch) * b_cnt + i, eb_cnt)
                    log_fn(f"E{epoch}.B{i:03d}/{b_cnt} loss:{loss.item():8.4f}; elp:{elp}, eta:{eta}")

            # for loader
            loss_avg = loss_ttl / loss_cnt
            log_fn(f"E:{epoch}/{e_cnt}: avg_loss:{loss_avg:8.4f} . ema_cnt:{ema_cnt}")
            loss_avg_arr.append(loss_avg)
            # self.scheduler.step()
            if epoch % 100 == 0 or epoch == e_cnt - 1:
                self.save_model(epoch)
            if test_per_epoch > 0 and (epoch % test_per_epoch == 0 or epoch == e_cnt - 1):
                ts_msg = f"albar_arr_arr[{len(albar_arr_arr)}, {len(albar_arr_arr[0])}]"
                log_fn(f"E{epoch}. calculate loss on test dataset...{ts_msg}")
                self.model.eval()
                test_loss_avg = self.get_avg_loss(self.model, test_loader, albar_arr_arr)
                self.model.train()
                log_fn(f"E{epoch}. test_loss_avg:{test_loss_avg:8.4f}")
                test_avg_arr.append(test_loss_avg)
                if self.ema_flag and epoch >= self.ema_start_epoch:
                    self.ema_helper.ema(self.m_ema)
                    test_ema_avg = self.get_avg_loss(self.m_ema, test_loader, albar_arr_arr)
                    test_ema_arr.append(test_ema_avg)
                    log_fn(f"E{epoch}. test_ema_avg :{test_ema_avg:8.4f}")
        # for epoch
        utils.output_list(loss_avg_arr, 'loss_avg')
        utils.output_list(test_avg_arr, 'test_avg')
        utils.output_list(test_ema_arr, 'test_ema')
    # train(self)

    def get_data_loaders(self):
        args, config = self.args, self.config
        dataset, test_dataset = get_dataset(args, config)
        batch_size = args.batch_size or config.training.batch_size
        train_loader = data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=config.data.num_workers,
        )
        log_fn(f"train dataset and data_loader:")
        log_fn(f"  root : {dataset.root}")
        log_fn(f"  split: {dataset.split}") if hasattr(dataset, 'split') else None
        log_fn(f"  len  : {len(dataset)}")
        log_fn(f"  batch_cnt  : {len(train_loader)}")
        log_fn(f"  batch_size : {batch_size}")
        log_fn(f"  shuffle    : True")
        log_fn(f"  num_workers: {config.data.num_workers}")

        test_per_epoch = args.test_per_epoch
        test_loader = data.DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=config.data.num_workers,
        )
        log_fn(f"test dataset and data loader:")
        log_fn(f"  root : {test_dataset.root}")
        log_fn(f"  len  : {len(test_dataset)}")
        log_fn(f"  test_per_epoch: {test_per_epoch}")
        log_fn(f"  test_data_dir : {args.test_data_dir}")
        log_fn(f"  batch_cnt     : {len(test_loader)}")
        log_fn(f"  batch_size    : {batch_size}")
        log_fn(f"  shuffle       : False")
        log_fn(f"  num_workers   : {config.data.num_workers}")
        return train_loader, test_loader

    def get_avg_loss(self, model, test_loader, albar_arr_arr):
        """
        By albar_arr_arr, we can make sure that the testing process is deterministic.
        The first testing round will generate the timesteps. and the consequence
        testing rounds will reuse those timesteps.
        :param model:
        :param test_loader:
        :param albar_arr_arr: albar array of array
        :return:
        """
        loss_ttl = 0.
        loss_cnt = 0
        with torch.no_grad():
            for albar_arr in albar_arr_arr:  # run multiple rounds, then get more accurate avg loss
                for i, (x, y) in enumerate(test_loader):
                    x = x.to(self.device)
                    x = data_transform(self.config, x)
                    e = torch.randn_like(x)

                    b_sz = x.size(0)  # batch size
                    if len(albar_arr) > i:
                        albar = albar_arr[i]
                    else:
                        t = torch.randint(low=self.ts_low, high=self.ts_high, size=(b_sz,), device=self.device)
                        albar = self.ts_to_albar(t)
                        albar_arr.append(albar)
                    albar4d = albar.view(-1, 1, 1, 1)
                    xt = x * albar4d.sqrt() + (1.0 - albar4d).sqrt() * e
                    output = model(xt, albar)
                    loss = (output - e).square().sum(dim=(1, 2, 3)).mean(dim=0)
                    loss_ttl += loss.item()
                    loss_cnt += 1
                # for loader
            # for idx of arr
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
        b_sz = x.size(0)  # batch size
        if not self._ts_log_flag:
            self._ts_log_flag = True
            log_fn(f"AlbarTraining::train_model() timestep: torch.randint(low={self.ts_low}, "
                   f"high={self.ts_high}, size=({b_sz},), device={self.device})")
        t = torch.randint(low=self.ts_low, high=self.ts_high, size=(b_sz,), device=self.device)
        albar = self.ts_to_albar(t)
        albar4d = albar.view(-1, 1, 1, 1)
        xt = x * albar4d.sqrt() + (1.0 - albar4d).sqrt() * epsilon
        output = self.model(xt, albar)
        loss = (output - epsilon).square().sum(dim=(1, 2, 3)).mean(dim=0)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.optim.grad_clip)
        self.optimizer.step()

        ema_update_flag = 0
        if self.ema_flag and epoch >= self.ema_start_epoch:
            ema_update_flag = self.ema_helper.update(self.model)
        return loss, ema_update_flag

    def load_model(self, states):
        self.model.load_state_dict(states['model'])
        op_st = states['optimizer']
        op_st["param_groups"][0]["eps"] = self.config.optim.eps
        self.optimizer.load_state_dict(op_st)
        self.scheduler.load_state_dict(states['scheduler'])
        start_epoch = states['cur_epoch']
        log_fn(f"  load_model_dict()...")
        log_fn(f"  resume optimizer  : eps={op_st['param_groups'][0]['eps']}")
        log_fn(f"  resume scheduler  : lr={self.scheduler.get_last_lr()[0]:8.6f}")
        log_fn(f"  resume start_epoch: {start_epoch}")
        if self.ema_flag:
            self.ema_helper.load_state_dict(states['ema_helper'])
            log_fn(f"  resume ema_helper : mu={self.ema_helper.mu:8.6f}")
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
            'cur_epoch': e_idx
        }
        if self.ema_flag and self.ema_helper:
            states['ema_helper'] = self.ema_helper.state_dict()

        fpath = os.path.join(self.args.result_dir, f"ckpt_E{e_idx:04d}.pth")
        log_fn(f"save ckpt dict: {fpath}")
        torch.save(states, fpath)
        fpath = os.path.join(self.args.result_dir, f"ckpt_{self.ts_low:03d}-{self.ts_high:03d}.pth")
        log_fn(f"save ckpt dict: {fpath}")
        torch.save(states, fpath)

# class
