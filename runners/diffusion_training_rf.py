"""
rf: Rectified Flow.
Borrow the idea of paper:
  Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow. ICLR 2023.
"""

import os
import time
import torch
import logging
import torch.utils.data as data
from torch.backends import cudnn

from datasets import get_dataset, data_transform
from functions import get_optimizer
from models.diffusion import Model
from models.ema import EMAHelper
from runners.diffusion import Diffusion
import utils


class DiffusionTrainingRectifiedFlow(Diffusion):
    """
    Diffusion training with rectified flow
    """
    def __init__(self, args, config, device=None, output_ab=False):
        super().__init__(args, config, device, output_ab=output_ab)
        self.model = None
        self.m_ema = None   # model for EMA
        self.optimizer = None
        self.scheduler = None
        self.ema_helper = None
        self.ema_flag = args.ema_flag
        self.ema_rate = args.ema_rate
        self.ema_start_epoch = args.ema_start_epoch
        self.ts_stride = args.ts_stride
        logging.info(f"DiffusionTrainingRectifiedFlow()")
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
        logging.info(f"  root       : {test_dataset.root}")
        logging.info(f"  split      : {test_dataset.split}") if hasattr(test_dataset, 'split') else None
        logging.info(f"  len        : {len(test_dataset)}")
        logging.info(f"  batch_cnt  : {len(test_loader)}")
        logging.info(f"  batch_size : {batch_size}")
        logging.info(f"  shuffle    : False")
        logging.info(f"  num_workers: {config.data.num_workers}")
        return train_loader, test_loader

    def init_models(self):
        args, config = self.args, self.config
        self.model = Model(config)
        self.m_ema = Model(config)
        self.m_ema.eval()  # only used for evaluation
        cnt, str_cnt = utils.count_parameters(self.model, log_fn=None)

        logging.info(f"DiffusionTrainingRectifiedFlow.model: -------------------")
        logging.info(f"  model name : {type(self.model).__name__}")
        logging.info(f"  param size : {cnt} => {str_cnt}")
        logging.info(f"  model.to({self.device})")
        self.model.to(self.device)
        self.m_ema.to(self.device)
        self.optimizer = get_optimizer(self.config, self.model.parameters(), self.args.lr)
        e_cnt = self.args.n_epochs
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=e_cnt)
        logging.info(f"DiffusionTrainingRectifiedFlow.optimizer:  -------------------")
        logging.info(f"  optimizer   : {type(self.optimizer).__name__}")
        logging.info(f"  lr          : {self.args.lr}")

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

    def loss_stat(self):
        # loss statistic
        train_loader, test_loader = self.get_data_loaders() # data loaders
        b_cnt = len(test_loader)
        self.init_models()             # models, optimizer and others
        self.ts_stride = self.args.ts_stride # re-assign again. as load_model() will change it.
        loss_avg_arr = []
        ts_list = list(range(self.ts_high, self.ts_low, -self.ts_stride))
        ts_list.reverse()
        ts_cnt = len(ts_list)
        data_start = time.time()
        logging.info(f"DiffusionTrainingRectifiedFlow::loss_stat()")
        logging.info(f"  ts_low        : {self.ts_low}")
        logging.info(f"  ts_high       : {self.ts_high}")
        logging.info(f"  ts_stride     : {self.ts_stride}")
        logging.info(f"  batch_size    : {self.args.batch_size}")
        logging.info(f"  batch_count   : {b_cnt}")
        logging.info(f"  ts_cnt        : {ts_cnt}")
        logging.info(f"  model         : self.m_ema")
        self.ema_helper.ema(self.m_ema)
        model = self.m_ema
        model.eval()
        for idx, ts_scalar in enumerate(ts_list):
            elp, eta = utils.get_time_ttl_and_eta(data_start, idx, ts_cnt)
            loss_ttl, loss_cnt = 0., 0
            with torch.no_grad():
                for bi, (x, y) in enumerate(test_loader):
                    x = x.to(self.device)
                    x = data_transform(self.config, x)
                    ts_tensor = torch.ones(x.size(0), dtype=torch.long, device=self.device)
                    ts_tensor *= ts_scalar
                    loss = self.calc_loss(model, x, ts_tensor)
                    loss_ttl += loss.item()
                    loss_cnt += 1
                # for loader
            # with
            loss_avg = loss_ttl / loss_cnt
            loss_avg_arr.append(loss_avg)
            logging.info(f"{idx:3d}/{ts_cnt} ts:{ts_scalar:4d}~{loss_avg:9.4f}; elp:{elp}, eta:{eta}")
        # for ts
        utils.output_list(loss_avg_arr, 'loss_avg')

    def train(self):
        args, config = self.args, self.config
        train_loader, test_loader = self.get_data_loaders() # data loaders
        e_cnt, start_epoch = self.init_models()             # models, optimizer and others
        log_interval = args.log_interval
        test_interval = args.test_interval
        b_cnt = len(train_loader)               # batch count
        eb_cnt = (e_cnt - start_epoch) * b_cnt  # epoch * batch
        loss_avg_arr = []
        data_start = time.time()
        self.model.train()
        logging.info(f"DiffusionTrainingRectifiedFlow::train()")
        logging.info(f"  log_interval  : {log_interval}")
        logging.info(f"  start_epoch   : {start_epoch}")
        logging.info(f"  epoch_cnt     : {e_cnt}")
        logging.info(f"  test_interval : {test_interval}")
        logging.info(f"  ts_range      : {args.ts_range}")
        logging.info(f"  ts_stride     : {self.ts_stride}")
        logging.info(f"  loss_dual     : {args.loss_dual}")
        logging.info(f"  loss_lambda   : {args.loss_lambda}")
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
                loss, loss_adj, upd_flag = self.train_model(x, epoch)
                loss_ttl += loss.item()
                loss_cnt += 1
                ema_cnt += upd_flag

                if i % log_interval == 0 or i == b_cnt - 1:
                    elp, eta = utils.get_time_ttl_and_eta(data_start, (epoch-start_epoch)*b_cnt+i, eb_cnt)
                    loss_str = f"loss:{loss.item():8.4f}"
                    if self.args.loss_dual: loss_str += f", loss_adj:{loss_adj:8.4f}"
                    logging.info(f"E{epoch}.B{i:03d}/{b_cnt} {loss_str}; elp:{elp}, eta:{eta}")
            # for loader
            loss_avg = loss_ttl / loss_cnt
            logging.info(f"E:{epoch}/{e_cnt}: avg_loss:{loss_avg:8.4f} . ema_cnt:{ema_cnt}")
            loss_avg_arr.append(loss_avg)
            # self.scheduler.step()
            if epoch % args.save_ckpt_interval == 0 or epoch == e_cnt - 1:
                self.save_model(epoch)
            if epoch == e_cnt - 1 or (test_interval > 0 and epoch > 0 and epoch % test_interval == 0):
                self.test_and_save_result(epoch, test_loader)
        # for epoch
        utils.output_list(loss_avg_arr, 'loss_avg')
        utils.output_list(self.test_avg_arr, 'test_data_avg')
        utils.output_list(self.test_ema_arr, 'test_data_ema_avg')

    def train_model(self, x_0_batch, epoch):
        config = self.config
        b_sz = x_0_batch.size(0)  # batch size
        ttl_range = self.ts_high - self.ts_low
        rdm_range = ttl_range // self.ts_stride
        t = torch.randint(low=0, high=rdm_range, size=(b_sz,), device=self.device)
        t += 1 # change range from [low, high) to (low, high]
        # ts_low is ground truth. So don't need to predict it.
        t *= self.ts_stride
        t += self.ts_low
        if not self._ts_log_flag:
            self._ts_log_flag = True
            logging.info(f"train_model() ttl_range=ts_high-ts_low: {ttl_range} = {self.ts_high} - {self.ts_low}")
            logging.info(f"train_model() rdm_range=ttl_range//ts_stride: {rdm_range} = {ttl_range} // {self.ts_stride}")
            logging.info(f"train_model() t: len:{len(t)}, t[0~1]:{t[0]} {t[1]}, t[-2~-1]:{t[-2]} {t[-1]}")
        self.optimizer.zero_grad()
        if self.args.loss_dual:
            t2 = torch.randint(low=0, high=rdm_range, size=(b_sz,), device=self.device)
            t2 += 1
            t2 *= self.ts_stride
            t2 += self.ts_low
            loss, loss_adj = self.calc_loss_dual(self.model, x_0_batch, t, t2)
            loss += loss_adj
        else:
            loss = self.calc_loss(self.model, x_0_batch, t)
            loss_adj = 0
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), config.optim.grad_clip)
        self.optimizer.step()

        ema_update_flag = 0
        if self.ema_flag and epoch >= self.ema_start_epoch:
            ema_update_flag = self.ema_helper.update(self.model)
        return loss, loss_adj, ema_update_flag

    def calc_loss(self, model, x_0_batch, t):
        epsilon = torch.randn_like(x_0_batch)
        z_0 = self.get_xt(x_0_batch, self.ts_low, epsilon)
        z_1 = self.get_xt(x_0_batch, self.ts_high, epsilon)
        target = z_1 - z_0
        cursor = torch.sub(t, self.ts_low) / (self.ts_high - self.ts_low)
        cursor = cursor.view(-1, 1, 1, 1)
        z_t = z_0 * (1 - cursor) + z_1 * cursor
        output = model(z_t, t)
        loss = (output - target).square().sum(dim=(1, 2, 3)).mean(dim=0)
        return loss

    def calc_loss_dual(self, model, x_0_batch, t1, t2):
        epsilon = torch.randn_like(x_0_batch)
        z_0 = self.get_xt(x_0_batch, self.ts_low, epsilon)
        z_1 = self.get_xt(x_0_batch, self.ts_high, epsilon)
        target = z_1 - z_0
        cursor1 = torch.sub(t1, self.ts_low) / (self.ts_high - self.ts_low)
        cursor1 = cursor1.view(-1, 1, 1, 1)
        z_t = z_0 * (1 - cursor1) + z_1 * cursor1
        output1 = model(z_t, t1)
        loss1 = (output1 - target).square().sum(dim=(1, 2, 3)).mean(dim=0)

        cursor2 = torch.sub(t2, self.ts_low) / (self.ts_high - self.ts_low)
        cursor2 = cursor2.view(-1, 1, 1, 1)
        z_t = z_0 * (1 - cursor2) + z_1 * cursor1
        output2 = model(z_t, t2)
        loss2 = (output2 - target).square().sum(dim=(1, 2, 3)).mean(dim=0)
        loss = (loss1 + loss2) / 2
        loss_adj = (output1 - output2).square().sum(dim=(1, 2, 3)).mean(dim=0)
        loss_adj *= self.args.loss_lambda
        return loss, loss_adj

    def get_xt(self, x_0_batch, ts_scalar, epsilon):
        """
        get xt from x0, epsilon and timestep.
        Pls note the timestep here has max range [0, 1000].
        If timestep is 0, just return x0.
        Else, timestep minus 1. This is to match the index of alpha_bar_list.
        :param x_0_batch:
        :param ts_scalar:
        :param epsilon:
        :return:
        """
        if ts_scalar == 0:
            return x_0_batch
        ts_scalar -= 1 # change max range from [1000, 1] to [999, 0]
        tensor_ts = torch.ones(x_0_batch.size(0)).long() * ts_scalar
        tensor_ts = tensor_ts.to(self.device)
        alpha_bar_t = self.alphas_cumprod.index_select(0, tensor_ts).view(-1, 1, 1, 1)
        # alpha_bar_t = (1000 - ts_scalar) / 1000
        # alpha_bar_t = torch.tensor(alpha_bar_t, device=self.device)
        x_t = x_0_batch * alpha_bar_t.sqrt() + torch.mul(epsilon, (1 - alpha_bar_t).sqrt())
        return x_t

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
        Also, to make sure different servers and different running use the same timesteps,
        we use the deterministic timestep for each image. For example, the timesteps for
        each round may be like:
        round 1: 0, 1, 2, 3, 4, 5, 6, 7, , , 999, 0, 1, 2, , , , ,
        round 2: 300, 301, 302, 303, , , , , 999, 0, 1, 2, , , , ,
        round 3: 600, 601, 602, 603, , , , , 999, 0, 1, 2, , , , ,
        :param model:
        :param test_loader:
        :param ts_arr_arr: timestep array
        :return:
        """
        def get_t():
            if len(ts_arr) > bi:
                return ts_arr[bi]
            ttl_range = self.ts_high - self.ts_low
            rdm_range = ttl_range // self.ts_stride
            ts = torch.arange(itr_start, itr_end, step=1, dtype=torch.int, device=self.device)
            ts %= rdm_range
            ts += 1  # change range from [low, high) to (low, high]
            # ts_low is ground truth. So don't need to predict it.
            ts *= self.ts_stride
            ts += self.ts_low
            # logging.info(f"get_avg_loss() ri:{ri}, bi:{bi:2d}, ts.len:{len(ts)} => {ts[0]:4d}~{ts[-1]:4d}")
            ts_arr.append(ts)
            return ts

        loss_ttl = 0.
        loss_cnt = 0
        with torch.no_grad():
            for ri, ts_arr in enumerate(ts_arr_arr):  # run multiple rounds, then get more accurate avg loss
                itr_end = ri * 300  # iterate ending index
                for bi, (x, y) in enumerate(test_loader):
                    x = x.to(self.device)
                    x = data_transform(self.config, x)
                    itr_start, itr_end = itr_end, itr_end + x.size(0)
                    t = get_t()
                    loss = self.calc_loss(model, x, t)
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
        logging.info(f"  load_model()...")
        logging.info(f"  resume optimizer  : eps={op_st['param_groups'][0]['eps']}")
        logging.info(f"  resume scheduler  : lr={self.scheduler.get_last_lr()[0]:8.6f}")
        logging.info(f"  resume start_epoch: {start_epoch}")
        logging.info(f"  self.ts_low       : {self.ts_low}")
        if self.ema_flag:
            self.ema_helper.load_state_dict(states['ema_helper'])
            logging.info(f"  resume ema_helper : mu={self.ema_helper.mu:8.6f}")
        self.ts_low    = states['ts_low']
        self.ts_high   = states['ts_high']
        self.ts_stride = states['ts_stride']
        logging.info(f"  self.ts_low       : {self.ts_low}")
        logging.info(f"  self.ts_high      : {self.ts_high}")
        logging.info(f"  self.ts_stride    : {self.ts_stride}")
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
            'cur_epoch'     : e_idx,
            'ts_low'        : self.ts_low,
            'ts_high'       : self.ts_high,
            'ts_stride'     : self.ts_stride,
        }
        if self.ema_flag and self.ema_helper:
            states['ema_helper'] = self.ema_helper.state_dict()

        save_ckpt_dir = self.args.save_ckpt_dir
        if not os.path.exists(save_ckpt_dir):
            logging.info(f"os.makedirs({save_ckpt_dir})")
            os.makedirs(save_ckpt_dir)
        fpath = os.path.join(save_ckpt_dir, f"ckpt_rf_E{e_idx:04d}.pth")
        logging.info(f"save ckpt dict: {fpath}")
        torch.save(states, fpath)
        fpath = os.path.join(save_ckpt_dir, f"ckpt_rf_{self.ts_low:03d}-{self.ts_high:03d}.pth")
        logging.info(f"save ckpt dict: {fpath}")
        torch.save(states, fpath)

# class
