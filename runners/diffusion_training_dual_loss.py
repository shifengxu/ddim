import logging
import os
import time

import torch
import torchvision.utils as tvu
from torch.backends import cudnn

import utils
from datasets import get_dataset, data_transform, inverse_data_transform
from functions import get_optimizer
from models.diffusion import Model
from models.ema import EMAHelper
from runners.diffusion import Diffusion

import torch.utils.data as data

from utils import count_parameters, log_info


def calc_fid(gpu, input1, input2, logger=log_info):
    import subprocess
    import re
    cmd = f"fidelity --gpu {gpu} --fid --input1 {input1} --input2 {input2} --silent"
    logger(f"cmd: {cmd}")
    cmd_arr = cmd.split(' ')
    res = subprocess.run(cmd_arr, stdout=subprocess.PIPE)
    output = str(res.stdout)
    # inception_score_mean: 11.24431
    # inception_score_std: 0.09522244
    # frechet_inception_distance: 71.2743
    logger(f"out: {output}")
    m = re.search(r'frechet_inception_distance: (\d+\.\d+)', output)
    fid = float(m.group(1))
    return fid

class DiffusionTrainingDualLoss(Diffusion):
    def __init__(self, args, config):
        logging.info(f"DiffusionTrainingDualLoss::__init__()...")
        super().__init__(args, config, device=config.device, output_ab=False)
        self.model = None
        self.m_ema = None   # model for EMA
        self.optimizer = None
        self.ema_helper = None
        self.ema_rate = args.ema_rate
        self.ema_start_epoch = args.ema_start_epoch

        self.batch_total = 0
        self.batch_counter = 0
        self.result_arr = []

        # full alpha_bar list, which includes the heading "1.0"
        if len(self.alphas_cumprod) == self.num_timesteps:
            arr = [torch.ones(1).to(self.device), self.alphas_cumprod]
            self.full_ab_list = torch.cat(arr, dim=0)
        else:
            self.full_ab_list = self.alphas_cumprod
        logging.info(f"ema_rate       : {self.ema_rate}")
        logging.info(f"ema_start_epoch: {self.ema_start_epoch}")
        logging.info(f"full_ab_list   : {len(self.full_ab_list)}")
        logging.info(f"alphas_cumprod : {len(self.alphas_cumprod)}")
        logging.info(f"DiffusionTrainingDualLoss::__init__()... Done")

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
        cnt, str_cnt = count_parameters(self.model, log_fn=None)
        model_name = type(self.model).__name__

        logging.info(f"model: {model_name} ===================")
        logging.info(f"  config type: {config.model.type}")
        logging.info(f"  param size : {cnt} => {str_cnt}")
        logging.info(f"  ema_rate   : {self.ema_rate}")
        logging.info(f"  ema_start_epoch: {self.ema_start_epoch}")
        logging.info(f"  stack_sz   : {self.model.stack_sz}") if model_name == 'ModelStack' else None
        logging.info(f"  ts_cnt     : {self.model.ts_cnt}") if model_name == 'ModelStack' else None
        logging.info(f"  brick_cvg  : {self.model.brick_cvg}") if model_name == 'ModelStack' else None
        logging.info(f"  model.to({self.device})")
        self.model.to(self.device)
        self.m_ema.to(self.device)
        self.optimizer = get_optimizer(self.config, self.model.parameters(), self.args.lr)
        logging.info(f"optimizer: {type(self.optimizer).__name__} ===================")
        logging.info(f"  lr: {self.args.lr}")

        self.ema_helper = EMAHelper(mu=self.ema_rate)
        logging.info(f"  ema_helper: EMAHelper(mu={self.ema_rate})")

        logging.info(f"  torch.nn.DataParallel(device_ids={args.gpu_ids})")
        self.model = torch.nn.DataParallel(self.model, device_ids=args.gpu_ids)
        self.m_ema = torch.nn.DataParallel(self.m_ema, device_ids=args.gpu_ids)
        cudnn.benchmark = True

    def train(self):
        args, config = self.args, self.config
        train_loader, test_loader = self.get_data_loaders() # data loaders
        self.init_models()             # models, optimizer and others
        log_itv  = args.log_interval
        save_itv = args.save_ckpt_interval
        save_eva = args.save_ckpt_eval
        ema_se   = self.ema_start_epoch
        ema_rate = self.ema_rate
        if ema_se == 0: ema_se = 1  # our epoch number starts from 1
        lr = args.lr
        e_cnt = args.n_epochs
        b_cnt = len(train_loader)         # batch count
        self.batch_total = e_cnt * b_cnt  # epoch * batch
        logging.info(f"DiffusionTrainingDualLoss::train()...")
        logging.info(f"log_itv  : {log_itv}")
        logging.info(f"save_itv : {save_itv}")
        logging.info(f"save_eva : {save_eva}")
        logging.info(f"e_cnt    : {e_cnt}")
        logging.info(f"b_cnt    : {b_cnt}")
        logging.info(f"ema_se   : {ema_se}")
        logging.info(f"ema_rate : {ema_rate}")
        logging.info(f"loss_dual: {args.loss_dual}")
        logging.info(f"l_lambda : {args.loss_lambda}")
        s_time = time.time()
        self.model.train()
        for epoch in range(1, e_cnt+1):
            msg = f"lr={lr:8.7f}; ema_start_epoch={ema_se}, ema_rate={ema_rate}"
            logging.info(f"Epoch {epoch}/{e_cnt} ---------- {msg}")
            if ema_se == epoch:
                logging.info(f"EMA register...")
                self.ema_helper.register(self.model)
            loss_ttl, loss_cnt, ema_cnt = 0., 0, 0
            for b_idx, (x, y) in enumerate(train_loader):
                self.batch_counter += 1
                x = x.to(self.device)
                x = data_transform(self.config, x)
                loss, loss_adj, upd_flag = self.train_batch(x, epoch, b_idx)
                loss_ttl += loss
                loss_cnt += 1
                ema_cnt += upd_flag

                if b_idx % log_itv == 0 or b_idx == b_cnt - 1:
                    elp, eta = utils.get_time_ttl_and_eta(s_time, self.batch_counter, self.batch_total)
                    logging.info(f"B{b_idx:03d}/{b_cnt} loss:{loss:.6f}, {loss_adj:.6f}; elp:{elp}, eta:{eta}")
            # for loader
            loss_avg = loss_ttl / loss_cnt
            logging.info(f"E:{epoch}/{e_cnt}: avg_loss:{loss_avg:.6f} . ema_cnt:{ema_cnt}")
            if epoch % save_itv == 0 or epoch == e_cnt:
                self.save_model(epoch)
                if save_eva: self.ema_sample_and_fid(epoch)
        # for epoch
        f_path = f"./sample_fid_all.txt"
        with open(f_path, 'w') as fptr:
            [fptr.write(f"{m}\n") for m in self.result_arr]
        # with

    def train_batch(self, x, epoch, b_idx):
        loss_dual, loss_lambda = self.args.loss_dual, self.args.loss_lambda
        grad_clip = self.config.optim.grad_clip
        b_sz = x.size(0)  # batch size
        if b_idx == 0 and epoch == 1:
            logging.info(f"train_model() ts:({self.ts_low}, {self.ts_high}, size=({b_sz},))")
        if loss_dual:
            loss, loss_adj = self.calc_loss_dual(x)
            loss_sum = loss + loss_adj * self.args.loss_lambda
        else:
            loss_sum = loss = self.calc_loss(x)
            loss_adj = torch.tensor(0.)
        self.optimizer.zero_grad()
        loss_sum.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
        self.optimizer.step()

        ema_update_flag = 0
        if epoch >= self.ema_start_epoch:
            ema_update_flag = self.ema_helper.update(self.model)
        return loss.item(), loss_adj.item(), ema_update_flag

    def calc_loss_dual(self, x0: torch.Tensor):
        b_sz = len(x0)
        e = torch.randn_like(x0)
        a = self.alphas_cumprod
        t1 = torch.randint(low=self.ts_low, high=self.ts_high, size=(b_sz,), device=self.device)
        at1 = a.index_select(0, t1).view(-1, 1, 1, 1)  # alpha_t
        x1 = x0 * at1.sqrt() + e * (1.0 - at1).sqrt()
        output1 = self.model(x1, t1.float())
        loss1 = (e - output1).square().mean()

        t2 = torch.randint(low=self.ts_low, high=self.ts_high, size=(b_sz,), device=self.device)
        at2 = a.index_select(0, t2).view(-1, 1, 1, 1)  # alpha_t
        x2 = x0 * at2.sqrt() + e * (1.0 - at2).sqrt()
        output2 = self.model(x2, t2.float())
        loss2 = (e - output2).square().mean()

        loss = (loss1 + loss2) / 2.
        loss_adj = (output1 - output2).square().mean()
        return loss, loss_adj

    def calc_loss(self, x0: torch.Tensor):
        """
        Assume batch size is 250; and image size is 32*32.
        :param x0:  shape [250, 3, 32, 32]  x0
        :return:
        """
        b_sz = len(x0)
        t = torch.randint(low=self.ts_low, high=self.ts_high, size=(b_sz,), device=self.device)
        e = torch.randn_like(x0)
        a = self.alphas_cumprod
        at = a.index_select(0, t).view(-1, 1, 1, 1)  # alpha_t
        x = x0 * at.sqrt() + e * (1.0 - at).sqrt()
        output = self.model(x, t.float())
        return (e - output).square().mean()

    def ema_sample_and_fid(self, epoch, apply_ema=True):
        logging.info(f"get_ema_fid()")
        args, config = self.args, self.config
        if apply_ema:
            self.ema_helper.ema(self.m_ema)
        img_cnt     = args.sample_count
        b_sz        = args.sample_batch_size
        steps_arr   = args.sample_steps_arr
        init_ts_arr = args.sample_init_ts_arr
        b_cnt = img_cnt // b_sz
        if b_cnt * b_sz < img_cnt:
            b_cnt += 1
        c_data = config.data
        c, h, w = c_data.channels, c_data.image_size, c_data.image_size
        s_fid1, s_dir = args.fid_input1, args.sample_output_dir
        logging.info(f"  epoch      : {epoch}")
        logging.info(f"  img_cnt    : {img_cnt}")
        logging.info(f"  b_sz       : {b_sz}")
        logging.info(f"  b_cnt      : {b_cnt}")
        logging.info(f"  c          : {c}")
        logging.info(f"  h          : {h}")
        logging.info(f"  w          : {w}")
        logging.info(f"  steps_arr  : {steps_arr}")
        logging.info(f"  init_ts_arr: {init_ts_arr}")
        msg_arr = []
        for init_ts in init_ts_arr:
            for steps in steps_arr:
                self.sample(epoch, steps, init_ts, s_dir)
                logging.info(f"fid_input1       : {s_fid1}")
                logging.info(f"sample_output_dir: {s_dir}")
                fid = calc_fid(args.gpu_ids[0], s_fid1, s_dir)
                msg = f"E{epoch:04d}_steps{steps:02d}_initTS{init_ts:.3f}\tFID{fid:7.3f}"
                logging.info(msg)
                msg_arr.append(msg)
            # for
        # for
        basename = os.path.basename(args.save_ckpt_path)
        stem, ext = os.path.splitext(basename)
        f_path = f"./sample_fid_{stem}_E{epoch:04d}.txt"
        with open(f_path, 'w') as fptr:
            [fptr.write(f"{m}\n") for m in msg_arr]
        # with
        self.result_arr.extend(msg_arr)

    def sample(self, epoch, steps, init_ts, img_dir):
        config = self.config
        ch, h, w = config.data.channels, config.data.image_size, config.data.image_size
        img_cnt = self.args.sample_count
        b_sz    = self.args.sample_batch_size
        b_cnt   = img_cnt // b_sz  # get the ceiling
        if b_sz * b_cnt < img_cnt: b_cnt += 1
        logging.info(f"DiffusionTrainingDualLoss::sample(epoch={epoch})...---")
        logging.info(f"  steps      : {steps}")
        logging.info(f"  init_ts    : {init_ts}")
        logging.info(f"  img_dir    : {img_dir}")
        logging.info(f"  img_cnt    : {img_cnt}")
        logging.info(f"  batch_size : {b_sz}")
        logging.info(f"  b_cnt      : {b_cnt}")
        with torch.no_grad():
            for b_idx in range(b_cnt):
                n = b_sz if b_idx + 1 < b_cnt else img_cnt - b_idx * b_sz
                logging.info(f"round: {b_idx}/{b_cnt}. to generate: {n}")
                x_t = torch.randn(n, ch, h, w, device=config.device)
                x_0 = self.sample_batch(x_t, steps, init_ts, b_idx)
                self.save_sample_batch(x_0, b_idx, b_sz, img_dir)
            # for r_idx
        # with

    def sample_batch(self, x_T, steps, init_ts, b_idx):
        skip = init_ts // steps
        seq = list(range(init_ts, 0, -skip))    # 1000 ~ 1
        seq = [s - 1 for s in seq]              # 999  ~ 0  -> this is timesteps when training
        if b_idx == 0:
            ab_len = len(self.alphas_cumprod)
            logging.info(f"sample_batch()seq=[{seq[0]}~{seq[-1]}], len={len(seq)}. ab_len:{ab_len}")
        b_sz = len(x_T)
        xt = x_T
        seq2 = seq[1:] + [-1]
        for i, j in zip(seq, seq2):
            at = self.full_ab_list[i+1]  # alpha_bar_t
            aq = self.full_ab_list[j+1]  # alpha_bar_{t-1}
            mt = at / aq
            t = (torch.ones(b_sz, device=self.device) * i)
            et = self.m_ema(xt, t)  # epsilon_t
            if b_idx == 0:
                log_info(f"sample_batch() ts={i:03d}, ab:{at:.6f}, mt:{mt:.6f}")
            xt_next = (xt - (1 - at).sqrt() * et) / mt.sqrt() + (1 - aq).sqrt() * et
            xt = xt_next
        # for
        return xt

    def save_sample_batch(self, x, b_idx, b_sz, img_dir):
        if not os.path.exists(img_dir):
            logging.info(f"os.makedirs({img_dir})")
            os.makedirs(img_dir)
        x = inverse_data_transform(self.config, x)
        img_cnt = len(x)
        img_path = None
        for i in range(img_cnt):
            img_id = b_idx * b_sz + i
            img_path = os.path.join(img_dir, f"{img_id:05d}.png")
            tvu.save_image(x[i], img_path)
        logging.info(f"Saved {img_cnt}: {img_path}")

    def load_model(self, states):
        self.model.load_state_dict(states['model'])
        op_st = states['optimizer']
        op_st["param_groups"][0]["eps"] = self.config.optim.eps
        self.optimizer.load_state_dict(op_st)
        self.ema_helper.load_state_dict(states['ema_helper'])
        start_epoch = states['cur_epoch']
        logging.info(f"  load_model_dict()...")
        logging.info(f"  resume optimizer  : eps={op_st['param_groups'][0]['eps']}")
        logging.info(f"  resume start_epoch: {start_epoch}")
        logging.info(f"  resume ema_helper : mu={self.ema_helper.mu:8.6f}")
        return start_epoch

    def save_model(self, e_idx):
        real_model = self.model
        if isinstance(real_model, torch.nn.DataParallel):
            real_model = real_model.module
        states = {
            'model'         : real_model.state_dict(),
            'optimizer'     : self.optimizer.state_dict(),
            'beta_schedule' : self.beta_schedule,
            'ema_helper'    : self.ema_helper.state_dict(),
            'cur_epoch'     : e_idx,
        }
        fpath = self.args.save_ckpt_path
        s_dir, s_name = os.path.split(fpath)
        if not os.path.exists(s_dir):
            os.makedirs(s_dir)
        stem, ext = os.path.splitext(s_name)
        fpath = os.path.join(s_dir, f"{stem}_{e_idx:04d}{ext}")
        logging.info(f"save ckpt dict: {fpath}")
        torch.save(states, fpath)

    def load_ckpt(self, ckpt_path, eval_mode=True):
        def apply_ema():
            log_info(f"  ema_helper: EMAHelper()")
            eh = EMAHelper(model)
            k = "ema_helper" if isinstance(states, dict) else -1
            log_info(f"  ema_helper: load from states[{k}]")
            eh.load_state_dict(states[k])
            log_info(f"  ema_helper: apply to model {type(model).__name__}")
            eh.ema_to_module(model)

        model = Model(self.config)
        log_info(f"load ckpt: {ckpt_path} . . .")
        states = torch.load(ckpt_path, map_location=self.device)
        if 'model' not in states:
            log_info(f"  !!! Not found 'model' in states. Will take it as pure model")
            log_info(f"  model.load_state_dict(states)")
            model.load_state_dict(states)
        else:
            key = 'model' if isinstance(states, dict) else 0
            log_info(f"  load_model_dict(states[{key}])...")
            model.load_state_dict(states[key], strict=True)
            if eval_mode:
                apply_ema()
        # endif
        log_info(f"  model({type(model).__name__}).to({self.device})")
        model = model.to(self.device)
        if len(self.args.gpu_ids) > 1:
            log_info(f"  torch.nn.DataParallel(model, device_ids={self.args.gpu_ids})")
            model = torch.nn.DataParallel(model, device_ids=self.args.gpu_ids)
        if eval_mode:
            model.eval()

        log_info(f"load ckpt: {ckpt_path} . . . Done")
        return model

# class
