import logging
import os
import time

import torch
from torch.backends import cudnn

from datasets import get_dataset, data_transform
from functions import get_optimizer
from functions.losses import loss_registry
from models.diffusion import Model
from models.ema import EMAHelper
from runners.diffusion import Diffusion

import torch.utils.data as data

from utils import count_parameters


class DiffusionTraining(Diffusion):
    def __init__(self, args, config, device=None):
        super().__init__(args, config, device)

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
        model = Model(config)
        # model = ModelStack(config)
        cnt, str_cnt = count_parameters(model, log_fn=None)
        model_name = type(model).__name__

        logging.info(f"model: {model_name} ===================")
        logging.info(f"  config type: {config.model.type}")
        logging.info(f"  param size : {cnt} => {str_cnt}")
        logging.info(f"  stack_sz   : {model.stack_sz}") if model_name == 'ModelStack' else None
        logging.info(f"  ts_cnt     : {model.ts_cnt}") if model_name == 'ModelStack' else None
        logging.info(f"  brick_cvg  : {model.brick_cvg}") if model_name == 'ModelStack' else None
        logging.info(f"  model.to({self.device})")
        model = model.to(self.device)
        optimizer = get_optimizer(self.config, model.parameters())

        if self.config.model.ema:
            ema_helper = EMAHelper(mu=self.config.model.ema_rate)
            ema_helper.register(model)
        else:
            ema_helper = None

        start_epoch = 0
        if self.args.resume_training:
            ckpt_path = os.path.join(self.args.log_path, "ckpt.pth")
            logging.info(f"resume_training: {ckpt_path}")
            states = torch.load(ckpt_path, map_location=self.device)
            model.load_state_dict(states[0])

            states[1]["param_groups"][0]["eps"] = self.config.optim.eps
            optimizer.load_state_dict(states[1])
            start_epoch = states[2]
            logging.info(f"  resume optimizer  : eps={states[1]['param_groups'][0]['eps']}")
            logging.info(f"  resume start_epoch: {start_epoch}")
            if self.config.model.ema:
                ema_helper.load_state_dict(states[-1])
                logging.info(f"  resume ema_helper : ...")
        logging.info(f"  torch.nn.DataParallel(device_ids={args.gpu_ids})")
        model = torch.nn.DataParallel(model, device_ids=args.gpu_ids)
        cudnn.benchmark = True
        logging.info(f"ema_helper: {type(ema_helper).__name__}")

        e_cnt = self.config.training.n_epochs
        b_cnt = len(train_loader)
        eb_cnt = e_cnt * b_cnt  # epoch * batch
        data_start = time.time()
        model.train()
        log_interval = 1 if model_name == 'ModelStack' else 5
        logging.info(f"log_interval: {log_interval}")
        for epoch in range(start_epoch, self.config.training.n_epochs):
            for i, (x, y) in enumerate(train_loader):
                x = x.to(self.device)
                x = data_transform(self.config, x)
                e = torch.randn_like(x)
                b = self.betas

                # antithetic sampling
                if model_name == 'ModelStack':
                    if epoch == 0 and i == 0: logging.info(f"train_model_stack()...")
                    loss = self.train_model_stack(model, optimizer, ema_helper, x, e, b)
                else:
                    if epoch == 0 and i == 0: logging.info(f"train_model(ts=[{self.ts_low}, {self.ts_high}])...")
                    loss = self.train_model(model, optimizer, ema_helper, x, e, b)

                if i % log_interval == 0 or i == b_cnt - 1:
                    elp, eta = self.get_time_ttl_and_eta(data_start, epoch * b_cnt + i, eb_cnt)
                    logging.info(f"E:{epoch}/{e_cnt}, B:{i:02d}/{b_cnt}, loss:{loss.item():8.4f}; elp:{elp}; eta:{eta}")

                step = epoch * b_cnt + i + 1
                tb_logger.add_scalar("loss", loss, global_step=step)
                if step % self.config.training.snapshot_freq == 0 or step == 1 \
                        or epoch == e_cnt - 1 and i == b_cnt - 1:
                    self.save_model(epoch, i, model, optimizer, ema_helper)
            # for
        # for

    def train_model(self, model, optimizer, ema_helper, x, e, b):
        config = self.config
        b_sz = x.size(0)  # batch size
        t = torch.randint(high=self.num_timesteps, size=(b_sz // 2 + 1,), device=self.device)
        t = torch.cat([t, self.num_timesteps - t - 1], dim=0)[:b_sz]
        # t = torch.randint(low=self.ts_low, high=self.ts_high, size=(b_sz,), device=self.device)
        loss = loss_registry[config.model.type](model, x, t, e, b)
        optimizer.zero_grad()
        loss.backward()
        try:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.optim.grad_clip)
        except Exception:
            pass
        optimizer.step()
        if self.config.model.ema:
            ema_helper.update(model)
        return loss

    def train_model_stack(self, model, optimizer, ema_helper, x, e, b):
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
            loss = loss_registry[config.model.type](model, x, t, e, b)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.optim.grad_clip)
            optimizer.step()
            if self.config.model.ema:
                ema_helper.update(model)
            total_loss += loss
        avg_loss = total_loss / stack_sz
        return avg_loss

    def save_model(self, e_idx, b_idx, model, optimizer, ema_helper):
        real_model = model.module if isinstance(model, torch.nn.DataParallel) else model
        states = [
            real_model.state_dict(),
            optimizer.state_dict(),
            e_idx,
        ]
        if self.config.model.ema:
            states.append(ema_helper.state_dict())

        fpath = os.path.join(self.args.log_path, f"ckpt_E{e_idx:04d}_B{b_idx:04d}.pth")
        logging.info(f"save: {fpath}")
        torch.save(states, fpath)
        fpath = os.path.join(self.args.log_path, "ckpt.pth")
        logging.info(f"save: {fpath}")
        torch.save(states, fpath)

# class
