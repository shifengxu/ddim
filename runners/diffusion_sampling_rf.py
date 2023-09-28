"""
Sampling with rectified flow
"""
import os
import sys
import time
import logging
import torch

import utils
from datasets import inverse_data_transform
from models.diffusion import Model
from models.ema import EMAHelper
from runners.diffusion import Diffusion

import torchvision.utils as tvu

class ModelWithTimestep:
    def __init__(self, model, ts_low=None, ts_high=None, ts_stride=None):
        self.model = model
        self.ts_low = ts_low
        self.ts_high = ts_high
        self.ts_stride = ts_stride
        self.ckpt_path = None
        self.index = -1 # index in mt stack

    def __call__(self, *args, **kwargs):
        return self.model(*args, **kwargs)
# class

class DiffusionSamplingRectifiedFlow(Diffusion):
    def __init__(self, args, config, device=None):
        super().__init__(args, config, device, output_ab=False)
        self.sample_count = self.args.sample_count
        self.ts_stride = args.ts_stride
        self.mt_stack = [] # ModelWithTimestep stack

    def mt_load(self, ckpt_path):
        """load ModelWithTimestep"""
        def apply_ema():
            logging.info(f"  ema_helper: EMAHelper()")
            ema_helper = EMAHelper()
            ema_helper.register(model)
            k = "ema_helper" if isinstance(states, dict) else -1
            logging.info(f"  ema_helper: load from states[{k}]")
            ema_helper.load_state_dict(states[k])
            logging.info(f"  ema_helper: apply to model {type(model).__name__}")
            ema_helper.ema(model)

        logging.info(f"load ckpt: {ckpt_path}")
        states = torch.load(ckpt_path, map_location=self.config.device)
        model = Model(self.config)
        model.load_state_dict(states['model'], strict=True)
        if not hasattr(self.args, 'ema_flag'):
            logging.info(f"  !!! Not found ema_flag in args. Assume it is true.")
            apply_ema()
        elif self.args.ema_flag:
            logging.info(f"  Found args.ema_flag: {self.args.ema_flag}.")
            apply_ema()

        logging.info(f"  model = model.to({self.device})")
        model = model.to(self.device)
        model.eval()
        if len(self.args.gpu_ids) > 1:
            logging.info(f"  torch.nn.DataParallel(model, device_ids={self.args.gpu_ids})")
            model = torch.nn.DataParallel(model, device_ids=self.args.gpu_ids)
        mt = ModelWithTimestep(model, states['ts_low'], states['ts_high'], states['ts_stride'])
        mt.ckpt_path = ckpt_path
        logging.info(f"  ts_low   : {mt.ts_low}")
        logging.info(f"  ts_high  : {mt.ts_high}")
        logging.info(f"  ts_stride: {mt.ts_stride}")
        return mt

    def init_mt_stack(self):
        ckpt_path_arr = self.get_ckpt_path_arr()
        if len(ckpt_path_arr) == 0:
            ckpt_path_arr = [self.args.sample_ckpt_path]
        for ckpt_path in ckpt_path_arr:
            mt = self.mt_load(ckpt_path)
            self.mt_stack.append(mt)
        if len(self.mt_stack) == 0:
            logging.info(f"!!! Not found any checkpoint. !!!")
            logging.info(f"sample_ckpt_path: {self.args.sample_ckpt_path}")
            logging.info(f"sample_ckpt_dir : {self.args.sample_ckpt_dir}")
            sys.exit(0)
        self.mt_stack.sort(key=lambda m: m.ts_low)
        mt_cnt = len(self.mt_stack)
        logging.info(f"Loaded {mt_cnt} ModelWithTimestep . . .")
        for i in range(mt_cnt):
            mt = self.mt_stack[i]
            mt.index = i
            logging.info(f"{i:3d} {mt.ts_low:4d} ~ {mt.ts_high:4d}, {mt.ts_stride:3d}. {mt.ckpt_path}")

    def sample(self):
        logging.info(f"DiffusionSamplingRectifiedFlow::sample()...")
        logging.info(f"  sample_output_dir : {self.args.sample_output_dir}")
        logging.info(f"  sample_ckpt_path  : {self.args.sample_ckpt_path}")
        logging.info(f"  sample_ckpt_dir   : {self.args.sample_ckpt_dir}")
        logging.info(f"  sample_count      : {self.sample_count}")
        logging.info(f"  ts_low            : {self.ts_low}")
        logging.info(f"  ts_high           : {self.ts_high}")
        logging.info(f"  ts_stride         : {self.ts_stride}")
        b_sz = self.args.sample_batch_size
        n_rounds = (self.sample_count - 1) // b_sz + 1  # get the ceiling
        logging.info(f"  batch_size        : {b_sz}")
        logging.info(f"  n_rounds          : {n_rounds}")

        self.init_mt_stack()

        time_start = time.time()
        config = self.config
        with torch.no_grad():
            for r_idx in range(n_rounds):
                n = b_sz if r_idx + 1 < n_rounds else self.sample_count - r_idx * b_sz
                logging.info(f"round: {r_idx}/{n_rounds}. to generate: {n}")
                x_t = torch.randn(  # normal distribution with mean 0 and variance 1
                    n,
                    config.data.channels,
                    config.data.image_size,
                    config.data.image_size,
                    device=self.device,
                )
                x0 = self.generalized_steps_rf(x_t, r_idx)
                self.save_images(x0, config, time_start, n_rounds, r_idx, b_sz)
            # for r_idx
        # with

    def save_images(self, x, config, time_start, n_rounds, r_idx, b_sz):
        x = inverse_data_transform(config, x)
        img_cnt = len(x)
        elp, eta = utils.get_time_ttl_and_eta(time_start, r_idx+1, n_rounds)
        img_dir = self.args.sample_output_dir
        if not os.path.exists(img_dir):
            logging.info(f"os.makedirs({img_dir})")
            os.makedirs(img_dir)
        logging.info(f"save {img_cnt} images to: {img_dir}. round:{r_idx}/{n_rounds}, elp:{elp}, eta:{eta}")
        img_first = img_last = None
        for i in range(img_cnt):
            img_id = r_idx * b_sz + i
            img_path = os.path.join(img_dir, f"{img_id:05d}.png")
            tvu.save_image(x[i], img_path)
            if i == 0: img_first = f"{img_id:05d}.png"
            if i == img_cnt - 1: img_last = f"{img_id:05d}.png"
        logging.info(f"image generated: {img_first} ~ {img_last}.")

    def find_mt(self, ts_scalar):
        """ find ModelWithTimestep """
        for mt in self.mt_stack:
            if mt.ts_low < ts_scalar <= mt.ts_high:
                return mt
        # for
        raise ValueError(f"Cannot find mt for ts_scalar {ts_scalar}")

    def generalized_steps_rf(self, x_T, r_idx=None):
        with torch.no_grad():
            b_sz = len(x_T)
            xt = x_T
            for ts_scalar in range(self.ts_high, self.ts_low, -self.ts_stride):
                t = (torch.ones(b_sz, device=self.device) * ts_scalar)
                mt = self.find_mt(ts_scalar)
                if r_idx == 0 and ts_scalar % 10 == 0:
                    logging.info(f"generalized_steps_rf(): ts_scalar={ts_scalar:4d}, mt_{mt.index}")
                grad = mt.model(xt, t)  # gradient
                delta = self.ts_stride / (mt.ts_high - mt.ts_low)
                delta = (torch.ones(b_sz, device=self.device) * delta).view(-1, 1, 1, 1)
                xt_next = xt - grad * delta
                xt = xt_next
            # for
        # with
        return xt

# class
