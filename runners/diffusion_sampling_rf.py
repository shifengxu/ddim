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

    def __call__(self, x, ts, *args, **kwargs):
        if not torch.is_tensor(ts) or len(ts.size()) == 0:
            # scalar or single element tensor
            ts = torch.ones(x.size(0), device=x.device) * ts
        return self.model(x, ts, *args, **kwargs)
# class

class DiffusionSamplingRectifiedFlow(Diffusion):
    def __init__(self, args, config, device=None):
        super().__init__(args, config, device, output_ab=False)
        self.sample_count = self.args.sample_count
        self.ts_stride = args.ts_stride
        self.mt_stack = [] # ModelWithTimestep stack
        self.predefined_ts_file = args.predefined_ts_file
        self.predefined_ts_geometric = args.predefined_ts_geometric

        # timestep array. make sure its last element is 0 ot ts_low, as x_0 is generated result
        self.ts_arr = None

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

    def sample(self, ts_arr=None):
        config = self.config
        self.init_mt_stack()

        if ts_arr is not None:
            self.ts_arr = ts_arr
        elif self.predefined_ts_geometric:
            logging.info(f"Init ts_arr by predefined_ts_geometric: {self.predefined_ts_geometric}")
            ratio = float(self.predefined_ts_geometric)
            self.ts_arr = utils.create_geometric_series(0., 940., ratio, 11)
            self.ts_arr.reverse()
        elif self.predefined_ts_file:
            logging.info(f"Init ts_arr by predefined_ts_file: {self.predefined_ts_file}")
            self.ts_arr = self.load_predefined_ts_file()
        else:
            self.ts_arr = list(range(self.ts_high, self.ts_low, -self.ts_stride))
            self.ts_arr.append(self.ts_low)
        b_sz = self.args.sample_batch_size
        n_rounds = (self.sample_count - 1) // b_sz + 1  # get the ceiling
        logging.info(f"DiffusionSamplingRectifiedFlow::sample()...")
        logging.info(f"  predefined_ts_geo : {self.predefined_ts_geometric}")
        logging.info(f"  predefined_ts_file: {self.predefined_ts_file}")
        logging.info(f"  sample_order      : {self.args.sample_order}")
        logging.info(f"  sample_output_dir : {self.args.sample_output_dir}")
        logging.info(f"  sample_ckpt_path  : {self.args.sample_ckpt_path}")
        logging.info(f"  sample_ckpt_dir   : {self.args.sample_ckpt_dir}")
        logging.info(f"  sample_count      : {self.sample_count}")
        logging.info(f"  ts_low            : {self.ts_low}")
        logging.info(f"  ts_high           : {self.ts_high}")
        logging.info(f"  ts_stride         : {self.ts_stride}")
        logging.info(f"  batch_size        : {b_sz}")
        logging.info(f"  n_rounds          : {n_rounds}")
        time_start = time.time()
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
                if self.args.sample_order == 1:
                    x0 = self.generalized_steps_rf(x_t, r_idx)
                elif self.args.sample_order == 2:
                    x0 = self.generalized_order2_rf(x_t, r_idx)
                elif self.args.sample_order == 3:
                    x0 = self.generalized_order3_rf(x_t, r_idx)
                else:
                    raise ValueError(f"Invalid sample_order: {self.args.sample_order}")
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
        """ sampling by ts_arr (timestep_arr) """
        ts_arr = self.ts_arr
        xt = x_T
        ts_idx, last_idx = 0, len(ts_arr) - 1
        while ts_idx < last_idx:
            ts_a, ts_b = ts_arr[ts_idx:ts_idx+2]
            mt = self.find_mt(ts_a)
            if r_idx == 0:
                logging.info(f"generalized_steps_rf()mt_{mt.index}: ts[{ts_idx:2d}]:{ts_a:7.2f} {ts_b:7.2f}")
            xt = self.sample_by_phase1(ts_a, ts_b, xt, mt)
            ts_idx += 1
        # for
        return xt

    def generalized_order2_rf(self, x_T, r_idx=None):
        ts_arr = self.ts_arr
        xt = x_T
        ts_idx, last_idx = 0, len(ts_arr) - 1
        while ts_idx < last_idx:
            if ts_idx + 2 > last_idx:
                break
            ts_a, ts_b, ts_c = ts_arr[ts_idx:ts_idx+3]
            mt = self.find_mt(ts_a)
            if r_idx == 0:
                logging.info(f"order2_rf()mt_{mt.index}: ts[{ts_idx:2d}]:{ts_a:7.2f} {ts_b:7.2f} {ts_c:7.2f}")
            xt = self.sample_by_phase2(ts_a, ts_b, ts_c, xt, mt)
            ts_idx += 2
        # while
        if ts_idx < last_idx:
            # now ts_idx + 1 == last_idx
            ts_a, ts_b = ts_arr[ts_idx:ts_idx+2]
            mt = self.find_mt(ts_a)
            if r_idx == 0:
                logging.info(f"order2_rf()mt_{mt.index}: ts[{ts_idx:2d}]:{ts_a:7.2f} {ts_b:7.2f}")
            xt = self.sample_by_phase1(ts_a, ts_b, xt, mt)
            ts_idx += 1
        # for
        return xt

    def generalized_order3_rf(self, x_T, r_idx=None):
        t_arr = self.ts_arr
        xt = x_T
        t_idx, last_idx = 0, len(t_arr) - 1
        while t_idx < last_idx:
            if t_idx + 4 > last_idx:
                break
            t_a, t_b, t_c, t_d = t_arr[t_idx:t_idx+4]
            mt = self.find_mt(t_a)
            if r_idx == 0:
                logging.info(f"order3_rf()mt_{mt.index}: ts[{t_idx:2d}]:{t_a:7.2f} {t_b:7.2f} {t_c:7.2f} {t_d:7.2f}")
            xt = self.sample_by_phase3(t_a, t_b, t_c, t_d, xt, mt)
            t_idx += 3
        # while
        if t_idx + 3 == last_idx:   # phases: [3, 3, 3, , , 3, 2, 1]
            t_a, t_b, t_c, t_d = t_arr[-4:]
            mt = self.find_mt(t_a)
            if r_idx == 0:
                logging.info(f"order3_rf()mt_{mt.index}: ts[{t_idx:2d}]:{t_a:7.2f} {t_b:7.2f} {t_c:7.2f}")
            xt = self.sample_by_phase2(t_a, t_b, t_c, xt, mt)
            mt = self.find_mt(t_c)
            if r_idx == 0:
                logging.info(f"order3_rf()mt_{mt.index}: ts[{t_idx+2:2d}]:{t_c:7.2f} {t_d:7.2f}")
            xt = self.sample_by_phase1(t_c, t_d, xt, mt)
        elif t_idx + 2 == last_idx:   # phases: [3, 3, 3, , , 3, 2]
            t_a, t_b, t_c = t_arr[-3:]
            mt = self.find_mt(t_a)
            if r_idx == 0:
                logging.info(f"order3_rf()mt_{mt.index}: ts[{t_idx:2d}]:{t_a:7.2f} {t_b:7.2f} {t_c:7.2f}")
            xt = self.sample_by_phase2(t_a, t_b, t_c, xt, mt)
        elif t_idx + 1 == last_idx:   # phases: [3, 3, 3, , ,3, 1]
            t_a, t_b = t_arr[-2:]
            mt = self.find_mt(t_a)
            if r_idx == 0:
                logging.info(f"order3_rf()mt_{mt.index}: ts[{t_idx:2d}]:{t_a:7.2f} {t_b:7.2f}")
            xt = self.sample_by_phase1(t_a, t_b, xt, mt)
        else:
            raise ValueError(f"Unexpected: t_idx:{t_idx}, last_idx:{last_idx}")
        return xt

    def sample_by_phase1(self, ts_a, ts_b, xt, mt):
        grad = mt(xt, ts_a)  # gradient
        delta = (ts_a - ts_b) / (mt.ts_high - mt.ts_low)
        delta = (torch.ones(1, device=self.device) * delta).view(-1, 1, 1, 1)
        xt_next = xt - grad * delta
        return xt_next

    def sample_by_phase2(self, ts_a, ts_b, ts_c, xt, mt):
        grad_a = mt(xt, ts_a)       # gradient at ts_a
        delta = (ts_a - ts_b) / (mt.ts_high - mt.ts_low)
        delta = (torch.ones(1, device=self.device) * delta).view(-1, 1, 1, 1)
        xt_b = xt - grad_a * delta  # xt at ts_b
        grad_b = mt(xt_b, ts_b)     # gradient at ts_b
        delta = (ts_a - ts_c) / (mt.ts_high - mt.ts_low)
        delta = (torch.ones(1, device=self.device) * delta).view(-1, 1, 1, 1)
        xt_next = xt - grad_b * delta
        return xt_next

    def sample_by_phase3(self, ts_a, ts_b, ts_c, ts_d, xt, mt):
        # use some schema similar with Simpson's rule:
        # grad = (g(a) + 3*g(b) + 2*g(c)) / 6
        grad_a = mt(xt, ts_a)       # gradient at ts_a
        delta = (ts_a - ts_b) / (mt.ts_high - mt.ts_low)
        delta = (torch.ones(1, device=self.device) * delta).view(-1, 1, 1, 1)
        xt_b = xt - grad_a * delta  # xt at ts_b
        grad_b = mt(xt_b, ts_b)     # gradient at ts_b
        delta = (ts_b - ts_c) / (mt.ts_high - mt.ts_low)
        delta = (torch.ones(1, device=self.device) * delta).view(-1, 1, 1, 1)
        xt_c = xt_b - grad_b * delta
        grad_c = mt(xt_c, ts_c)
        grad = (grad_a + 3 * grad_b + 2 * grad_c) / 6
        delta = (ts_a - ts_d) / (mt.ts_high - mt.ts_low)
        delta = (torch.ones(1, device=self.device) * delta).view(-1, 1, 1, 1)
        xt_next = xt - grad * delta
        return xt_next

    def load_predefined_ts_file(self):
        f_path = self.predefined_ts_file
        if not os.path.exists(f_path):
            raise Exception(f"File not found: {f_path}")
        if not os.path.isfile(f_path):
            raise Exception(f"Not file: {f_path}")
        logging.info(f"load_predefined_ts_file(): {f_path}")
        with open(f_path, 'r') as f_ptr:
            lines = f_ptr.readlines()
        cnt_empty = 0
        cnt_comment = 0
        ts_arr = []  # alpha_bar array
        for line in lines:
            line = line.strip()
            if line == '':              # empty line
                cnt_empty += 1
                continue
            elif line.startswith('#'):  # line is like "# order     : 2"
                cnt_comment += 1
                continue
            arr = line.split(':')
            ts_arr.append(float(arr[0]))
        ab2s = lambda ff: ' '.join([f"{f:8.6f}" for f in ff])
        logging.info(f"  cnt_empty  : {cnt_empty}")
        logging.info(f"  cnt_comment: {cnt_comment}")
        logging.info(f"  cnt_valid  : {len(ts_arr)}")
        logging.info(f"  ab[:4]     : [{ab2s(ts_arr[:4])}]")
        logging.info(f"  ab[-4:]    : [{ab2s(ts_arr[-4:])}]")
        ts_arr.append(0)
        logging.info(f"  append 0 to ts_arr. New len: {len(ts_arr)}")
        return ts_arr

# class
