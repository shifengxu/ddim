import os
import time
import logging

import torch

import utils
from datasets import inverse_data_transform
from models.diffusion import Model
from models.ema import EMAHelper
from runners.diffusion import Diffusion

import torchvision.utils as tvu


class DiffusionSamplingFast(Diffusion):
    def __init__(self, args, config, device=None):
        super().__init__(args, config, device, output_ab=False)
        self.sample_count = self.args.sample_count
        self.sample_img_init_id = self.args.sample_img_init_id
        ab_list = list(args.ab_list)
        ab_lis2 = [1.] + list(args.ab_list)[:-1]
        ab_list.reverse()
        ab_lis2.reverse()
        self.reversed_alpha_bar = torch.tensor(ab_list, device=self.device)
        self.reversed_alpha_bar2 = torch.tensor(ab_lis2, device=self.device)
        self.model_map = {}  # alpha_bar_str ==> model
        f2s = lambda arr: ' ' .join([f"{f:10.8f}" for f in arr])
        self.f2s = f2s
        logging.info(f"DiffusionSamplingFast()")
        logging.info(f"  args.sample_output_dir : {self.args.sample_output_dir}")
        logging.info(f"  args.sample_type       : {self.args.sample_type}")
        logging.info(f"  args.skip_type         : {self.args.skip_type}")
        logging.info(f"  args.timesteps         : {self.args.timesteps}")
        logging.info(f"  num_timesteps          : {self.num_timesteps}")
        logging.info(f"  sample_count           : {self.sample_count}")
        logging.info(f"  sample_img_init_id     : {self.sample_img_init_id}")
        logging.info(f"  reversed_alpha_bar     : {f2s(self.reversed_alpha_bar)}")
        logging.info(f"  reversed_alpha_bar2    : {f2s(self.reversed_alpha_bar2)}")
        self._batch_idx = 0

    def fetch_ckpt_path_arr(self):
        root_dir = self.args.sample_ckpt_dir
        ckpt_path_arr = []
        if not root_dir or not os.path.exists(root_dir):
            return ckpt_path_arr
        for fname in os.listdir(root_dir):
            if fname.endswith(".ckpt"):
                ckpt_path_arr.append(os.path.join(root_dir, fname))
        ckpt_path_arr.sort()
        return ckpt_path_arr

    def load_model_map(self, ckpt_path_arr):
        args, config = self.args, self.config
        in_ch = args.model_in_channels
        out_ch = args.model_in_channels
        res = args.data_resolution
        ts_type = 'continuous'

        for i, ckpt_path in enumerate(ckpt_path_arr):
            model = Model(config, in_channels=in_ch, out_channels=out_ch, resolution=res, ts_type=ts_type)
            states = torch.load(ckpt_path, map_location=self.device)
            ab_list = states['ab_list']
            logging.info(f"load ckpt {i: 2d} : {ckpt_path}. ab_list={self.f2s(ab_list)}")
            model.load_state_dict(states['model'])
            if args.ema_flag:
                ema_helper = EMAHelper()
                ema_helper.register(model)
                ema_helper.load_state_dict(states['ema_helper'])
                logging.info(f"  ema_helper.ema(model)")
                ema_helper.ema(model)

            model = model.to(self.device)
            if len(self.args.gpu_ids) > 0:
                model = torch.nn.DataParallel(model, device_ids=self.args.gpu_ids)
            for ab in ab_list:
                ab_str = f"{ab:10.8f}"
                if ab_str in self.model_map:
                    raise ValueError(f"Duplicate ab_str found: {ab_str}. The 2nd is in {ckpt_path}")
                self.model_map[ab_str] = model
            # for
        # for
        return self.model_map

    def sample(self):
        config = self.config
        ckpt_path_arr = self.fetch_ckpt_path_arr()
        self.load_model_map(ckpt_path_arr)
        b_sz = self.args.sample_batch_size or config.sampling.batch_size
        n_rounds = (self.sample_count - 1) // b_sz + 1  # get the ceiling
        logging.info(f"DiffusionSamplingFast::sample()")
        logging.info(f"  batch_size : {b_sz}")
        logging.info(f"  n_rounds   : {n_rounds}")
        img_dir = self.args.sample_output_dir
        if not os.path.exists(img_dir):
            logging.info(f"  os.makedirs({img_dir})")
            os.makedirs(img_dir)
        time_start = time.time()
        with torch.no_grad():
            for r_idx in range(n_rounds):
                self._batch_idx = r_idx
                n = b_sz if r_idx + 1 < n_rounds else self.sample_count - r_idx * b_sz
                logging.info(f"round: {r_idx}/{n_rounds}. to generate: {n}")
                cd = config.data
                x_t = torch.randn(n, cd.channels, cd.image_size, cd.image_size, device=self.device)
                x = self.fast_steps(x_t, n)
                self.save_image(x, config, time_start, n_rounds, r_idx, b_sz)
            # for r_idx
        # with

    def save_image(self, x, config, time_start, n_rounds, r_idx, b_sz):
        x = inverse_data_transform(config, x)
        img_cnt = len(x)
        elp, eta = utils.get_time_ttl_and_eta(time_start, r_idx+1, n_rounds)
        img_dir = self.args.sample_output_dir
        logging.info(f"save {img_cnt} images to: {img_dir}. round:{r_idx}/{n_rounds}, elp:{elp}, eta:{eta}")
        img_last = None
        for i in range(img_cnt):
            img_id = r_idx * b_sz + i + self.sample_img_init_id
            img_path = os.path.join(img_dir, f"{img_id:05d}.png")
            tvu.save_image(x[i], img_path)
            if i == img_cnt - 1: img_last = f"{img_id:05d}.png"
        logging.info(f"  image generated: {img_last}")

    def fast_steps(self, x_T, b_sz):
        xt = x_T
        # idx_arr = [0.99900, 0.64916, 0.37434, 0.17454, 0.04975]
        # idx_arr = torch.tensor(idx_arr, device=self.device)
        for at, aq in zip(self.reversed_alpha_bar, self.reversed_alpha_bar2):
            if self._batch_idx == 0:
                logging.info(f"fast_steps(): at={at:10.8f}, aq={aq:10.8f}")
            at4d = at.view(-1, 1, 1, 1) # alpha_bar_t
            aq4d = aq.view(-1, 1, 1, 1) # alpha_bar_{t-1}
            mt4d = (at/aq).view(-1, 1, 1, 1)
            at1d = at.expand(b_sz)
            ab_str = f"{at:10.8f}"
            model = self.model_map.get(ab_str, None)
            if model is None: raise ValueError(f"Not found model for alpha_bar: {ab_str}")
            et = model(xt, at1d)        # epsilon_t
            xt_next = (xt - (1 - at4d).sqrt() * et) / mt4d.sqrt() + (1 - aq4d).sqrt() * et
            xt = xt_next
        # for
        return xt

# class
