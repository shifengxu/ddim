import os
import time
import logging

import numpy as np
import torch

import utils
from datasets import inverse_data_transform
from models.diffusion import Model, ModelStack
from runners.diffusion import Diffusion

import torchvision.utils as tvu


class DiffusionSamplingContinuous(Diffusion):
    def __init__(self, args, config, device=None):
        super().__init__(args, config, device)
        self.sample_count = args.sample_count
        self.sample_img_init_id = args.sample_img_init_id

    def sample(self):
        config = self.config
        model = Model(config, ts_type='continuous')
        model = self.model_load_from_local(model)
        model.eval()
        logging.info(f"DiffusionSamplingContinuous::sample()...")
        logging.info(f"  args.sample_output_dir : {self.args.sample_output_dir}")
        logging.info(f"  args.sample_type       : {self.args.sample_type}")
        logging.info(f"  args.skip_type         : {self.args.skip_type}")
        logging.info(f"  args.timesteps         : {self.args.timesteps}")
        logging.info(f"  num_timesteps          : {self.num_timesteps}")
        logging.info(f"  sample_count           : {self.sample_count}")
        logging.info(f"  sample_img_init_id     : {self.sample_img_init_id}")
        b_sz = self.args.sample_batch_size or config.sampling.batch_size
        n_rounds = (self.sample_count - 1) // b_sz + 1  # get the ceiling
        logging.info(f"  batch_size             : {b_sz}")
        logging.info(f"  n_rounds               : {n_rounds}")
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
                x = self.sample_image(x_t, model)
                self.save_image(x, config, time_start, n_rounds, r_idx, b_sz)
            # for r_idx
        # with

    def save_image(self, x, config, time_start, n_rounds, r_idx, b_sz):
        x = inverse_data_transform(config, x)
        img_cnt = len(x)
        elp, eta = utils.get_time_ttl_and_eta(time_start, r_idx+1, n_rounds)
        img_dir = self.args.sample_output_dir
        logging.info(f"save {img_cnt} images to: {img_dir}. round:{r_idx}/{n_rounds}, elp:{elp}, eta:{eta}")
        img_path = None
        for i in range(img_cnt):
            img_id = r_idx * b_sz + i + self.sample_img_init_id
            img_path = os.path.join(img_dir, f"{img_id:05d}.png")
            tvu.save_image(x[i], img_path)
        logging.info(f"  image generated: {img_path}.")

    def sample_image(self, x, model):
        seq = range(0, self.num_timesteps, 1)
        seq = [float(i)/1000. for i in seq]
        if not hasattr(self, '_ts_type_collapse_flag'):
            # log it only once. Then it will show in logs and won't mess up the logs.
            logging.info(f"Collapse timestep seq with 1000 as ts_type is {self.args.ts_type}")
            setattr(self, '_ts_type_collapse_flag', True)
        # if
        xt = self.sample_image_conti_steps(x, seq, model)
        return xt

    def sample_image_conti_steps(self, x_T, ts_seq, model):
        msg = f"seq=[{ts_seq[-1]}~{ts_seq[0]}], len={len(ts_seq)}"
        b_sz = x_T.size(0)
        xt = x_T
        with torch.no_grad():
            for i, ts in enumerate(reversed(ts_seq)):
                t = (torch.ones(b_sz) * ts).to(x_T.device)   # [0.999., 0.999.]
                t_2d = t.reshape((-1, 1))
                at = utils.linear_interpolate(t_2d, self.t_array, self.ab_array)
                aq = utils.linear_interpolate(t_2d, self.t_array, self.aq_array)
                mt = utils.linear_interpolate(t_2d, self.t_array, self.a_array)
                at = at.view(-1, 1, 1, 1) # alpha_bar_t
                aq = aq.view(-1, 1, 1, 1) # alpha_bar_{t-1}
                mt = mt.view(-1, 1, 1, 1) # alpha_t.
                et = model(xt, t)         # epsilon_t
                if i % 50 == 0:
                    logging.info(f"sample_image_conti_steps(): {msg}; i={i}")
                xt_next = (xt - (1 - at).sqrt() * et) / mt.sqrt() + (1 - aq).sqrt() * et
                xt = xt_next
            # for
        # with
        return xt

# class
