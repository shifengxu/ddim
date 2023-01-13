import os
import time
import numpy as np
import torch

from albar_model import AlbarModel
from datasets import inverse_data_transform
from runners.diffusion import Diffusion
from utils import log_info, get_time_ttl_and_eta
import torchvision.utils as tvu
log_fn = log_info

class AlbarSampling(Diffusion):
    def __init__(self, args, config, device=None):
        super().__init__(args, config, device)
        self.sample_count = 50000    # sample image count
        self.sample_img_init_id = 0  # sample image init ID. useful when generate image in parallel.

    def sample(self):
        config = self.config
        args = self.args
        model = AlbarModel(config, albar_range=args.albar_range, log_fn=log_fn)
        model = self.model_load_from_local(model)
        model.eval()
        self.sample_count = args.sample_count
        self.sample_img_init_id = args.sample_img_init_id
        log_fn(f"AlbarSampling::sample(self, {type(model).__name__})...")
        log_fn(f"  args.sample_output_dir : {args.sample_output_dir}")
        log_fn(f"  args.sample_type       : {args.sample_type}")
        log_fn(f"  args.skip_type         : {args.skip_type}")
        log_fn(f"  args.timesteps         : {args.timesteps}")
        log_fn(f"  num_timesteps          : {self.num_timesteps}")
        log_fn(f"  sample_count           : {self.sample_count}")
        log_fn(f"  sample_img_init_id     : {self.sample_img_init_id}")
        b_sz = args.sample_batch_size or config.sampling.batch_size
        n_rounds = (self.sample_count - 1) // b_sz + 1  # get the ceiling
        log_fn(f"  batch_size             : {b_sz}")
        log_fn(f"  n_rounds               : {n_rounds}")
        log_fn(f"  Generating image samples for FID evaluation")
        if not os.path.exists(args.sample_output_dir):
            log_fn(f"  os.makedirs({args.sample_output_dir})")
            os.makedirs(args.sample_output_dir)
        time_start = time.time()
        with torch.no_grad():
            for r_idx in range(n_rounds):
                n = b_sz if r_idx + 1 < n_rounds else self.sample_count - r_idx * b_sz
                log_fn(f"round: {r_idx}/{n_rounds}. to generate: {n}")
                x_t = torch.randn(  # normal distribution with mean 0 and variance 1
                    n,
                    config.data.channels,
                    config.data.image_size,
                    config.data.image_size,
                    device=self.device,
                )
                self.sample_fid_vanilla(x_t, model, config, time_start, n_rounds, r_idx, b_sz)
            # for r_idx
        # with

    def sample_fid_vanilla(self, x_t, model, config, time_start, n_rounds, r_idx, b_sz):
        x = self.sample_image(x_t, model)
        x = inverse_data_transform(config, x)
        img_cnt = len(x)
        elp, eta = get_time_ttl_and_eta(time_start, r_idx+1, n_rounds)
        img_dir = self.args.sample_output_dir
        log_fn(f"save {img_cnt} images to: {img_dir}. round:{r_idx}/{n_rounds}, elp:{elp}, eta:{eta}")
        img_first = img_last = None
        for i in range(img_cnt):
            img_id = r_idx * b_sz + i + self.sample_img_init_id
            img_path = os.path.join(img_dir, f"{img_id:05d}.png")
            tvu.save_image(x[i], img_path)
            if i == 0: img_first = f"{img_id:05d}.png"
            if i == img_cnt - 1: img_last = f"{img_id:05d}.png"
        log_fn(f"image generated: {img_first} ~ {img_last}. "
               f"init:{self.sample_img_init_id}; cnt:{self.sample_count}")

    def sample_image(self, x, model):
        if self.args.skip_type == "uniform":
            skip = self.num_timesteps // self.args.timesteps
            seq = range(0, self.num_timesteps, skip)
        elif self.args.skip_type == "quad":
            seq = (np.linspace(0, np.sqrt(self.num_timesteps * 0.8), self.args.timesteps) ** 2)
            seq = [int(s) for s in list(seq)]
        else:
            raise NotImplementedError
        xt = self.albar_steps(x, seq, model)
        return xt

    def albar_steps(self, x_T, seq, model):
        """
        Original paper: Denoising Diffusion Implicit Models. ICLR. 2021
        :param x_T: x_T in formula; it has initial Gaussian Noise
        :param seq:    timestep t sequence
        :param model:
        :return:
        """
        msg = f"diffusion::seq=[{seq[-1]}~{seq[0]}], len={len(seq)}"
        b_sz = len(x_T)
        xt = x_T
        for i in reversed(seq):
            t = (torch.ones(b_sz) * i).to(self.device)   # [999., 999.]
            ab_t = self.alphas_cumprod.index_select(0, t.long()) # alpha_bar_t
            ab_s = self.alphas_cumproq.index_select(0, t.long()) # alpha_bar_{t-1}
            e_t = model(xt, ab_t)  # epsilon_t
            if i % 50 == 0: log_fn(f"albar_steps(): {msg}; i={i}")
            # simplified version of the formula.
            ab_t4d = ab_t.view(-1, 1, 1, 1)
            ab_s4d = ab_s.view(-1, 1, 1, 1)
            a_t4d = ab_t4d / ab_s4d  # alpha_t
            xt_next = (xt - (1 - ab_t4d).sqrt() * e_t) / a_t4d.sqrt() + (1 - ab_s4d).sqrt() * e_t
            xt = xt_next
        # for
        return xt

# class
