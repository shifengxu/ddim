import os
import time
import logging

import numpy as np
import torch

import utils
from datasets import inverse_data_transform
from models.diffusion import Model
from runners.diffusion import Diffusion

import torchvision.utils as tvu


class DiffusionDpmSolver(Diffusion):
    def __init__(self, args, config, device=None):
        super().__init__(args, config, device)
        self.sample_count = 50000    # sample image count
        self.sample_img_init_id = 0  # sample image init ID. useful when generate image in parallel.

    def save_images(self, config, x, time_start, r_idx, n_rounds, b_sz):
        x = inverse_data_transform(config, x)
        img_cnt = len(x)
        elp, eta = utils.get_time_ttl_and_eta(time_start, r_idx+1, n_rounds)
        img_dir = self.args.sample_output_dir
        logging.info(f"save {img_cnt} images to: {img_dir}. round:{r_idx}/{n_rounds}, elp:{elp}, eta:{eta}")
        img_first = img_last = None
        for i in range(img_cnt):
            img_id = r_idx * b_sz + i + self.sample_img_init_id
            img_path = os.path.join(img_dir, f"{img_id:05d}.png")
            tvu.save_image(x[i], img_path)
            if i == 0: img_first = f"{img_id:05d}.png"
            if i == img_cnt - 1: img_last = f"{img_id:05d}.png"
        logging.info(f"images generated: {img_first} ~ {img_last}.")

    def sample(self):
        config, args = self.config, self.args
        model = Model(config)
        model = self.model_load_from_local(model)
        model.eval()

        self.sample_count = args.sample_count
        self.sample_img_init_id = args.sample_img_init_id
        logging.info(f"DiffusionDpmSolver::sample(self, {type(model).__name__})...")
        logging.info(f"  args.sample_output_dir : {args.sample_output_dir}")
        logging.info(f"  args.sample_type       : {args.sample_type}")
        logging.info(f"  args.skip_type         : {args.skip_type}")
        logging.info(f"  args.timesteps         : {args.timesteps}")
        logging.info(f"  num_timesteps          : {self.num_timesteps}")
        logging.info(f"  sample_count           : {self.sample_count}")
        logging.info(f"  sample_img_init_id     : {self.sample_img_init_id}")
        b_sz = args.sample_batch_size or config.sampling.batch_size
        n_rounds = (self.sample_count - 1) // b_sz + 1  # get the ceiling
        logging.info(f"  batch_size             : {b_sz}")
        logging.info(f"  n_rounds               : {n_rounds}")
        logging.info(f"  Generating image samples for FID evaluation")
        if not os.path.exists(args.sample_output_dir):
            logging.info(f"  os.makedirs({args.sample_output_dir})")
            os.makedirs(args.sample_output_dir)
        time_start = time.time()
        d = config.data
        with torch.no_grad():
            for r_idx in range(n_rounds):
                n = b_sz if r_idx + 1 < n_rounds else self.sample_count - r_idx * b_sz
                logging.info(f"DiffusionDpmSolver::round: {r_idx}/{n_rounds}. to generate: {n}")
                x_t = torch.randn(n, d.channels, d.image_size, d.image_size, device=self.device)
                x = self.sample_by_dpm_solver(x_t, model)
                self.save_images(config, x, time_start, r_idx, n_rounds, b_sz)
            # for r_idx
        # with

    def sample_by_dpm_solver(self, x_T, model):
        from models.dpm_solver_pytorch import NoiseScheduleVP, model_wrapper, DPM_Solver
        # You need to firstly define your model and the extra inputs of your model,
        # And initialize an `x_T` from the standard normal distribution.
        # `model` has the format: model(x_t, t_input, **model_kwargs).
        # If your model has no extra inputs, just let model_kwargs = {}.
        #
        # If you use discrete-time DPMs, you need to further define the
        # beta arrays for the noise schedule.
        #
        # model = ....
        # model_kwargs = {...}
        # x_T = ...
        # betas = ....

        # 1. Define the noise schedule.
        noise_schedule = NoiseScheduleVP(schedule='discrete', alphas_cumprod=self.alphas_cumprod)

        # 2. Convert your discrete-time `model` to the continuous-time
        # noise prediction model. Here is an example for a diffusion model
        # `model` with the noise prediction type ("noise") .
        model_fn = model_wrapper(
            model,
            noise_schedule,
            model_type="noise",  # or "x_start" or "v" or "score"
            model_kwargs={},
        )

        # 3. Define dpm-solver and sample by singlestep DPM-Solver.
        # (We recommend singlestep DPM-Solver for unconditional sampling)
        # You can adjust the `steps` to balance the computation
        # costs and the sample quality.
        dpm_solver = DPM_Solver(model_fn, noise_schedule, algorithm_type="dpmsolver")
        # Can also try
        # dpm_solver = DPM_Solver(model_fn, noise_schedule, algorithm_type="dpmsolver++")

        # You can use steps = 10, 12, 15, 20, 25, 50, 100.
        # Empirically, we find that steps in [10, 20] can generate quite good samples.
        # And steps = 20 can almost converge.
        x_sample = dpm_solver.sample(
            x_T,
            steps=20,
            order=3,
            skip_type="time_uniform",
            method="singlestep",
        )
        return x_sample

# class
