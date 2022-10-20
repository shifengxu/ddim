import os
import time
import logging

import numpy as np
import torch

import utils
from datasets import inverse_data_transform
from functions.ckpt_util import get_ckpt_path
from models.diffusion import Model, ModelStack
from runners.diffusion import Diffusion

import torchvision.utils as tvu

from utils import count_parameters


class DiffusionLatentSampling(Diffusion):
    def __init__(self, args, config, device=None):
        super().__init__(args, config, device)
        self.sample_count = 50000    # sample image count
        self.sample_img_init_id = 0  # sample image init ID. useful when generate image in parallel.

    def model_load_from_local(self, model):
        if self.args.sample_ckpt_path:
            ckpt_path = self.args.sample_ckpt_path
        elif getattr(self.config.sampling, "ckpt_id", None) is None:
            ckpt_path = os.path.join(self.args.log_path, "ckpt.pth")
        else:
            ckpt_path = os.path.join(self.args.log_path, f"ckpt_{self.config.sampling.ckpt_id}.pth")
        logging.info(f"load ckpt: {ckpt_path}")
        states = torch.load(ckpt_path, map_location=self.config.device)
        if isinstance(states, dict):
            model.load_state_dict(states['model'], strict=True)
        else:
            model.load_state_dict(states[0], strict=True)
        model = model.to(self.device)
        model = torch.nn.DataParallel(model, device_ids=self.args.gpu_ids)
        logging.info(f"model({type(model).__name__})")
        logging.info(f"  model.to({self.device})")
        logging.info(f"  torch.nn.DataParallel(model, device_ids={self.args.gpu_ids})")
        return model

    def model_stack_load_from_local(self, model: ModelStack):
        root_dir = self.args.sample_ckpt_dir
        if self.args.sample_stack_size == 10:
            ckpt_path_arr = [
                os.path.join(root_dir, "ckpt_000-100.pth"),
                os.path.join(root_dir, "ckpt_100-200.pth"),
                os.path.join(root_dir, "ckpt_200-300.pth"),
                os.path.join(root_dir, "ckpt_300-400.pth"),
                os.path.join(root_dir, "ckpt_400-500.pth"),
                os.path.join(root_dir, "ckpt_500-600.pth"),
                os.path.join(root_dir, "ckpt_600-700.pth"),
                os.path.join(root_dir, "ckpt_700-800.pth"),
                os.path.join(root_dir, "ckpt_800-900.pth"),
                os.path.join(root_dir, "ckpt_900-1000.pth"),
            ]
        elif self.args.sample_stack_size == 8:
            ckpt_path_arr = [
                os.path.join(root_dir, "ckpt_000-125.pth"),
                os.path.join(root_dir, "ckpt_125-250.pth"),
                os.path.join(root_dir, "ckpt_250-375.pth"),
                os.path.join(root_dir, "ckpt_375-500.pth"),
                os.path.join(root_dir, "ckpt_500-625.pth"),
                os.path.join(root_dir, "ckpt_625-750.pth"),
                os.path.join(root_dir, "ckpt_750-875.pth"),
                os.path.join(root_dir, "ckpt_875-1000.pth"),
            ]
        elif self.args.sample_stack_size == 4:
            ckpt_path_arr = [
                os.path.join(root_dir, "ckpt_000-250.pth"),
                os.path.join(root_dir, "ckpt_250-500.pth"),
                os.path.join(root_dir, "ckpt_500-750.pth"),
                os.path.join(root_dir, "ckpt_750-1000.pth"),
            ]
        else:  # other cases, need manual handling.
            ckpt_path_arr = [
                # f"{root_dir}/exp/model_S10E200/ckpt_000-100.pth",
                # f"{root_dir}/exp/model_S10E200/ckpt_100-200.pth",
                # f"{root_dir}/exp/model_S10E200/ckpt_200-300.pth",
                # f"{root_dir}/exp/model_S10E200/ckpt_300-400.pth",
                # f"{root_dir}/exp/model_S10E200/ckpt_400-500.pth",
                # f"{root_dir}/exp/model_S10E200/ckpt_500-600.pth",
                # f"{root_dir}/exp/model_S10E200/ckpt_600-700.pth",
                # f"{root_dir}/exp/model_S10E200/ckpt_700-800.pth",
                # f"{root_dir}/exp/model_S10E200/ckpt_800-900.pth",
                # f"{root_dir}/exp/model_S10E200/ckpt_900-1000.pth",
                f"{root_dir}/exp/model_stack_epoch500/ckpt_000-250.pth",
                f"{root_dir}/exp/model_stack_epoch500/ckpt_250-500.pth",
                f"{root_dir}/exp/model_stack_epoch500/ckpt_500-750.pth",
                f"{root_dir}/exp/model_stack_epoch500/ckpt_750-1000.pth",
            ]
        cnt, str_cnt = count_parameters(model, log_fn=None)
        logging.info(f"Loading ModelStack(stack_sz: {model.stack_sz})")
        logging.info(f"  config type  : {self.config.model.type}")
        logging.info(f"  param size   : {cnt} => {str_cnt}")
        logging.info(f"  ckpt_path_arr: {len(ckpt_path_arr)}")
        ms = model.model_stack
        for i, ckpt_path in enumerate(ckpt_path_arr):
            logging.info(f"  load ckpt {i: 2d} : {ckpt_path}")
            states = torch.load(ckpt_path, map_location=self.config.device)
            if isinstance(states, dict):
                # This is for backward-compatibility. As previously, the states is a list ([]).
                # And that was not convenient for adding or dropping items.
                # Therefore, we change to use dict.
                ms[i].load_state_dict(states['model'], strict=True)
            else:
                ms[i].load_state_dict(states[0], strict=True)
        # for
        model = model.to(self.device)
        model = torch.nn.DataParallel(model, device_ids=self.args.gpu_ids)
        logging.info(f"  model.to({self.device})")
        logging.info(f"  torch.nn.DataParallel(model, device_ids={self.args.gpu_ids})")
        return model

    def sample(self):
        if self.args.sample_stack_size > 1:
            model = ModelStack(self.config, self.args.sample_stack_size)
            model = self.model_stack_load_from_local(model)
        else:
            in_channels = self.args.model_in_channels
            out_channels = self.args.model_in_channels
            resolution = self.args.data_resolution
            model = Model(self.config, in_channels=in_channels, out_channels=out_channels, resolution=resolution)
            model = self.model_load_from_local(model)
        model.eval()

        config = self.config
        self.sample_count = self.args.sample_count
        self.sample_img_init_id = self.args.sample_img_init_id
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
                    in_channels or config.data.channels,
                    config.data.image_size,
                    config.data.image_size,
                    device=self.device,
                )
                self.sample_fid_vanilla(x_t, model, time_start, n_rounds, r_idx, b_sz)
                real_model = model.module if isinstance(model, torch.nn.DataParallel) else model
                if isinstance(real_model, ModelStack):
                    logging.info(f"ModelStack brick hit counter: {real_model.brick_hit_counter}")
            # for r_idx
        # with

    def sample_fid_vanilla(self, x_t, model, time_start, n_rounds, r_idx, b_sz):
        x = self.sample_image(x_t, model)
        img_cnt = len(x)
        elp, eta = utils.get_time_ttl_and_eta(time_start, r_idx+1, n_rounds)
        o_dir = self.args.sample_output_dir
        logging.info(f"save {img_cnt} latent to: {o_dir}. round:{r_idx}/{n_rounds}, elp:{elp}, eta:{eta}")
        img_last = None
        for i in range(img_cnt):
            img_id = r_idx * b_sz + i + self.sample_img_init_id
            img_cls = img_id // 1000
            img_dir = os.path.join(o_dir, f"{img_cls*1000:05d}")
            utils.make_dirs_if_need(img_dir)
            img_path = os.path.join(img_dir, f"{img_id:05d}")
            np.save(img_path, x[i].cpu().numpy())
            if i == img_cnt - 1: img_last = img_path
        logging.info(f"latent generated. last: {img_last}. "
                     f"init:{self.sample_img_init_id}; cnt:{self.sample_count}")

    def sample_image(self, x, model, last=True):
        if self.args.sample_type == "generalized":
            if self.args.skip_type == "uniform":
                skip = self.num_timesteps // self.args.timesteps
                seq = range(0, self.num_timesteps, skip)
            elif self.args.skip_type == "quad":
                seq = (np.linspace(0, np.sqrt(self.num_timesteps * 0.8), self.args.timesteps) ** 2)
                seq = [int(s) for s in list(seq)]
            else:
                raise NotImplementedError
            from functions.denoising import generalized_steps

            xs = generalized_steps(x, seq, model, self.betas, eta=self.args.eta)
            x = xs
        elif self.args.sample_type == "ddpm_noisy":
            if self.args.skip_type == "uniform":
                skip = self.num_timesteps // self.args.timesteps
                seq = range(0, self.num_timesteps, skip)
            elif self.args.skip_type == "quad":
                seq = (
                    np.linspace(
                        0, np.sqrt(self.num_timesteps * 0.8), self.args.timesteps
                    )
                    ** 2
                )
                seq = [int(s) for s in list(seq)]
            else:
                raise NotImplementedError
            from functions.denoising import ddpm_steps

            x = ddpm_steps(x, seq, model, self.betas)
        else:
            raise NotImplementedError
        if last:
            x = x[0][-1]
        return x

# class
