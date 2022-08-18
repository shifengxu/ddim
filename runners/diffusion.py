import os
import logging
import time

import numpy as np
import torch
import torch.utils.data as data
import torch.backends.cudnn as cudnn

from models.diffusion import Model, ModelStack
from models.ema import EMAHelper
from functions import get_optimizer
from functions.losses import loss_registry
from datasets import get_dataset, data_transform


from utils import count_parameters


def torch2hwcuint8(x, clip=False):
    if clip:
        x = torch.clamp(x, -1, 1)
    x = (x + 1.0) / 2.0
    return x


def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
    def sigmoid(x):
        return 1 / (np.exp(-x) + 1)

    if beta_schedule == "quad":
        betas = (
            np.linspace(
                beta_start ** 0.5,
                beta_end ** 0.5,
                num_diffusion_timesteps,
                dtype=np.float64,
            )
            ** 2
        )
    elif beta_schedule == "linear":
        betas = np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "const":
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "jsd":  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1.0 / np.linspace(
            num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "sigmoid":
        betas = np.linspace(-6, 6, num_diffusion_timesteps)
        betas = sigmoid(betas) * (beta_end - beta_start) + beta_start
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas


class Diffusion(object):
    def __init__(self, args, config, device=None):
        self.args = args
        self.config = config
        if device is None:
            device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.device = device

        self.model_var_type = config.model.var_type
        betas = get_beta_schedule(
            beta_schedule=config.diffusion.beta_schedule,
            beta_start=config.diffusion.beta_start,
            beta_end=config.diffusion.beta_end,
            num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps,
        )
        betas = self.betas = torch.from_numpy(betas).float().to(self.device)
        self.num_timesteps = betas.shape[0]
        logging.info(f"Diffusion() ========================")
        logging.info(f"  device        : {self.device}")
        logging.info(f"  model_var_type: {self.model_var_type}")
        logging.info(f"  num_timesteps : {self.num_timesteps}")
        ts_range = args.ts_range
        if len(ts_range) == 0:
            ts_range = [0, self.num_timesteps]
        self.ts_low = ts_range[0]   # timestep low bound, inclusive
        self.ts_high = ts_range[1]  # timestep high bound, exclusive
        logging.info(f"  ts_low        : {self.ts_low}")
        logging.info(f"  ts_high       : {self.ts_high}")

        alphas = 1.0 - betas
        alphas_cumprod = alphas.cumprod(dim=0)
        alphas_cumprod_prev = torch.cat(
            [torch.ones(1).to(device), alphas_cumprod[:-1]], dim=0
        )
        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        if self.model_var_type == "fixedlarge":
            self.logvar = betas.log()
            # torch.cat(
            # [posterior_variance[1:2], betas[1:]], dim=0).log()
        elif self.model_var_type == "fixedsmall":
            self.logvar = posterior_variance.clamp(min=1e-20).log()

    @staticmethod
    def get_time_ttl_and_eta(time_start, elapsed_iter, total_iter):
        """
        Get estimated total time and ETA time.
        :param time_start:
        :param elapsed_iter:
        :param total_iter:
        :return: string of elapsed time, string of ETA
        """

        def sec_to_str(sec):
            val = int(sec)  # seconds in int type
            s = val % 60
            val = val // 60  # minutes
            m = val % 60
            val = val // 60  # hours
            h = val % 24
            d = val // 24  # days
            return f"{d}-{h:02d}:{m:02d}:{s:02d}"

        elapsed_time = time.time() - time_start  # seconds elapsed
        elp = sec_to_str(elapsed_time)
        if elapsed_iter == 0:
            eta = 'NA'
        else:
            # seconds
            eta = elapsed_time * (total_iter - elapsed_iter) / elapsed_iter
            eta = sec_to_str(eta)
        return elp, eta
