import logging
import math

import numpy as np
import torch


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
        logging.info(f"Diffusion() ========================")
        logging.info(f"  device        : {self.device}")
        logging.info(f"  model_var_type: {self.model_var_type}")

        self.beta_schedule = args.beta_schedule or config.diffusion.beta_schedule
        self.beta_cos_expo = args.beta_cos_expo
        self.alphas, self.alphas_cumprod, self.betas = self.get_alphas_and_betas(config)
        self.num_timesteps = self.betas.shape[0]
        ts_range = args.ts_range
        if len(ts_range) == 0:
            ts_range = [0, self.num_timesteps]
        self.ts_low = ts_range[0]   # timestep low bound, inclusive
        self.ts_high = ts_range[1]  # timestep high bound, exclusive
        self._ts_log_flag = False   # only internal flag
        logging.info(f"  ts_low        : {self.ts_low}")
        logging.info(f"  ts_high       : {self.ts_high}")
        logging.info(f"  num_timesteps : {self.num_timesteps}")
        logging.info(f"  beta_schedule : {self.beta_schedule}")
        logging.info(f"  beta_cos_expo : {self.beta_cos_expo}")
        self.output_alphas_and_betas()

        if self.model_var_type == "fixedlarge":
            self.logvar = self.betas.log()
            # torch.cat(
            # [posterior_variance[1:2], betas[1:]], dim=0).log()
        elif self.model_var_type == "fixedsmall":
            ac = self.alphas_cumprod
            alphas_cumprod_prev = torch.cat([torch.ones(1).to(device), ac[:-1]], dim=0)
            posterior_variance = self.betas * (1.0 - alphas_cumprod_prev) / (1.0 - ac)
            self.logvar = posterior_variance.clamp(min=1e-20).log()

    def get_alphas_and_betas(self, config):
        device = self.device
        ts_cnt = config.diffusion.num_diffusion_timesteps
        if self.beta_schedule == "cosine":
            # cosine scheduler is from the following paper:
            # ICML. 2021. Alex Nichol. Improved Denoising Diffusion Probabilistic Models
            # In this option, it composes alphas_cumprod firstly, then alphas and betas.
            alphas_cumprod = [] # alpha cumulate array
            for i in range(ts_cnt):
                t = i / ts_cnt
                ac = math.cos((t + 0.008) / 1.008 * math.pi / 2) ** self.beta_cos_expo
                alphas_cumprod.append(ac)
            alphas_cumprod = torch.Tensor(alphas_cumprod).float().to(device)
            divisor = torch.cat([torch.ones(1).to(device), alphas_cumprod[:-1]], dim=0)
            alphas = torch.div(alphas_cumprod, divisor)
            betas = 1 - alphas
        else:
            betas = get_beta_schedule(
                beta_schedule=self.beta_schedule,
                beta_start=config.diffusion.beta_start,
                beta_end=config.diffusion.beta_end,
                num_diffusion_timesteps=ts_cnt,
            )
            betas = torch.from_numpy(betas).float().to(device)
            alphas = 1.0 - betas
            alphas_cumprod = alphas.cumprod(dim=0)
        return alphas, alphas_cumprod, betas

    def output_alphas_and_betas(self):
        def num2str(num_arr):
            flt_arr = [float(n) for n in num_arr]       # tensor to float
            str_arr = [f"{f:.6f}"[1:] for f in flt_arr] # float to string. 0.0001 => ".000100"
            return " ".join(str_arr)

        cnt = len(self.betas)
        itv = 10    # interval
        i = 0
        while i < cnt:
            r = min(i+itv, cnt)  # right bound
            logging.info(f"betas[{i:03d}~{r:03d}]:\t{num2str(self.betas[i:r])}")
            i += itv
        i = 0
        while i < cnt:
            r = min(i+itv, cnt)  # right bound
            logging.info(f"alphas[{i:03d}~{r:03d}]:\t{num2str(self.alphas[i:r])}")
            i += itv
        i = 0
        while i < cnt:
            r = min(i+itv, cnt)  # right bound
            logging.info(f"alphas_cumprod[{i:03d}~{r:03d}]:\t{num2str(self.alphas_cumprod[i:r])}")
            i += itv
