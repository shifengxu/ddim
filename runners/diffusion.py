import logging
import math
import os

import numpy as np
import torch

from models.diffusion import ModelStack
from utils import count_parameters, extract_ts_range


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


def num2str(num_arr):
    flt_arr = [float(n) for n in num_arr]  # tensor to float
    str_arr = [f"{f:.6f}" for f in flt_arr]  # float to string. 0.0001 => ".000100"
    return " ".join(str_arr)


def output_arr(name, arr):
    cnt = len(arr)
    itv = 10    # interval
    i = 0
    while i < cnt:
        r = min(i+itv, cnt)  # right bound
        logging.info(f"{name}[{i:03d}~]: {num2str(arr[i:r])}")
        i += itv


class Diffusion(object):
    def __init__(self, args, config, device=None):
        self.args = args
        self.config = config
        if device is None:
            device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.device = device
        self.model_var_type = config.model.var_type
        logging.info(f"Diffusion() ========================")

        self.beta_schedule = args.beta_schedule or config.diffusion.beta_schedule
        self.alphas, self.alphas_cumprod, self.betas = self.get_alphas_and_betas(config)
        output_arr('betas', self.betas)
        output_arr('alphas', self.alphas)
        output_arr('alphas_cumprod', self.alphas_cumprod)
        # define alpha_cumprod[t-1]. this will be used when sampling images.
        # denote the above variable: alphas_cumproq
        arr = [torch.ones(1).to(self.device), self.alphas_cumprod[:-1]]
        self.alphas_cumproq = torch.cat(arr, dim=0)
        self.num_timesteps = self.betas.shape[0]
        ts_range = args.ts_range
        if len(ts_range) == 0:
            ts_range = [0, self.num_timesteps]
        self.ts_low = ts_range[0]   # timestep low bound, inclusive
        self.ts_high = ts_range[1]  # timestep high bound, exclusive
        self._ts_log_flag = False   # only internal flag
        logging.info(f"  device        : {self.device}")
        logging.info(f"  model_var_type: {self.model_var_type}")
        logging.info(f"  ts_low        : {self.ts_low}")
        logging.info(f"  ts_high       : {self.ts_high}")
        logging.info(f"  num_timesteps : {self.num_timesteps}")
        logging.info(f"  beta_schedule : {self.beta_schedule}")

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
        """
        Explanation_1:
            beta:linear     idx: 0          idx: 999
            beta            0.0001          0.02
            alpha           0.9999          0.98
            aacum           0.9999          0.00004
            aacum_sqrt      0.999949999     0.006324555
            noise           0.0001          0.99996
            noise_sqrt      0.01            0.99998
        Notes:
            aacum: is alpha accumulation: a0*a1*a2...
            noise: is just: 1 - accum
        :param config:
        :return:
        """
        device = self.device
        ts_cnt = config.diffusion.num_diffusion_timesteps
        if self.beta_schedule == "cosine":
            # cosine scheduler is from the following paper:
            # ICML. 2021. Alex Nichol. Improved Denoising Diffusion Probabilistic Models
            # In this option, it composes alphas_cumprod firstly, then alphas and betas.
            alphas_cumprod = [] # alpha cumulate array
            for i in range(ts_cnt):
                t = i / ts_cnt
                ac = math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2
                alphas_cumprod.append(ac)
            alphas_cumprod = torch.Tensor(alphas_cumprod).float().to(device)
            divisor = torch.cat([torch.ones(1).to(device), alphas_cumprod[:-1]], dim=0)
            alphas = torch.div(alphas_cumprod, divisor)
            betas = 1 - alphas
        elif self.beta_schedule.startswith('cos:'):
            expo_str = self.beta_schedule.split(':')[1]  # "cos:2.2"
            expo = float(expo_str)
            alphas_cumprod = []  # alpha cumulate array
            for i in range(ts_cnt):
                t = i / ts_cnt
                ac = math.cos((t + 0.008) / 1.008 * math.pi / 2) ** expo
                alphas_cumprod.append(ac)
            alphas_cumprod = torch.Tensor(alphas_cumprod).float().to(device)
            divisor = torch.cat([torch.ones(1).to(device), alphas_cumprod[:-1]], dim=0)
            alphas = torch.div(alphas_cumprod, divisor)
            betas = 1 - alphas
        elif self.beta_schedule.startswith("noise_rt_expo:"):
            # noise is: 1 - alpha_accumulated
            expo_str = self.beta_schedule.split(':')[1]  # "noise_rt_expo:2.2"
            expo = float(expo_str)
            n_low, n_high = 0.008, 0.999 # old value
            # n_low, n_high = 0.001, 0.9999  # if "noise_rt_expo:1", got FID 27.792929 on CIFAR-10
            sq_root = np.linspace(n_low, n_high, ts_cnt, dtype=np.float64)
            sq_root = torch.from_numpy(sq_root).float().to(device)
            if expo != 1.0:
                sq_root = torch.pow(sq_root, expo)
            output_arr(self.beta_schedule, sq_root)
            sq = torch.mul(sq_root, sq_root)
            alphas_cumprod = 1 - sq
            divisor = torch.cat([torch.ones(1).to(device), alphas_cumprod[:-1]], dim=0)
            alphas = torch.div(alphas_cumprod, divisor)
            betas = 1 - alphas
        elif self.beta_schedule.startswith('aacum_rt_expo:'):
            expo_str = self.beta_schedule.split(':')[1]  # "aacum_rt_expo:2.2"
            expo = float(expo_str)
            n_high, n_low = 0.9999, 0.0008 # old value
            # n_high, n_low = 0.9999, 0.001
            # given: 0.9999, 0.001
            #   if "aacum_rt_expo:1",   got FID 22.608681 on CIFAR-10
            #   if "aacum_rt_expo:1.5", got FID 49.226592 on CIFAR-10
            sq_root = np.linspace(n_high, n_low, ts_cnt, dtype=np.float64)
            sq_root = torch.from_numpy(sq_root).float().to(device)
            if expo != 1.0:
                sq_root = torch.pow(sq_root, expo)
            output_arr(self.beta_schedule, sq_root)
            alphas_cumprod = torch.mul(sq_root, sq_root)
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

    def generalized_steps(self, x_T, seq, model, **kwargs):
        """
        Original paper: Denoising Diffusion Implicit Models. ICLR. 2021
        :param x_T: x_T in formula; it has initial Gaussian Noise
        :param seq:    timestep t sequence
        :param model:
        :param kwargs:
        :return:
        """
        eta = kwargs.get("eta", 0)
        msg = f"diffusion::seq=[{seq[-1]}~{seq[0]}], len={len(seq)}"
        with torch.no_grad():
            b_sz = x_T.size(0)
            xt = x_T
            for i in reversed(seq):
                t = (torch.ones(b_sz) * i).to(x_T.device)   # [999., 999.]
                at = self.alphas_cumprod.index_select(0, t.long()).view(-1, 1, 1, 1) # alpha_t
                aq = self.alphas_cumproq.index_select(0, t.long()).view(-1, 1, 1, 1) # alpha_{t-1}
                mt = self.alphas.index_select(0, t.long()).view(-1, 1, 1, 1)
                et = model(xt, t)               # epsilon_t
                if eta == 0:
                    if i % 50 == 0: logging.info(f"generalized_steps(eta==0): {msg}; i={i}")
                    # simplified version of the formula.
                    # when at is too small, divide by at may have issue.
                    xt_next = (xt - (1 - at).sqrt() * et) / mt.sqrt() + (1 - aq).sqrt() * et
                else:
                    if i % 50 == 0: logging.info(f"generalized_steps(): {msg}; i={i}")
                    x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()  # formula (9)
                    sigma_t = eta * ((1 - at / aq) * (1 - aq) / (1 - at)).sqrt()  # formula (16)
                    c2 = (1 - aq - sigma_t ** 2).sqrt()
                    xt_next = aq.sqrt() * x0_t + c2 * et + sigma_t * torch.randn_like(x_T)  # formula (12)
                xt = xt_next
            # for
        # with
        return xt

    def get_ckpt_path_arr(self):
        root_dir = self.args.sample_ckpt_dir
        ckpt_path_arr = []
        if not root_dir or not os.path.exists(root_dir):
            return ckpt_path_arr
        for fname in os.listdir(root_dir):
            if fname.startswith("ckpt") and fname.endswith(".pth"):
                ckpt_path_arr.append(os.path.join(root_dir, fname))
        ckpt_path_arr.sort()
        return ckpt_path_arr

    def model_stack_load_from_local(self, model: ModelStack, ckpt_path_arr):
        cnt, str_cnt = count_parameters(model, log_fn=None)
        logging.info(f"Loading ModelStack(stack_sz: {model.stack_sz})")
        logging.info(f"  config type  : {self.config.model.type}")
        logging.info(f"  param size   : {cnt} => {str_cnt}")
        logging.info(f"  ckpt_path_arr: {len(ckpt_path_arr)}")
        ms = model.model_stack
        for i, ckpt_path in enumerate(ckpt_path_arr):
            model.tsr_stack[i] = extract_ts_range(ckpt_path)
            logging.info(f"  load ckpt {i: 2d} : {ckpt_path}. ts: {model.tsr_stack[i]}")
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
        if len(self.args.gpu_ids) > 0:
            model = torch.nn.DataParallel(model, device_ids=self.args.gpu_ids)
        logging.info(f"  model.to({self.device})")
        logging.info(f"  torch.nn.DataParallel(model, device_ids={self.args.gpu_ids})")
        return model

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
        logging.info(f"model({type(model).__name__})")
        logging.info(f"  model.to({self.device})")
        model = model.to(self.device)
        if len(self.args.gpu_ids) > 1:
            logging.info(f"torch.nn.DataParallel(model, device_ids={self.args.gpu_ids})")
            model = torch.nn.DataParallel(model, device_ids=self.args.gpu_ids)
        return model
