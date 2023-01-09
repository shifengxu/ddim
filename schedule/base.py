import math
import os

import numpy as np
import torch
from torch import nn, Tensor

import utils


class ScheduleBase:
    @staticmethod
    def load_floats(f_path, log_fn=utils.log_info):
        if not os.path.exists(f_path):
            raise Exception(f"File not found: {f_path}")
        if not os.path.isfile(f_path):
            raise Exception(f"Not file: {f_path}")
        log_fn(f"load_floats() from file: {f_path}")
        with open(f_path, 'r') as f:
            lines = f.readlines()
        cnt_empty = 0
        cnt_comment = 0
        f_arr = []
        for line in lines:
            line = line.strip()
            if line == '':
                cnt_empty += 1
                continue
            if line.startswith('#'):
                cnt_comment += 1
                continue
            flt = float(line)
            f_arr.append(flt)
        log_fn(f"  cnt_empty  : {cnt_empty}")
        log_fn(f"  cnt_comment: {cnt_comment}")
        log_fn(f"  cnt_valid  : {len(f_arr)}")
        weights = torch.tensor(f_arr, dtype=torch.float64)
        log_fn(f"  weights first 5: {weights[:5].numpy()}")
        log_fn(f"  weights last 5 : {weights[-5:].numpy()}")
        return weights

    @staticmethod
    def accumulate_variance(alpha: Tensor, aacum: Tensor, weight_arr: Tensor):
        """
        accumulate variance from x_1000 to x_1.
        """
        numerator = ((1-aacum).sqrt() - (alpha-aacum).sqrt())**2
        numerator *= weight_arr
        sub_var = numerator / aacum
        final_var = torch.sum(sub_var)
        return final_var

    @staticmethod
    def get_schedule_from_file(f_path, log_fn=utils.log_info):
        if not os.path.exists(f_path):
            raise Exception(f"File not found: {f_path}")
        if not os.path.isfile(f_path):
            raise Exception(f"Not file: {f_path}")
        log_fn(f"Read file: {f_path}")
        with open(f_path, 'r') as f:
            lines = f.readlines()
        cnt_empty = 0
        cnt_comment = 0
        s_type = ''
        f_arr = []
        for line in lines:
            line = line.strip()
            if line == '':
                cnt_empty += 1
                continue
            if line.startswith('#'):
                cnt_comment += 1
                continue
            if line.startswith('type:'):
                s_type = line.split(':')[1]
                continue
            flt = float(line)
            f_arr.append(flt)
        log_fn(f"  cnt_valid  : {len(f_arr)}")
        log_fn(f"  cnt_empty  : {cnt_empty}")
        log_fn(f"  cnt_comment: {cnt_comment}")
        log_fn(f"  s_type     : {s_type}")
        if s_type == 'alpha':
            alphas = torch.tensor(f_arr).float()
            betas = 1.0 - alphas
            alphas_cumprod = torch.cumprod(alphas, dim=0)
        else:
            raise Exception(f"Unsupported s_type: {s_type} from file {f_path}")
        return betas, alphas, alphas_cumprod

    @staticmethod
    def _get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_timesteps):
        def sigmoid(x):
            return 1 / (np.exp(-x) + 1)

        if beta_schedule == "quad":
            betas = (
                    np.linspace(
                        beta_start ** 0.5,
                        beta_end ** 0.5,
                        num_timesteps,
                        dtype=np.float64,
                    )
                    ** 2
            )
        elif beta_schedule == "linear":
            betas = np.linspace(
                beta_start, beta_end, num_timesteps, dtype=np.float64
            )
        elif beta_schedule == "const":
            betas = beta_end * np.ones(num_timesteps, dtype=np.float64)
        elif beta_schedule == "jsd":  # 1/T, 1/(T-1), 1/(T-2), ..., 1
            betas = 1.0 / np.linspace(
                num_timesteps, 1, num_timesteps, dtype=np.float64
            )
        elif beta_schedule == "sigmoid":
            betas = np.linspace(-6, 6, num_timesteps)
            betas = sigmoid(betas) * (beta_end - beta_start) + beta_start
        else:
            raise NotImplementedError(beta_schedule)
        assert betas.shape == (num_timesteps,)
        return betas

    @staticmethod
    def get_alpha_cumprod(beta_schedule, ts_cnt=1000):
        if beta_schedule == "cosine":
            alphas_cumprod = [] # alpha cumulate array
            for i in range(ts_cnt):
                t = i / ts_cnt
                ac = math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2
                alphas_cumprod.append(ac)
            return torch.Tensor(alphas_cumprod).float()
        elif beta_schedule.startswith('cos:'):
            expo_str = beta_schedule.split(':')[1]  # "cos:2.2"
            expo = float(expo_str)
            alphas_cumprod = []  # alpha cumulate array
            for i in range(ts_cnt):
                t = i / ts_cnt
                ac = math.cos((t + 0.008) / 1.008 * math.pi / 2) ** expo
                alphas_cumprod.append(ac)
            return torch.Tensor(alphas_cumprod).float()
        elif beta_schedule.startswith("noise_rt_expo:"):
            # noise is: 1 - alpha_accumulated
            expo_str = beta_schedule.split(':')[1]  # "noise_rt_expo:2.2"
            expo = float(expo_str)
            n_low, n_high = 0.008, 0.999 # old value
            # n_low, n_high = 0.001, 0.9999  # if "noise_rt_expo:1", got FID 27.792929 on CIFAR-10
            sq_root = np.linspace(n_low, n_high, ts_cnt, dtype=np.float64)
            sq_root = torch.from_numpy(sq_root).float()
            if expo != 1.0:
                sq_root = torch.pow(sq_root, expo)
            sq = torch.mul(sq_root, sq_root)
            return 1 - sq
        elif beta_schedule.startswith('aacum_rt_expo:'):
            expo_str = beta_schedule.split(':')[1]  # "aacum_rt_expo:2.2"
            expo = float(expo_str)
            n_high, n_low = 0.9999, 0.0008 # old value
            # n_high, n_low = 0.9999, 0.001
            # given: 0.9999, 0.001
            #   if "aacum_rt_expo:1",   got FID 22.608681 on CIFAR-10
            #   if "aacum_rt_expo:1.5", got FID 49.226592 on CIFAR-10
            sq_root = np.linspace(n_high, n_low, ts_cnt, dtype=np.float64)
            sq_root = torch.from_numpy(sq_root).float()
            if expo != 1.0:
                sq_root = torch.pow(sq_root, expo)
            return torch.mul(sq_root, sq_root)
        elif beta_schedule.startswith('file:'):
            f_path = beta_schedule.split(':')[1]
            betas, alphas, alphas_cumprod = ScheduleBase.get_schedule_from_file(f_path)
        else:
            betas = ScheduleBase._get_beta_schedule(
                beta_schedule=beta_schedule,
                beta_start=0.0001,
                beta_end=0.02,
                num_timesteps=ts_cnt,
            )
            betas = torch.from_numpy(betas).float()
            alphas = 1.0 - betas
            alphas_cumprod = alphas.cumprod(dim=0)
        return alphas_cumprod


class Schedule1Model(nn.Module):
    def __init__(self, out_channels=1000):
        super().__init__()
        # self.linear1 = torch.nn.Linear(1,  50, dtype=torch.float64)
        # self.linear2 = torch.nn.Linear(50,  500, dtype=torch.float64)
        # self.linear3 = torch.nn.Linear(500,  out_channels, dtype=torch.float64)
        #
        # # the max threshold of the alpha-accumulated
        # self.linearMax = torch.nn.Sequential(
        #     torch.nn.Linear(1, 100, dtype=torch.float64),
        #     torch.nn.Linear(100, 1, dtype=torch.float64),
        # )

        # The two-level linear is better than pure nn.Parameter().
        # Pure nn.Parameter() means such:
        #   self.aa_max = torch.nn.Parameter(torch.ones((1,), dtype=torch.float64), requires_grad=True)
        self.linear1 = torch.nn.Linear(1000,  2000, dtype=torch.float64)
        self.linear2 = torch.nn.Linear(2000,  2000, dtype=torch.float64)
        self.linear3 = torch.nn.Linear(2000,  out_channels, dtype=torch.float64)

        # The two-level linear is better than pure nn.Parameters().
        # Pure nn.Parameter() means such:
        #   self.aa_max = torch.nn.Parameter(torch.ones((1,), dtype=torch.float64), requires_grad=True)
        self.linearMax = torch.nn.Sequential(
            torch.nn.Linear(1, 100, dtype=torch.float64),
            torch.nn.Linear(100, 1, dtype=torch.float64),
        )
        # the seed. we choose value 0.5. And it is better than value 1.0
        ones_k = torch.mul(torch.ones((1000,), dtype=torch.float64), 0.5)
        self.seed_k = torch.nn.Parameter(ones_k, requires_grad=False)
        ones_1 = torch.mul(torch.ones((1,), dtype=torch.float64), 0.5)
        self.seed_1 = torch.nn.Parameter(ones_1, requires_grad=False)

    def gradient_clip(self):
        if self.linear1.weight.grad is not None:
            self.linear1.weight.grad = torch.tanh(self.linear1.weight.grad)
            self.linear1.weight.grad = torch.tanh(self.linear1.weight.grad)
        if self.linear2.weight.grad is not None:
            self.linear2.weight.grad = torch.tanh(self.linear2.weight.grad)
            self.linear2.weight.grad = torch.tanh(self.linear2.weight.grad)
        if self.linear3.weight.grad is not None:
            self.linear3.weight.grad = torch.tanh(self.linear3.weight.grad)
            self.linear3.weight.grad = torch.tanh(self.linear3.weight.grad)

    def forward(self, simple_mode=False):
        # output1 = self.linear1(input_seed)
        # # output1 = output1 * torch.sigmoid(output1)
        # output2 = self.linear2(output1)
        # # output2 = output2 * torch.sigmoid(output2)
        # output3 = self.linear3(output2)
        # output = torch.softmax(output3, dim=0)
        #
        # aa_max = self.linearMax(input_seed)
        # aa_max = torch.sigmoid(aa_max)

        output = self.linear1(self.seed_k)
        output = self.linear2(output)
        output = self.linear3(output)
        output = torch.softmax(output, dim=0)
        aa_max = self.linearMax(self.seed_1)
        aa_max = torch.sigmoid(aa_max)
        if simple_mode:
            return output, aa_max

        aacum = torch.cumsum(output, dim=0)
        aacum = torch.flip(aacum, dims=(0,))
        aa_max = (0.9 + 0.0999999999*aa_max)  # make sure aa_max is in (0.9, 1)
        aacum = aacum * aa_max
        aa_prev = torch.cat([torch.ones(1).to(aa_max.device), aacum[:-1]], dim=0)
        alpha = torch.div(aacum, aa_prev)

        # make sure alpha > aacum.
        # Or else, the 2nd epoch will have output: tensor([nan, nan, ,,,])
        alpha[0] += 1e-12

        coefficient = ((1-aacum).sqrt() - (alpha-aacum).sqrt()) / alpha.sqrt()
        return alpha, aacum, coefficient
