import os
import torch
from torch import Tensor

from schedule.base import ScheduleBase
from utils import log_info


class VarSimulator3:
    """
    Variance simulator3. Combine discretization error and prediction error.
    The idea is:
    a, given x_t and t, find epsilon_{theta}^(t)
    b, find "real" x_{t-1}, and calculate real epsilon^*
    c, compare epsilon_{theta}^(t) and epsilon^*
    Please note, "real" x_{t-1} means it is obtained without discretization error or prediction error.
    """
    def __init__(self, model, data_loader, data_limit=10000, steps=10,
                 beta_schedule='linear', ts_cnt=1000, log_fn=log_info,
                 cache_dir='.', device='cuda:0'):
        self.ts_cnt = ts_cnt
        self.beta_schedule = beta_schedule
        self.x_arr = ScheduleBase.get_alpha_cumprod(beta_schedule, self.ts_cnt).to(device)
        self.model = model
        self.model.eval()
        self.data_loader = data_loader
        self.data_limit = data_limit
        self.steps = steps
        self.log_fn = log_fn
        self.cache_dir = cache_dir
        self.one_step_cache_file = os.path.join(cache_dir, 'vs3_one_step_cache.txt')
        self.device = device
        log_fn(f"VarSimulator3()")
        log_fn(f"  model        : {type(model).__name__}")
        if isinstance(model, torch.nn.DataParallel):
            log_fn(f"  model        : {type(model.module).__name__}")
        log_fn(f"  vs_data_limit: {data_limit}")
        log_fn(f"  vs_steps     : {steps}")
        log_fn(f"  vs_beta_sched: {beta_schedule}")
        log_fn(f"  ts_cnt       : {ts_cnt}")
        log_fn(f"  device       : {device}")
        log_fn(f"  cache_file   : {self.one_step_cache_file}")
        self.one_step_cache_map = self.one_step_cache_map_load()

    def one_step_cache_map_load(self):
        if not os.path.exists(self.one_step_cache_file):
            return {}
        with open(self.one_step_cache_file, 'r') as fptr:
            lines = fptr.readlines()
        c_map = {}
        log_info(f"VarSimulator3::one_step_cache_map_load()...")
        for line in lines:  # line is like: 0.9999 -> 0.8888 : 33.33
            line = line.strip()
            if line == '' or line.startswith('#'): continue
            key, val = line.split(':')
            key, val = key.strip(), val.strip()
            c_map[key] = torch.tensor(float(val))
            log_info(f"  {key} : {val}")
        # for
        return c_map

    def one_step_cache_map_set(self, key, val):
        self.one_step_cache_map[key] = val
        with open(self.one_step_cache_file, 'w') as fptr:
            for k, v in self.one_step_cache_map.items():
                fptr.write(f"{k} : {v:13.8f}\n")
            # for
        # with

    def __call__(self, ab_arr: Tensor, include_index=False):
        ab_arr = ab_arr.float()
        ab_cnt = len(ab_arr)
        ab_0 = torch.tensor([0.99989998], dtype=torch.float, device=ab_arr.device)
        ab_seq = torch.cat([ab_0, ab_arr], dim=0)  # full alpha_bar sequence, include timestep 0
        log_info(f"VarSimulator3::__call__()")
        str_arr = [f"{ab:.8f}" for ab in ab_seq]
        log_info(f"  ab_seq: {' '.join(str_arr)}")
        mse_arr = []
        for i in range(1, len(ab_seq)):
            ab_t = ab_seq[i]
            ab_s = ab_seq[i-1]
            key = f"{ab_s:.6f} <- {ab_t:.6f}"
            mse = self.one_step_cache_map.get(key, None)
            if mse is None:
                log_info(f"VarSimulator3::calc_mse_one_step(){i:2d}/{ab_cnt}")
                mse = self.calc_mse_one_step(ab_s, ab_t)
                self.one_step_cache_map_set(key, mse)
            else:
                log_info(f"VarSimulator3::calc_mse_one_step(){i:2d}/{ab_cnt}: cached: {mse:13.8f}")
            mse_arr.append(mse)

        mse_arr = torch.tensor(mse_arr, device=self.device)
        if include_index:
            ts_arr = [self._binary_search(ab) for ab in ab_arr]
            return mse_arr, ts_arr
        return mse_arr

    def calc_mse_one_step(self, ab_s: Tensor, ab_t: Tensor):
        """
        :param ab_t: is alpha_bar_{t}
        :param ab_s: is alpha_bar_{t-1}
        :return:
        """
        steps = self.steps
        # the trajectory schedule for now is: evenly divide alpha_bar
        ab_arr = torch.linspace(ab_s.item(), ab_t.item(), steps=steps+1, device=ab_t.device)
        ts_arr = [self._binary_search(ab) for ab in ab_arr]
        ts_arr = torch.tensor(ts_arr, dtype=torch.float, device=ab_t.device)
        arr2str = lambda arr: ' '.join([f"{a:.8f}" for a in arr])
        log_info(f"  ab_arr: {arr2str(ab_arr)}")
        arr2str = lambda arr: ' '.join([f"{a:10.2f}" for a in arr])
        log_info(f"  ts_arr: {arr2str(ts_arr)}")
        ab_t = ab_t.view(-1, 1, 1, 1)   # alpha_bar_t
        ab_s = ab_s.view(-1, 1, 1, 1)   # alpha_bar_{t-1}
        al_t = ab_t / ab_s              # alpha
        coef = (1-ab_t).sqrt() / al_t.sqrt() - (1-ab_s).sqrt()
        ts = ts_arr[0]
        mse_sum = 0.0
        mse_cnt = 0
        with torch.no_grad():
            for i, (x0, y) in enumerate(self.data_loader):
                b_sz = x0.shape[0]
                x0 = x0.to(self.device)
                x0 = 2.0 * x0 - 1.0
                eps_gt = torch.randn_like(x0, dtype=torch.float)  # epsilon_ground_truth
                xt = x0 * ab_t.sqrt() + eps_gt * (1.0 - ab_t).sqrt()
                eps_pd = self.model(xt, ts.expand(b_sz))  # epsilon_predict
                xs_real = self.calc_xs(xt, ab_arr, ts_arr, eps_gt)  # xs is x_{t-1}
                eps_real = (xt / al_t.sqrt() - xs_real) / coef
                mse = (eps_pd - eps_real).square().sum(dim=(1, 2, 3)).mean(dim=0)
                msePdGt = (eps_pd - eps_gt).square().sum(dim=(1, 2, 3)).mean(dim=0)
                mseRlGt = (eps_real - eps_gt).square().sum(dim=(1, 2, 3)).mean(dim=0)
                mse_sum += mse * b_sz
                mse_cnt += b_sz
                mse = mse_sum / mse_cnt if mse_cnt > 0 else 0.0
                log_info(f"  vs3:{mse_cnt: 5d}: mse={mse:.4f}; msePdGt:{msePdGt:.4f}; mseRlGt:{mseRlGt:.4f}")
                if mse_cnt >= self.data_limit:
                    break
            # for
        # with
        mse = mse_sum / mse_cnt
        return mse

    def calc_xs(self, xt, ab_arr, ts_arr, eps_gt):
        """"""
        ab_t, ab_n = ab_arr[-1], ab_arr[-2]    # alpha_bar_t, alpha_bar_next
        al_t = ab_t / ab_n  # alpha
        # log_info(f"    ab_t: {ab_t:.8f}; ab_n: {ab_n:.8f}; al_t: {al_t:.8f}")
        # for the first step, we use ground-truth epsilon
        x_n = xt / al_t.sqrt() - ((1-ab_t).sqrt()/al_t.sqrt() - (1-ab_n).sqrt()) * eps_gt
        for i in reversed(range(0, len(ab_arr) - 2)):
            ab_t, ab_n = ab_arr[i+1], ab_arr[i]
            al_t = ab_t / ab_n
            # log_info(f"        ab_t: {ab_t:.8f}; ab_n: {ab_n:.8f}; al_t: {al_t:.8f}")
            ts = ts_arr[i+1]
            eps_pd = self.model(x_n, ts.expand(x_n.shape[0]))
            x_n = x_n / al_t.sqrt() - ((1-ab_t).sqrt()/al_t.sqrt() - (1-ab_n).sqrt()) * eps_pd
        return x_n

    def _binary_search(self, aacum: Tensor):
        """Use binary search to find timestep"""
        # define left bound index and right bound index
        # x_arr is from big to small
        if aacum >= self.x_arr[0]:
            return 0
        elif aacum <= self.x_arr[-1]:
            return self.ts_cnt - 1
        lbi = 0             # left bound index
        rbi = self.ts_cnt   # right bound index
        while lbi < rbi:
            mi = int(((lbi + rbi) / 2))
            if aacum > self.x_arr[mi]:
                rbi = mi
            else:
                lbi = mi
            if lbi + 1 == rbi:
                break
        # while
        # after iteration, lbi will be the target index

        # But the input aacum value may have difference with x_arr. So here
        # we handle the difference and make the result smooth
        # Firstly, find the right-hand index: re-use variable "rbi"
        lb = self.x_arr[lbi]
        rb = self.x_arr[rbi]
        portion = (lb - aacum) / (lb - rb)
        res = lbi + portion
        return res

# class
