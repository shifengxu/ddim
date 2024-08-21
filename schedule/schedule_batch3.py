import os
import time
import torch
from torch import optim
from torch import Tensor

import utils
from schedule.base import ScheduleBase, ScheduleAlphaModel
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
                 beta_schedule='linear', ts_cnt=1000, log_info=log_info,
                 cache_dir='.', device='cuda:0'):
        self.ts_cnt = ts_cnt
        self.beta_schedule = beta_schedule
        self.x_arr = ScheduleBase.get_alpha_cumprod(beta_schedule, self.ts_cnt).to(device)
        self.model = model
        self.model.eval()
        self.data_loader = data_loader
        self.data_limit = data_limit
        self.steps = steps
        self.log_info = log_info
        self.cache_dir = cache_dir
        self.one_step_cache_file = os.path.join(cache_dir, 'vs3_one_step_cache.txt')
        self.device = device
        log_info(f"VarSimulator3()")
        log_info(f"  model        : {type(model).__name__}")
        if isinstance(model, torch.nn.DataParallel):
            log_info(f"  model        : {type(model.module).__name__}")
        log_info(f"  vs_data_limit: {data_limit}")
        log_info(f"  vs_steps     : {steps}")
        log_info(f"  vs_beta_sched: {beta_schedule}")
        log_info(f"  ts_cnt       : {ts_cnt}")
        log_info(f"  device       : {device}")
        log_info(f"  cache_file   : {self.one_step_cache_file}")
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

class ScheduleBatch3:
    def __init__(self, args, model=None, data_loader=None):
        log_info(f"ScheduleBatch3() =======================================")
        self.vs = VarSimulator3(model, data_loader, data_limit=args.vs_data_limit,
                                steps=args.vs_steps, beta_schedule=args.vs_beta_sched,
                                cache_dir=args.output_dir, device=args.device)
        self.args           = args
        self.lr             = args.lr
        self.output_dir     = args.output_dir
        self.n_epochs       = args.n_epochs
        self.ab_low         = args.ab_low
        self.ab_low_lambda  = args.ab_low_lambda
        self.device         = args.device
        self.log_interval   = args.log_interval

    def train(self, order=1):
        log_info(f"ScheduleBatch3::train()")
        if not os.path.exists(self.output_dir):
            log_info(f"  os.makedirs({self.output_dir})")
            os.makedirs(self.output_dir)
        m_arr = [f"lr           : {self.lr}",
                 f"n_epochs     : {self.n_epochs}",
                 f"ab_low       : {self.ab_low}",
                 f"ab_low_lambda: {self.ab_low_lambda}",
                 f"vs_batch_size: {self.args.vs_batch_size}",
                 f"vs_data_limit: {self.args.vs_data_limit}",
                 f"initial_seed : {torch.initial_seed()}"]  # message array
        [log_info('  ' + m) for m in m_arr]
        f_path = os.path.join(self.output_dir, f"abDetail_schedule_batch3.txt")
        log_info(f"  output_file  : {f_path}")

        vrg_m = ScheduleAlphaModel(out_channels=self.args.ab_steps)
        vrg_m.to(self.device)
        optimizer = optim.SGD(vrg_m.parameters(), lr=self.lr, momentum=0.0)
        vrg_m.train()
        start_time = time.time()
        loss_low = None
        loss_ori = None
        e_cnt = self.n_epochs
        for e_idx in range(0, e_cnt):
            optimizer.zero_grad()
            alpha, aacum = vrg_m()
            weight_arr, idx_arr = self.vs(aacum, include_index=True)
            loss_var, coef, weight_arr, numerator, sub_var = self.calc_loss(alpha, aacum, weight_arr, order)
            if loss_ori is None: loss_ori = loss_var.item()
            aa_min = aacum[-1]
            loss_min = torch.square(aa_min - self.ab_low) * self.ab_low_lambda
            loss = torch.add(loss_var, loss_min)
            loss.backward()
            vrg_m.gradient_clip()
            optimizer.step()
            mm = list(m_arr)
            mm.append(f"Epoch       : {e_idx:06d}; loss:{loss:05.6f} = {loss_var:05.6f} + {loss_min:05.6f}")
            mm.append(f"loss_var    : {loss_ori:10.6f} => {loss_var:10.6f}")
            mm.append(f"vrg_m.out_ch: {vrg_m.out_channels}")
            f_path = os.path.join(self.output_dir, f"abDetail_schedule_batch3_{e_idx:03d}.txt")
            self.detail_save(f_path, alpha, aacum, idx_arr, weight_arr, coef, numerator, sub_var, mm)

            if e_idx % self.log_interval == 0 or e_idx == e_cnt - 1:
                elp, eta = utils.get_time_ttl_and_eta(start_time, e_idx, e_cnt)
                log_info(f"E{e_idx:05d}/{e_cnt} loss: {loss_var:.5f} {loss_min:.5f}."
                       f" a:{alpha[0]:.8f}~{alpha[-1]:.8f};"
                       f" aa:{aacum[0]:.8f}~{aacum[-1]:.5f}. elp:{elp}, eta:{eta} ------------------")
                if loss_low is None or loss_low > loss.item():
                    loss_low = loss.item()
                # if
            # if
        # for e_idx
        return f_path

    def calc_loss(self, alpha, aacum, weight_arr, order):
        if order == 1:
            loss_var, coef, numerator, sub_var = ScheduleBase.accumulate_variance(alpha, aacum, weight_arr, True)
            return loss_var, coef, weight_arr, numerator, sub_var
        elif order == 2:
            return self.calc_loss_order2(aacum, weight_arr)
        elif order == 3:
            return self.calc_loss_order3_v2(aacum, weight_arr)
        else:
            raise ValueError(f"Unsupported order {order}")

    @staticmethod
    def calc_loss_order2(aacum, weight_arr):
        a_cnt = len(aacum)
        if a_cnt % 2 == 0:
            # aacum index and weight index. if a_cnt is 8, then index is 0 ~ 7
            idx_a = list(range(1, a_cnt, 2))  # [1, 3, 5, 7]
            idx_w = list(range(0, a_cnt, 2))  # [0, 2, 4, 6]
        else:
            # aacum index and weight index. if a_cnt is 9, then index is 0 ~ 8
            idx_a = list(range(0, a_cnt, 2))  # [0, 2, 4, 6, 8]
            idx_w = list(range(1, a_cnt, 2))  # [1, 3, 5, 7]
            idx_w = [0] + idx_w  # [0, 1, 3, 5, 7]
        idx_a = torch.tensor(idx_a, dtype=torch.long, device=aacum.device)
        idx_w = torch.tensor(idx_w, dtype=torch.long, device=aacum.device)
        aa2 = torch.index_select(aacum, dim=0, index=idx_a)  # new aacum
        tmp = [torch.ones((1,), device=aacum.device), aa2[:-1]]
        tmp = torch.cat(tmp, dim=0)
        al2 = aa2 / tmp  # new alpha
        wt1 = torch.index_select(weight_arr, dim=0, index=idx_a)
        wt2 = torch.index_select(weight_arr, dim=0, index=idx_w)  # new weight
        wt2 += 1.0 * wt1  # plus some weight
        loss_var, c, n, s = ScheduleBase.accumulate_variance(al2, aa2, wt2, True)
        coef = torch.zeros_like(aacum)
        numerator = torch.zeros_like(aacum)
        sub_var = torch.zeros_like(aacum)
        weight = torch.zeros_like(aacum)
        coef[idx_a] = c
        numerator[idx_a] = n
        sub_var[idx_a] = s
        weight[idx_a] = wt2
        return loss_var, coef, weight, numerator, sub_var

    @staticmethod
    def calc_loss_order3(aacum, weight_arr):
        """
        For detailed explanation, please see the doc: Readme_DPM_Solver3_predefined.docx
        :param aacum:
        :param weight_arr:
        :return:
        """
        a_cnt = len(aacum)
        if a_cnt % 3 == 0:
            # if a_cnt is  9, then index will be 0 ~ 8.
            # Then inner jump size array is [3, 3, 2, 1]
            # aacum idx will be [0, 2, 5, 8]
            # weight index is complicated. For the case of jump size 3, each step involves 2 weights.
            # weight series 1 : [0, 1, 5, 8]
            # weight series 2:  [      3, 6]
            # aacum index and weight index.
            idx_a = [0, 2] + list(range(5, a_cnt, 3))  # [0, 2, 5, 8]
            idx_w = [0, 1] + list(range(5, a_cnt, 3))  # [0, 1, 5, 8] weight series 1
            idx_v = list(range(3, a_cnt, 3))           # [      3, 6] weight series 2
            idx_a = torch.tensor(idx_a, dtype=torch.long, device=aacum.device)
            idx_w = torch.tensor(idx_w, dtype=torch.long, device=aacum.device)
            idx_v = torch.tensor(idx_v, dtype=torch.long, device=aacum.device)
            aa3 = torch.index_select(aacum, dim=0, index=idx_a)       # new aacum
            wt3 = torch.index_select(weight_arr, dim=0, index=idx_w)  # new weight
            wtv = torch.index_select(weight_arr, dim=0, index=idx_v)
            wt3[2:] += wtv # append series 2 into series 1.
        elif a_cnt % 3 == 1:
            # if a_cnt is 10, then index will be 0 ~ 9.
            # Then inner jump size array [3, 3, 3, 1]
            # aacum idx will be [0, 3, 6, 9]
            # weight series 1 : [0, 3, 6, 9]
            # weight series 2:  [   1, 4, 7]
            # aacum index and weight index.
            idx_a = list(range(0, a_cnt, 3))  # [0, 3, 6, 9]
            idx_w = list(range(0, a_cnt, 3))  # [0, 3, 6, 9] weight series 1
            idx_v = list(range(1, a_cnt, 3))  # [   1, 4, 7] weight series 2
            idx_a = torch.tensor(idx_a, dtype=torch.long, device=aacum.device)
            idx_w = torch.tensor(idx_w, dtype=torch.long, device=aacum.device)
            idx_v = torch.tensor(idx_v, dtype=torch.long, device=aacum.device)
            aa3 = torch.index_select(aacum, dim=0, index=idx_a)       # new aacum
            wt3 = torch.index_select(weight_arr, dim=0, index=idx_w)  # new weight
            wtv = torch.index_select(weight_arr, dim=0, index=idx_v)
            wt3[1:] += wtv # append series 2 into series 1.
        else: # a_cnt % 3 == 2
            # If a_cnt is 11, then the index will be 0 ~ 10
            # Then inner jump size array [3, 3, 3, 2]
            # aacum idx will be [1, 4, 7, 10]
            # weight series 1 : [0, 4, 7, 10]
            # weight series 2:  [   2, 5,  8]
            # aacum index and weight index.
            idx_a = list(range(1, a_cnt, 3))        # [1, 4, 7, 10]
            idx_w = [0] + list(range(4, a_cnt, 3))  # [0, 4, 7, 10] weight series 1
            idx_v = list(range(2, a_cnt, 3))        # [   2, 5,  8] weight series 2
            idx_a = torch.tensor(idx_a, dtype=torch.long, device=aacum.device)
            idx_w = torch.tensor(idx_w, dtype=torch.long, device=aacum.device)
            idx_v = torch.tensor(idx_v, dtype=torch.long, device=aacum.device)
            aa3 = torch.index_select(aacum, dim=0, index=idx_a)       # new aacum
            wt3 = torch.index_select(weight_arr, dim=0, index=idx_w)  # new weight
            wtv = torch.index_select(weight_arr, dim=0, index=idx_v)
            wt3[1:] += wtv # append series 2 into series 1.
        tmp = [torch.ones((1,), device=aacum.device), aa3[:-1]]
        tmp = torch.cat(tmp, dim=0)
        al3 = aa3 / tmp  # new alpha
        loss_var, c, n, s = ScheduleBase.accumulate_variance(al3, aa3, wt3, True)
        coef = torch.zeros_like(aacum)
        numerator = torch.zeros_like(aacum)
        sub_var = torch.zeros_like(aacum)
        weight = torch.zeros_like(aacum)
        coef[idx_a] = c
        numerator[idx_a] = n
        sub_var[idx_a] = s
        weight[idx_a] = wt3
        return loss_var, coef, weight, numerator, sub_var

    @staticmethod
    def calc_loss_order3_v2(aacum, weight_arr):
        """
        the original code follows the DPM Solver-3 formula, but not working well with time_uniform.
        So try the below code.
        :param aacum:
        :param weight_arr:
        :return:
        """
        a_cnt = len(aacum)
        if a_cnt % 3 == 0:
            # if a_cnt is  9, then index will be 0 ~ 8.
            idx_a = list(range(2, a_cnt, 3))  # [2, 5, 8]
            idx_w = list(range(2, a_cnt, 3))  # [2, 5, 8] weight series 1
            idx_v = list(range(0, a_cnt, 3))  # [0, 3, 6] weight series 2
            idx_u = list(range(1, a_cnt, 3))  # [1, 4, 7] weight series 3
            idx_a = torch.tensor(idx_a, dtype=torch.long, device=aacum.device)
            idx_w = torch.tensor(idx_w, dtype=torch.long, device=aacum.device)
            idx_v = torch.tensor(idx_v, dtype=torch.long, device=aacum.device)
            idx_u = torch.tensor(idx_u, dtype=torch.long, device=aacum.device)
            aa3 = torch.index_select(aacum, dim=0, index=idx_a)       # new aacum
            wt3 = torch.index_select(weight_arr, dim=0, index=idx_w)  # new weight
            wtv = torch.index_select(weight_arr, dim=0, index=idx_v)
            wtu = torch.index_select(weight_arr, dim=0, index=idx_u)
            wt3 += wtv # append series 2 into series 1.
            wt3 += wtu # append series 3 into series 1.
        elif a_cnt % 3 == 1:
            # if a_cnt is 10, then index will be 0 ~ 9.
            idx_a = [0, 1, 2, 3] + list(range(6, a_cnt, 3)) # [0, 1, 2, 3, 6, 9]
            idx_w = [0, 1, 2, 3] + list(range(6, a_cnt, 3)) # [0, 1, 2, 3, 6, 9] weight series 1
            idx_v = list(range(4, a_cnt, 3))                # [            4, 7] weight series 2
            idx_u = list(range(5, a_cnt, 3))                # [            5, 8] weight series 3
            idx_a = torch.tensor(idx_a, dtype=torch.long, device=aacum.device)
            idx_w = torch.tensor(idx_w, dtype=torch.long, device=aacum.device)
            idx_v = torch.tensor(idx_v, dtype=torch.long, device=aacum.device)
            idx_u = torch.tensor(idx_u, dtype=torch.long, device=aacum.device)
            aa3 = torch.index_select(aacum, dim=0, index=idx_a)       # new aacum
            wt3 = torch.index_select(weight_arr, dim=0, index=idx_w)  # new weight
            wtv = torch.index_select(weight_arr, dim=0, index=idx_v)
            wtu = torch.index_select(weight_arr, dim=0, index=idx_u)
            wt3[4:] += wtv + wtu # append series 2 & 3 into series 1.
        else: # a_cnt % 3 == 2
            # If a_cnt is 11, then the index will be 0 ~ 10
            idx_a = list(range(1, a_cnt, 3))        # [1, 4, 7, 10]
            idx_w = [0] + list(range(4, a_cnt, 3))  # [0, 4, 7, 10] weight series 1
            idx_v = list(range(2, a_cnt, 3))        # [   2, 5,  8] weight series 2
            idx_u = [1] + list(range(3, a_cnt, 3))  # [1, 3, 6,  9] weight series 3
            idx_a = torch.tensor(idx_a, dtype=torch.long, device=aacum.device)
            idx_w = torch.tensor(idx_w, dtype=torch.long, device=aacum.device)
            idx_v = torch.tensor(idx_v, dtype=torch.long, device=aacum.device)
            idx_u = torch.tensor(idx_u, dtype=torch.long, device=aacum.device)
            aa3 = torch.index_select(aacum, dim=0, index=idx_a)       # new aacum
            wt3 = torch.index_select(weight_arr, dim=0, index=idx_w)  # new weight
            wtv = torch.index_select(weight_arr, dim=0, index=idx_v)
            wtu = torch.index_select(weight_arr, dim=0, index=idx_u)
            wt3 += wtu # append series 2 & 3 into series 1.
            wt3[1:] += wtv
        tmp = [torch.ones((1,), device=aacum.device), aa3[:-1]]
        tmp = torch.cat(tmp, dim=0)
        al3 = aa3 / tmp  # new alpha
        loss_var, c, n, s = ScheduleBase.accumulate_variance(al3, aa3, wt3, True)
        coef = torch.zeros_like(aacum)
        numerator = torch.zeros_like(aacum)
        sub_var = torch.zeros_like(aacum)
        weight = torch.zeros_like(aacum)
        coef[idx_a] = c
        numerator[idx_a] = n
        sub_var[idx_a] = s
        weight[idx_a] = wt3
        return loss_var, coef, weight, numerator, sub_var

    @staticmethod
    def detail_save(f_path, alpha, aacum, idx_arr, weight_arr, coef, numerator, sub_var, m_arr):
        combo = []
        for i in range(len(aacum)):
            s = f"{aacum[i]:8.6f}: {idx_arr[i]:3.0f}: {alpha[i]:8.6f};" \
                f" {coef[i]:8.6f}*{weight_arr[i]:11.6f}={numerator[i]:9.6f};" \
                f" {numerator[i]:9.6f}/{aacum[i]:8.6f}={sub_var[i]:10.6f}"
            s = s.replace('0.000000', '0.0     ')
            combo.append(s)
        m_arr.append('aacum : ts : alpha   ; coef    *weight     =numerator; numerator/aacum   =sub_var')
        log_info(f"Save file: {f_path}")
        with open(f_path, 'w') as f_ptr:
            [f_ptr.write(f"# {m}\n") for m in m_arr]
            [f_ptr.write(f"{s}\n") for s in combo]
        # with

# class
