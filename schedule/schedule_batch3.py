import os
import time
import torch
from torch import optim

import utils
from schedule.base import ScheduleBase, ScheduleAlphaModel
from schedule.var_simulator3 import VarSimulator3

log_fn = utils.log_info

class ScheduleBatch3:
    def __init__(self, args, model=None, data_loader=None):
        log_fn(f"ScheduleBatch3() =======================================")
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
        log_fn(f"ScheduleBatch3::train()")
        if not os.path.exists(self.output_dir):
            log_fn(f"  os.makedirs({self.output_dir})")
            os.makedirs(self.output_dir)
        m_arr = [f"lr           : {self.lr}",
                 f"n_epochs     : {self.n_epochs}",
                 f"ab_low       : {self.ab_low}",
                 f"ab_low_lambda: {self.ab_low_lambda}",
                 f"vs_batch_size: {self.args.vs_batch_size}",
                 f"vs_data_limit: {self.args.vs_data_limit}",
                 f"initial_seed : {torch.initial_seed()}"]  # message array
        [log_fn('  ' + m) for m in m_arr]
        f_path = os.path.join(self.output_dir, f"abDetail_schedule_batch3.txt")
        log_fn(f"  output_file  : {f_path}")

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
                log_fn(f"E{e_idx:05d}/{e_cnt} loss: {loss_var:.5f} {loss_min:.5f}."
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
        log_fn(f"Save file: {f_path}")
        with open(f_path, 'w') as f_ptr:
            [f_ptr.write(f"# {m}\n") for m in m_arr]
            [f_ptr.write(f"{s}\n") for s in combo]
        # with

# class
