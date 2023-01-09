import numpy as np
import torch

from base import ScheduleBase
from utils import log_info, output_list


class VarSimulator:
    """Variance simulator: by numpy.polyfit"""
    def __init__(self, beta_schedule, y_arr, w=None, deg=8, log_fn=log_info):
        self.deg = deg
        self.w = w
        self.ts_cnt = len(y_arr)
        self.x_arr = ScheduleBase.get_alpha_cumprod(beta_schedule, self.ts_cnt)
        self.y_arr = y_arr
        self.p_arr = np.polyfit(self.x_arr, y_arr, deg, w=w)
        self.log_fn = log_fn
        log_fn(f"VarSimulator()")
        log_fn(f"  schedule  : {beta_schedule}")
        log_fn(f"  ts_cnt    : {self.ts_cnt}")
        log_fn(f"  x_arr     : {len(self.x_arr)}")
        log_fn(f"  y_arr     : {len(self.y_arr)}")
        log_fn(f"  deg       : {deg}")
        log_fn(f"  weight    : {None if w is None else len(w)}")
        log_fn(f"  x_arr[:5] : {self.arr_to_str(self.x_arr[:5])}")
        log_fn(f"  x_arr[-5:]: {self.arr_to_str(self.x_arr[-5:])}")
        log_fn(f"  y_arr[:5] : {self.arr_to_str(self.y_arr[:5])}")
        log_fn(f"  y_arr[-5:]: {self.arr_to_str(self.y_arr[-5:])}")
        log_fn(f"  w[:5]     : {self.arr_to_str(self.w[:5])}") if w is not None else None
        log_fn(f"  w[-5:]    : {self.arr_to_str(self.w[-5:])}") if w is not None else None
        output_list(self.p_arr, 'p_arr', ftm_str="{:7.3f}")

    @staticmethod
    def arr_to_str(arr, precision_cnt=8):
        fs = "{:." + str(precision_cnt) + "f}"
        s_arr = [fs.format(f) for f in arr]
        res = ', '.join(s_arr)
        return f"[{res}]"

    def __call__(self, aacum_val):
        if torch.is_tensor(aacum_val):
            return self._call_tensor(aacum_val)

        deg = self.deg
        p_arr = self.p_arr
        res = p_arr[deg]
        radix = 1.
        deg -= 1
        while deg >= 0:
            radix *= aacum_val
            res += p_arr[deg] * radix
        return res

    def _call_tensor(self, aacum):
        deg = self.deg
        p_arr = self.p_arr
        res = torch.ones_like(aacum, dtype=torch.float64, device=aacum.device)
        res *= p_arr[deg]
        deg -= 1
        radix = torch.ones_like(aacum, dtype=torch.float64, device=aacum.device)
        while deg >= 0:
            radix *= aacum
            res += p_arr[deg] * radix
            deg -= 1
        return res

# class

def test():
    """ unit test"""

if __name__ == '__main__':
    test()
