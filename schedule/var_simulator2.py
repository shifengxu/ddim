import math

import torch
from torch import Tensor

from base import ScheduleBase
from utils import log_info


class VarSimulator2:
    """
    Variance simulator2. Use binary search
    """
    def __init__(self, beta_schedule, y_arr, log_fn=log_info, mode='vivid'):
        self.ts_cnt = len(y_arr)
        self.x_arr = ScheduleBase.get_alpha_cumprod(beta_schedule, self.ts_cnt)
        self.y_arr = y_arr
        self.log_fn = log_fn
        self.mode = mode
        log_fn(f"VarSimulator2()")
        log_fn(f"  schedule  : {beta_schedule}")
        log_fn(f"  mode      : {mode}")
        log_fn(f"  ts_cnt    : {self.ts_cnt}")
        log_fn(f"  x_arr     : {len(self.x_arr)}")
        log_fn(f"  y_arr     : {len(self.y_arr)}")
        log_fn(f"  x_arr[:5] : {self.arr_to_str(self.x_arr[:5])}")
        log_fn(f"  x_arr[-5:]: {self.arr_to_str(self.x_arr[-5:])}")
        log_fn(f"  y_arr[:5] : {self.arr_to_str(self.y_arr[:5])}")
        log_fn(f"  y_arr[-5:]: {self.arr_to_str(self.y_arr[-5:])}")

    @staticmethod
    def arr_to_str(arr, precision_cnt=8):
        fs = "{:." + str(precision_cnt) + "f}"
        s_arr = [fs.format(f) for f in arr]
        res = ', '.join(s_arr)
        return f"[{res}]"

    def __call__(self, aacum: Tensor):
        if self.mode == 'static':
            return self.y_arr
        elif self.mode == 'vivid' or self.mode == 'dynamic':
            return self._binary_search(aacum)
        else:
            raise Exception(f"Unknown mode: {self.mode}")

    def _binary_search(self, aacum: Tensor):
        """Use binary search"""
        # define left bound index and right bound index
        lbi = torch.zeros_like(aacum, dtype=torch.long)
        rbi = torch.ones_like(aacum, dtype=torch.long)
        rbi *= (self.ts_cnt - 1)
        iter_cnt = math.ceil(math.log(self.ts_cnt, 2))
        for _ in range(iter_cnt):
            mdi = torch.floor(torch.div(lbi + rbi,  2))  # middle index
            mdi = mdi.long()
            flag0 = aacum <= self.x_arr[mdi]
            flag1 = ~flag0
            lbi[flag0] = mdi[flag0]
            rbi[flag1] = mdi[flag1]
        # for
        # after iteration, lbi will be the target index
        flag0 = aacum <= self.x_arr[rbi]  # handle the case aacum == x_arr[-1]
        lbi[flag0] = rbi[flag0]

        # a2s = lambda x: ', '.join([f"{i:3d}" for i in x[:5]])  # arr to str
        # self.log_fn(f"lbi[:5]  : {a2s(lbi[:5])}")
        # self.log_fn(f"lbi[-5:] : {a2s(lbi[-5:])}")
        return self.y_arr[lbi]

    def to(self, device):
        self.x_arr = self.x_arr.to(device)
        self.y_arr = self.y_arr.to(device)

# class

def test():
    """ unit test"""

if __name__ == '__main__':
    test()
