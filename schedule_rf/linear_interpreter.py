import math
import os.path

import torch
from torch import Tensor

from utils import log_info


class LinearInterpreter:
    """
    Linear Interpreter. Use binary search.
    It is initialized with x_arr and y_arr. And given an x_value, it returns its y_value.
    By default, x_value is in [0, 1].
    """
    def __init__(self, x_arr=None, y_arr=None, y_arr_file=None, log_fn=log_info):
        log_fn(f"LinearInterpreter()")
        log_fn(f"  input x_arr     : {'None' if x_arr is None else 'len=' + str(len(x_arr))}")
        log_fn(f"  input y_arr     : {'None' if y_arr is None else 'len=' + str(len(y_arr))}")
        log_fn(f"  input y_arr_file: {y_arr_file}")
        if y_arr:
            log_fn(f"  Ignored y_arr_file, and just use y_arr")
            self.y_arr = y_arr
        else:
            log_fn(f"  Loading y_arr_file . . .")
            self.y_arr = self._load_y_arr_from_file(y_arr_file)
        if x_arr:
            self.x_arr = x_arr
        else:
            log_fn(f"  set up x_arr with y_arr")
            cnt = len(self.y_arr)
            self.x_arr = [float(i) / cnt for i in range(1, cnt + 1)]
        if len(self.x_arr) != len(self.y_arr):
            raise ValueError(f"x_arr and y_arr length not match: {len(self.x_arr)}, {len(self.y_arr)}")
        if len(self.x_arr) < 2:
            raise ValueError(f"x_arr length is less than 2. x_arr: {self.x_arr}")

        self.x_arr = torch.tensor(self.x_arr, dtype=torch.double)  # input x, output y.
        self.y_arr = torch.tensor(self.y_arr, dtype=torch.double)
        self.log_fn = log_fn
        self.val_cnt = len(self.y_arr)
        self.iter_cnt = math.ceil(math.log(self.val_cnt, 2))
        arr2str = lambda arr: '[' + ', '.join(["{:9.4f}".format(f) for f in arr]) + ']'
        log_fn(f"  val_cnt   : {self.val_cnt}")
        log_fn(f"  iter_cnt  : {self.iter_cnt}")
        log_fn(f"  x_arr len : {len(self.x_arr)}")
        log_fn(f"  y_arr len : {len(self.y_arr)}")
        log_fn(f"  x_arr[:5] : {arr2str(self.x_arr[:5])}")
        log_fn(f"  x_arr[-5:]: {arr2str(self.x_arr[-5:])}")
        log_fn(f"  y_arr[:5] : {arr2str(self.y_arr[:5])}")
        log_fn(f"  y_arr[-5:]: {arr2str(self.y_arr[-5:])}")

    @staticmethod
    def _load_y_arr_from_file(file_path):
        if not os.path.exists(file_path):
            raise ValueError(f"File not exist: {file_path}")
        with open(file_path, 'r') as fptr:
            lines = fptr.readlines()
        y_arr = []
        for line in lines:
            line = line.strip()
            if line == '' or line.startswith('#'):
                continue
            y_val = float(line.split(':')[0])  # in case line is like: 33.33: <index>
            y_arr.append(y_val)
        # for
        return y_arr

    def __call__(self, x_value: Tensor, include_index=False):
        return self._binary_search(x_value, include_index)

    def _binary_search(self, x_value: Tensor, include_index=False):
        """Use binary search"""
        # define left bound index and right bound index
        lbi = torch.zeros_like(x_value, dtype=torch.long)
        rbi = torch.ones_like(x_value, dtype=torch.long)
        rbi *= (self.val_cnt - 1)
        for _ in range(self.iter_cnt):
            mdi = torch.floor(torch.div(lbi + rbi,  2))  # middle index
            mdi = mdi.long()
            flag0 = x_value >= self.x_arr[mdi]
            flag1 = ~flag0
            lbi[flag0] = mdi[flag0]
            rbi[flag1] = mdi[flag1]
        # for
        # after iteration, lbi will be the target index
        res = self.y_arr[lbi]

        # But the input x_value value may have difference with x_arr. So here
        # we handle the difference and make the result smooth
        # Firstly, find the right-hand index: re-use variable "rbi"
        rbi = torch.ones_like(x_value, dtype=torch.long)
        rbi *= (self.val_cnt - 1)
        rbi = torch.minimum(torch.add(lbi, 1), rbi)

        # make the result smooth
        flag = torch.lt(lbi, rbi)
        lb_arr = self.x_arr[lbi]
        rb_arr = self.x_arr[rbi]
        portion = (lb_arr[flag] - x_value[flag]) / (lb_arr[flag] - rb_arr[flag])
        res[flag] = res[flag] * (1 - portion) + self.y_arr[rbi][flag] * portion

        # a2s = lambda x: ', '.join([f"{i:3d}" for i in x[:5]])  # arr to str
        # self.log_fn(f"lbi[:5]  : {a2s(lbi[:5])}")
        # self.log_fn(f"lbi[-5:] : {a2s(lbi[-5:])}")
        if include_index:
            return res, lbi
        return res

    def to(self, device):
        self.x_arr = self.x_arr.to(device)
        self.y_arr = self.y_arr.to(device)

# class

def test_fn():
    """ unit test"""
    x_arr = [0, 1, 2, 3, 4, 5, 6]
    y_arr = [0, 1, 2, 3, 2, 1, 0]
    # y_arr = [5, 4, 3, 2, 1, 0]
    li = LinearInterpreter(x_arr, y_arr)
    in_tensor = torch.tensor([0.9998, 1.8985, 2.6473, 3.3407, 4.0940])
    y_tensor = li(in_tensor)
    for i in range(len(in_tensor)):
        print(f"x:{in_tensor[i]:6.4f} -> y:{y_tensor[i]:6.4f}")

if __name__ == '__main__':
    test_fn()
