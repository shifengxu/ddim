import argparse
import torch.nn as nn
import math
import os
import datetime
import time

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def count_parameters(model: nn.Module, log_fn=None):
    def prt(x):
        if log_fn: log_fn(x)

    prt(f"count_parameters({type(model)}) ------------")
    prt('  requires_grad  name  count  size')
    counter = 0
    for name, param in model.named_parameters():
        s_list = list(param.size())
        prt(f"  {param.requires_grad} {name} {param.numel()} = {s_list}")
        c = param.numel()
        counter += c
    # for
    str_size = convert_size_str(counter)
    prt(f"  total  : {counter} {str_size}")
    return counter, str_size

def convert_size_str(size_bytes):
    if size_bytes == 0:
        return "0B"
    size_name = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)
    return "%s %s" % (s, size_name[i])

def extract_ts_range(path):
    """
    Extract timestamp range from file path string
    :param path: "./exp/model_stack_epoch500/ckpt_750-1000.pth"
    :return:
    """
    base = os.path.basename(path)       # ckpt_750-1000.pth
    stem, _ = os.path.splitext(base)    # ckpt_750-1000
    temp = stem.split("_")              # [ckpt, 750-1000]
    nums = temp[-1].split("-")          # ["750", "1000"]
    if len(nums) != 2:
        raise ValueError(f"Cannot find timestamp range from ckpt file: {path}")
    return [int(nums[0]), int(nums[1])]

def log_info(*args):
    dtstr = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    print(f"[{dtstr}]", *args)

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

def make_dirs_if_need(*f_dirs, log_fn=log_info):
    f_path = os.path.join(*f_dirs)
    if os.path.exists(f_path):
        return f_path
    log_fn(f"mkdir: {f_path}")
    os.makedirs(f_path)
    return f_path

def output_list(lst, name, ftm_str="{:10.8f}", log_fn=log_info):
    def num2str(num_arr):
        flt_arr = [float(n) for n in num_arr]
        str_arr = [ftm_str.format(f) for f in flt_arr]
        return " ".join(str_arr)

    if lst is None or len(lst) == 0:
        log_fn(f"{name}: {lst}")
        return
    cnt = len(lst)
    for i in range(0, cnt, 10):
        r = min(i+10, cnt)  # right bound
        log_fn(f"{name}[{i:04d}~]: {num2str(lst[i:r])}")

def save_list(lst, name, f_path: str, msg=None):
    def num2str(num_arr):
        flt_arr = [float(n) for n in num_arr]
        str_arr = [f"{f:.11f}" for f in flt_arr]
        return " ".join(str_arr)

    cnt = len(lst) if lst is not None else 0
    with open(f_path, 'w') as f_ptr:
        if msg:
            f_ptr.write(f"# {msg}\n")
        for i in range(0, cnt, 10):
            r = min(i + 10, cnt)  # right bound
            f_ptr.write(f"{name}[{i:04d}~]: {num2str(lst[i:r])}\n")
    # with

def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace
