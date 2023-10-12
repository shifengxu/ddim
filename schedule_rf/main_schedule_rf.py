import argparse
import sys
import os
import random
import numpy as np
import torch
from torch.backends import cudnn

import utils
from schedule_rf.linear_interpreter import LinearInterpreter
from schedule_rf.scheduler_rf import ScheduleRectifiedFlow

current_dir = os.path.dirname(__file__)     # current dir
parent_dir = os.path.dirname(current_dir)   # parent dir
# add current dir and parent dir into python-path.
# this it to facilitate the Linux start.exe file.
if current_dir not in sys.path:
    sys.path.append(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

log_fn = utils.log_info

torch.set_printoptions(sci_mode=False)
torch.autograd.set_detect_anomaly(True)

def parse_args():
    parser = argparse.ArgumentParser(description=globals()["__doc__"])
    parser.add_argument('--gpu_ids', type=int, nargs='+', default=[5])
    parser.add_argument('--lr', type=float, default=0.000001)
    parser.add_argument('--n_epochs', type=int, default=500)
    parser.add_argument('--max_ts', type=float, default=0.94, help='max timestep')
    parser.add_argument('--loss_decay', type=float, default=1)
    parser.add_argument('--delta_expo', type=float, default=2)
    parser.add_argument('--steps', type=int, default=10, help='steps count for sampling')
    parser.add_argument('--log_interval', type=int, default=50)
    parser.add_argument("--seed", type=int, default=1234, help="Random seed. 0 means ignore")
    parser.add_argument('--mse_file', type=str, default='./output6_schedule_rf/mse_list_500.txt')
    parser.add_argument('--output_file', type=str, default='./output6_schedule_rf/details_saved.txt')
    args = parser.parse_args()

    if not args.mse_file:
        raise Exception(f"Argument mse_file is empty")

    # add device
    gpu_ids = args.gpu_ids
    log_fn(f"gpu_ids : {gpu_ids}")
    args.device = f"cuda:{gpu_ids[0]}" if torch.cuda.is_available() and gpu_ids else "cpu"

    # set random seed
    seed = args.seed  # if seed is 0. then ignore it.
    log_fn(f"args.seed : {seed}")
    if seed:
        log_fn(f"  torch.manual_seed({seed})")
        log_fn(f"  np.random.seed({seed})")
        log_fn(f"  random.seed({seed})")
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
    if seed and torch.cuda.is_available():
        log_fn(f"  torch.cuda.manual_seed({seed})")
        log_fn(f"  torch.cuda.manual_seed_all({seed})")
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    log_fn(f"final seed: torch.initial_seed(): {torch.initial_seed()}")

    cudnn.benchmark = True
    return args

def main():
    args = parse_args()
    log_fn(f"pid : {os.getpid()}")
    log_fn(f"cwd : {os.getcwd()}")
    log_fn(f"args: {args}")
    lip = LinearInterpreter(y_arr_file=args.mse_file)
    scheduler = ScheduleRectifiedFlow(weight_simulator=lip, steps=args.steps, max_ts=args.max_ts,
                                      loss_decay=args.loss_decay, delta_expo=args.delta_expo,
                                      lr=args.lr, n_epochs=args.n_epochs, log_interval=args.log_interval,
                                      output_file=args.output_file, device=args.device)
    scheduler.train()
    # from linear_interpreter import test_fn
    # test_fn()
    return 0

if __name__ == "__main__":
    sys.exit(main())
