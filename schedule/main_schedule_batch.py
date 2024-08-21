"""
Find beta schedule by vivid variance, in batch.
Vivid variance is also final variance by default.
"""
import argparse
import sys
import os

from schedule.schedule_batch import ScheduleBatch
from torch.backends import cudnn

# add current dir and parent dir into python-path.
# this it to facilitate the Linux start.exe file.
cur_dir = os.path.dirname(__file__)
prt_dir = os.path.dirname(cur_dir)  # parent dir
if cur_dir not in sys.path:
    sys.path.append(cur_dir)
if prt_dir not in sys.path:
    sys.path.append(prt_dir)

from base import *

log_fn = utils.log_info

torch.set_printoptions(sci_mode=False)
torch.autograd.set_detect_anomaly(True)

def parse_args():
    parser = argparse.ArgumentParser(description=globals()["__doc__"])
    parser.add_argument('--gpu_ids', type=int, nargs='+', default=[7])
    parser.add_argument('--todo', type=str, default='train')
    parser.add_argument("--n_epochs", type=int, default=1000)
    parser.add_argument('--log_interval', type=int, default=200)
    parser.add_argument('--lr', type=float, default=0.000001)
    parser.add_argument('--lp', type=float, default=0.01, help='learning_portion')
    parser.add_argument('--output_dir', type=str, default='./output7_vividvar')
    parser.add_argument('--alpha_bar_dir', type=str, default='./output7_vividvar/alpha_bar_dir')
    parser.add_argument('--aa_low', type=float, default=0.0001, help="Alpha Accum lower bound")
    parser.add_argument("--aa_low_lambda", type=float, default=10000000)
    parser.add_argument("--weight_file", type=str, default='./output7_vividvar/res_mse_avg_list.txt')
    parser.add_argument("--beta_schedule", type=str, default="linear")
    args = parser.parse_args()

    if not args.weight_file:
        raise Exception(f"Argument weight_file is empty")

    # add device
    gpu_ids = args.gpu_ids
    log_fn(f"gpu_ids : {gpu_ids}")
    args.device = f"cuda:{gpu_ids[0]}" if torch.cuda.is_available() and gpu_ids else "cpu"
    cudnn.benchmark = True

    if not os.path.exists(args.output_dir):
        log_fn(f"os.makedirs({args.output_dir})")
        os.makedirs(args.output_dir)

    return args

def main():
    args = parse_args()
    log_fn(f"pid : {os.getpid()}")
    log_fn(f"cwd : {os.getcwd()}")
    log_fn(f"args: {args}")
    sb = ScheduleBatch(args)
    f_path = './output7_vividvar/alpha_bar_dir/dpm_alphaBar_steps20.txt'
    sb.schedule_single(f_path, sb.lr, sb.lp)
    return 0

if __name__ == "__main__":
    sys.exit(main())
