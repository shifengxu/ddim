"""
Find beta schedule by vivid variance, in batch.
Vivid variance is also final variance by default.
"""
import argparse
import random
import sys
import os

import yaml
import torch.utils.data as data

cur_dir = os.path.dirname(__file__)
prt_dir = os.path.dirname(cur_dir)  # parent dir
if cur_dir not in sys.path:
    sys.path.append(cur_dir)
if prt_dir not in sys.path:
    sys.path.append(prt_dir)

from datasets import get_dataset
from models.ema import EMAHelper
from schedule.schedule_batch3 import ScheduleBatch3
from models.diffusion import Model
from torch.backends import cudnn

# add current dir and parent dir into python-path.
# this it to facilitate the Linux start.exe file.
from utils import dict2namespace, str2bool

from base import *

log_fn = utils.log_info

torch.set_printoptions(sci_mode=False)
torch.autograd.set_detect_anomaly(True)

def parse_args():
    parser = argparse.ArgumentParser(description=globals()["__doc__"])
    parser.add_argument('--gpu_ids', type=int, nargs='+', default=[7, 6])
    parser.add_argument("--data_dir", type=str, default="./exp")
    parser.add_argument("--config", type=str, default='./configs/cifar10.yml')
    parser.add_argument('--todo', type=str, default='train')
    parser.add_argument("--seed", type=int, default=1234, help="Random seed. 0 means ignore")

    parser.add_argument("--sample_ckpt_path", type=str, default="./exp/ema-cifar10-model-790000.ckpt")
    parser.add_argument("--ema_flag", type=str2bool, default=True)

    parser.add_argument('--output_dir', type=str, default='./output7_dpmSolver')
    parser.add_argument("--vs_batch_size", type=int, default=2000)
    parser.add_argument("--vs_data_limit", type=int, default=2000)
    parser.add_argument("--vs_steps", type=int, default=5)
    parser.add_argument("--vs_beta_sched", type=str, default="linear")

    parser.add_argument("--n_epochs", type=int, default=1000)
    parser.add_argument('--log_interval', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-6)
    parser.add_argument('--ab_low', type=float, default=0.0002, help="Alpha bar lower bound")
    parser.add_argument("--ab_low_lambda", type=float, default=1e9)
    parser.add_argument("--ab_steps", type=int, default=10)
    args = parser.parse_args()

    # add device
    gpu_ids = args.gpu_ids
    log_fn(f"gpu_ids : {gpu_ids}")
    args.device = f"cuda:{gpu_ids[0]}" if torch.cuda.is_available() and gpu_ids else "cpu"
    cudnn.benchmark = True

    if not os.path.exists(args.output_dir):
        log_fn(f"os.makedirs({args.output_dir})")
        os.makedirs(args.output_dir)

    # parse config file
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    new_config = dict2namespace(config)

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

    return args, new_config

def get_data_loaders(args, config):
    batch_size = args.vs_batch_size
    dataset, test_dataset = get_dataset(args, config)
    train_loader = data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=config.data.num_workers,
    )
    log_fn(f"train dataset and data loader:")
    log_fn(f"  root       : {dataset.root}")
    log_fn(f"  split      : {dataset.split}") if hasattr(dataset, 'split') else None
    log_fn(f"  len        : {len(dataset)}")
    log_fn(f"  batch_cnt  : {len(train_loader)}")
    log_fn(f"  batch_size : {batch_size}")
    log_fn(f"  shuffle    : True")
    log_fn(f"  num_workers: {config.data.num_workers}")

    test_loader = data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
    )
    log_fn(f"test dataset and loader:")
    log_fn(f"  root          : {test_dataset.root}")
    log_fn(f"  len           : {len(test_dataset)}")
    log_fn(f"  batch_cnt     : {len(test_loader)}")
    log_fn(f"  batch_size    : {batch_size}")
    log_fn(f"  shuffle       : False")
    log_fn(f"  num_workers   : {config.data.num_workers}")
    return train_loader, test_loader

def model_load_from_local(args, model):
    def apply_ema():
        log_fn(f"  ema_helper: EMAHelper()")
        ema_helper = EMAHelper()
        ema_helper.register(model)
        k = "ema_helper" if isinstance(states, dict) else -1
        log_fn(f"  ema_helper: load from states[{k}]")
        ema_helper.load_state_dict(states[k])
        log_fn(f"  ema_helper: apply to model {type(model).__name__}")
        ema_helper.ema(model)

    ckpt_path = args.sample_ckpt_path
    log_fn(f"load ckpt: {ckpt_path}")
    states = torch.load(ckpt_path, map_location=args.device)
    if 'model' not in states:
        log_fn(f"  !!! Not found 'model' in states. Will take it as pure model")
        model.load_state_dict(states)
    else:
        key = 'model' if isinstance(states, dict) else 0
        model.load_state_dict(states[key], strict=True)
        ckpt_tt = states.get('ts_type', 'discrete')
        model_tt = model.ts_type
        if ckpt_tt != model_tt:
            raise ValueError(f"ts_type not match. ckpt_tt={ckpt_tt}, model_tt={model_tt}")
        if not hasattr(args, 'ema_flag'):
            log_fn(f"  !!! Not found ema_flag in args. Assume it is true.")
            apply_ema()
        elif args.ema_flag:
            log_fn(f"  Found args.ema_flag: {args.ema_flag}.")
            apply_ema()
    # endif

    log_fn(f"  model({type(model).__name__}).to({args.device})")
    model = model.to(args.device)
    if len(args.gpu_ids) > 1:
        log_fn(f"  torch.nn.DataParallel(model, device_ids={args.gpu_ids})")
        model = torch.nn.DataParallel(model, device_ids=args.gpu_ids)
    return model

def main():
    args, config = parse_args()
    log_fn(f"pid : {os.getpid()}")
    log_fn(f"cwd : {os.getcwd()}")
    log_fn(f"args: {args}")
    model = Model(config)
    model_load_from_local(args, model)
    train_loader, test_loader = get_data_loaders(args, config)
    sb = ScheduleBatch3(args, model=model, data_loader=train_loader)
    sb.train()
    return 0

if __name__ == "__main__":
    sys.exit(main())
