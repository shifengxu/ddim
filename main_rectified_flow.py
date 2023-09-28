import argparse
import traceback

import logging
import yaml
import sys
import os
import torch
import numpy as np
import torch.backends.cudnn as cudnn

from runners.diffusion_sampling_rf import DiffusionSamplingRectifiedFlow
from runners.diffusion_training_rf import DiffusionTrainingRectifiedFlow
from utils import str2bool, dict2namespace

torch.set_printoptions(sci_mode=False)


def parse_args_and_config():
    parser = argparse.ArgumentParser(description=globals()["__doc__"])

    parser.add_argument("--config", type=str, default='./configs/cifar10.yml')
    parser.add_argument("--todo", type=str, default='sample', help="train|sample")
    parser.add_argument('--gpu_ids', nargs='+', type=int, default=[7])
    parser.add_argument("--n_epochs", type=int, default=1000, help="0 mean epoch number from config file")
    parser.add_argument('--lr', type=float, default=0.0002, help="learning rate")
    parser.add_argument("--seed", type=int, default=1234, help="Random seed. 0 means ignore")
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--test_interval", type=int, default=10, help='calc loss on test dataset. 0 means no calc.')

    # data
    parser.add_argument("--data_dir", type=str, default="./exp")
    parser.add_argument("--batch_size", type=int, default=200, help="0 mean to use size from config file")

    # model
    parser.add_argument('--ts_range', nargs='+', type=int, default=[0, 1000])
    parser.add_argument('--ts_stride', type=int, default=100)
    parser.add_argument("--beta_schedule", type=str, default="linear")
    parser.add_argument("--save_ckpt_interval", type=int, default=50)
    parser.add_argument("--save_ckpt_dir", type=str, default='./output0_tmp')

    # sampling
    parser.add_argument("--sample_count", type=int, default='50000', help="sample image count")
    parser.add_argument("--sample_batch_size", type=int, default=1000, help="0 mean from config file")
    parser.add_argument("--sample_ckpt_path", type=str, default='./output0_tmp/ckpt_rf_E0200.pth')
    parser.add_argument("--sample_ckpt_dir", type=str, default='')
    parser.add_argument("--sample_output_dir", type=str, default="./output0_tmp/generated_rf")

    # training
    parser.add_argument('--ema_flag', type=str2bool, default=True, help='EMA flag')
    parser.add_argument('--ema_rate', type=float, default=0.999, help='mu in EMA. 0 means using value from config')
    parser.add_argument('--ema_start_epoch', type=int, default=0, help='EMA start epoch')
    parser.add_argument("--resume_training", type=str2bool, default=False)
    parser.add_argument("--resume_ckpt", type=str, default="./exp/logs/doc/ckpt.pth")

    args = parser.parse_args()
    args.eta = 0  # backward-compatibility

    # parse config file
    with open(args.config, "r") as f: config = yaml.safe_load(f)
    new_config = dict2namespace(config)

    # setup logger
    logger = logging.getLogger()
    formatter = logging.Formatter("[%(asctime)s] %(message)s")
    handler1 = logging.StreamHandler(stream=sys.stdout)
    handler1.setFormatter(formatter)
    logger.addHandler(handler1)
    level = getattr(logging, 'INFO', None)
    logger.setLevel(level)

    # add device
    gpu_ids = args.gpu_ids
    logging.info(f"gpu_ids : {gpu_ids}")
    device = torch.device(f"cuda:{gpu_ids[0]}") if torch.cuda.is_available() and gpu_ids else torch.device("cpu")
    new_config.device = device
    args.device = device
    logging.info(f"data_dir: {args.data_dir}")
    logging.info(f"ts_range: {args.ts_range}")

    # set random seed
    seed = args.seed  # if seed is 0. then ignore it.
    logging.info(f"args.seed : {seed}")
    if seed:
        logging.info(f"  torch.manual_seed({seed})")
        logging.info(f"  np.random.seed({seed})")
        torch.manual_seed(seed)
        np.random.seed(seed)
    if seed and torch.cuda.is_available():
        logging.info(f"  torch.cuda.manual_seed_all({seed})")
        torch.cuda.manual_seed_all(seed)
    logging.info(f"final seed: torch.initial_seed(): {torch.initial_seed()}")

    cudnn.benchmark = True

    return args, new_config

def main():
    args, config = parse_args_and_config()
    logging.info(f"pid : {os.getpid()}")
    logging.info(f"cwd : {os.getcwd()}")
    logging.info(f"args: {args}")

    logging.info(f"main_rectified_flow::{args.todo} ===================================")
    try:
        if args.todo == 'sample':
            runner = DiffusionSamplingRectifiedFlow(args, config, device=config.device)
            runner.sample()
        elif args.todo == 'train':
            runner = DiffusionTrainingRectifiedFlow(args, config, device=config.device)
            runner.train()
        elif args.todo == 'loss_stat':
            runner = DiffusionTrainingRectifiedFlow(args, config, device=config.device)
            runner.loss_stat()
        else:
            raise Exception(f"Invalid todo: {args.todo}")
    except RuntimeError:
        logging.info(traceback.format_exc())

    return 0


if __name__ == "__main__":
    sys.exit(main())
