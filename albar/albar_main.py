import argparse
import yaml
import sys
import os
import torch
import numpy as np
import torch.backends.cudnn as cudnn

# add current dir and parent dir into python-path.
# this it to facilitate the Linux start.exe file.
cur_dir = os.path.dirname(__file__)
prt_dir = os.path.dirname(cur_dir)  # parent dir
if cur_dir not in sys.path:
    sys.path.append(cur_dir)
if prt_dir not in sys.path:
    sys.path.append(prt_dir)

from albar_sampling import AlbarSampling
from albar_training import AlbarTraining
from utils import str2bool, dict2namespace, log_info

torch.set_printoptions(sci_mode=False)
log_fn = log_info

def parse_args_and_config():
    parser = argparse.ArgumentParser(description=globals()["__doc__"])
    parser.add_argument("--config", type=str, default='./configs/cifar10.yml')
    parser.add_argument('--gpu_ids', nargs='+', type=int, default=[7])
    parser.add_argument("--albar_range", type=int, default=1)
    parser.add_argument("--data_dir", type=str, default="./exp")
    parser.add_argument("--result_dir", type=str, default="./result")
    parser.add_argument("--test_data_dir", type=str, default="../vq-vae-2-python/image_dataset/FFHQ32x32_test")
    parser.add_argument("--test_per_epoch", type=int, default=10, help='calc loss on test dataset. 0 means no calc.')
    parser.add_argument('--lr', type=float, default=0., help="learning rate")
    parser.add_argument("--seed", type=int, default=1234, help="Random seed. 0 means ignore")
    parser.add_argument("--n_epochs", type=int, default=0, help="0 mean epoch number from config file")
    parser.add_argument("--batch_size", type=int, default=250, help="0 mean to use size from config file")
    parser.add_argument('--ema_flag', type=str2bool, default=True, help='EMA flag')
    parser.add_argument('--ema_rate', type=float, default=0.99, help='mu in EMA. 0 means using value from config')
    parser.add_argument('--ema_start_epoch', type=int, default=50, help='EMA start epoch')
    parser.add_argument("--todo", type=str, default='train', help="train|sample")
    parser.add_argument("--sample_count", type=int, default='50000', help="sample image count")
    parser.add_argument("--sample_img_init_id", type=int, default='0', help="sample image init ID")
    parser.add_argument("--sample_ckpt_path", type=str, default='')
    parser.add_argument("--sample_batch_size", type=int, default='500', help="0 mean from config file")
    parser.add_argument("--sample_output_dir", type=str, default="exp/image_sampled")
    parser.add_argument("--fid", action="store_true", default=True)
    parser.add_argument("--resume_training", type=str2bool, default=False)
    parser.add_argument("--resume_ckpt", type=str, default="./exp/logs/doc/ckpt.pth")
    parser.add_argument("--use_pretrained", action="store_true")
    parser.add_argument("--sample_type", type=str, default="generalized", help="generalized or ddpm_noisy")
    parser.add_argument("--skip_type", type=str, default="uniform", help="uniform or quadratic")
    parser.add_argument("--timesteps", type=int, default=1000, help="number of steps involved")
    parser.add_argument("--beta_schedule", type=str, default="cosine")
    parser.add_argument("--eta", type=float, default=0.0)
    parser.add_argument('--ts_range', nargs='+', type=int, default=[])
    args = parser.parse_args()

    # parse config file
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    new_config = dict2namespace(config)

    # add device
    gpu_ids = args.gpu_ids
    log_fn(f"gpu_ids : {gpu_ids}")
    device = torch.device(f"cuda:{gpu_ids[0]}") if torch.cuda.is_available() and gpu_ids else torch.device("cpu")
    new_config.device = device
    args.device = device
    log_fn(f"result_dir: {args.result_dir}")
    if not os.path.exists(args.result_dir):
        log_fn(f"os.makedirs({args.result_dir})")
        os.makedirs(args.result_dir)

    # set random seed
    seed = args.seed  # if seed is 0. then ignore it.
    log_fn(f"args.seed : {seed}")
    if seed:
        log_fn(f"  torch.manual_seed({seed})")
        log_fn(f"  np.random.seed({seed})")
        torch.manual_seed(seed)
        np.random.seed(seed)
    if seed and torch.cuda.is_available():
        log_fn(f"  torch.cuda.manual_seed_all({seed})")
        torch.cuda.manual_seed_all(seed)
    log_fn(f"final seed: torch.seed(): {torch.seed()}")

    cudnn.benchmark = True

    return args, new_config

def main():
    args, config = parse_args_and_config()
    log_fn(f"pid : {os.getpid()}")
    log_fn(f"cwd : {os.getcwd()}")
    log_fn(f"args: {args}")

    if args.todo == 'sample':
        log_fn(f"main_albar::sample ===================================")
        runner = AlbarSampling(args, config, device=config.device)
        runner.sample()
    elif args.todo == 'train':
        log_fn(f"main_albar::train ===================================")
        runner = AlbarTraining(args, config, device=config.device)
        runner.train()
    else:
        raise Exception(f"Invalid todo: {args.todo}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
