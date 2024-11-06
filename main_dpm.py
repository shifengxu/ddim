import argparse
import traceback
import logging
import yaml
import sys
import os
import torch
import numpy as np
import torch.backends.cudnn as cudnn

from runners.diffusion_dpm_solver import DiffusionDpmSolver
from utils import str2bool, dict2namespace

torch.set_printoptions(sci_mode=False)


def parse_args_and_config():
    parser = argparse.ArgumentParser(description=globals()["__doc__"])

    parser.add_argument("--config", type=str, default='./configs/cifar10.yml')
    parser.add_argument('--gpu_ids', nargs='+', type=int, default=[7])
    parser.add_argument("--seed", type=int, default=1234, help="Random seed. 0 means ignore")
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--todo", type=str, default='dpmSolver')
    parser.add_argument("--ema_flag", type=str2bool, default=True)
    parser.add_argument("--ts_type", type=str, default='discrete', help="discrete|continuous")
    parser.add_argument("--repeat_times", type=int, default=5, help='run XX times to get avg FID')
    parser.add_argument("--dpm_order", type=int, default=0, help='force DPM order to be XX. 0 means ignore.')
    parser.add_argument("--use_predefined_ts", type=str2bool, default=False)
    parser.add_argument("--steps_arr", nargs='+', type=int, default=[20])
    parser.add_argument("--order_arr", nargs='+', type=int, default=[1])
    parser.add_argument("--skip_type_arr", nargs='+', type=str, default=["time_uniform"])
    parser.add_argument("--fid_input1", type=str, default='cifar10-train')
    parser.add_argument("--ab_original_dir", type=str, default='phase1_ab_ori', help='original alpha_bar dir')
    parser.add_argument("--ab_scheduled_dir", type=str, default='phase2_ab_sch', help='scheduled alpha_bar dir')
    parser.add_argument("--ab_summary_dir", type=str, default='phase3_ab_sum', help='alpha_bar summary dir')
    parser.add_argument("--sample_count", type=int, default='50', help="sample image count")
    parser.add_argument("--sample_ckpt_path", type=str, default='./exp/ema-cifar10-model-790000.ckpt')
    parser.add_argument("--sample_batch_size", type=int, default=50, help="0 mean from config file")
    parser.add_argument("--sample_output_dir", type=str, default="./output7_vividvar/generated")
    # parser.add_argument("--predefined_aap_file", type=str, default="./output7_vividvar/res_aacum_0020.txt")
    # parser.add_argument("--predefined_aap_file", type=str, default="geometric_ratio:1.07")
    # parser.add_argument("--predefined_aap_file", type=str, default="all_scheduled_dir:./exp/dpm_alphaBar.scheduled")
    parser.add_argument("--predefined_aap_file", type=str, default="")
    parser.add_argument("--beta_schedule", type=str, default="linear")
    parser.add_argument("--noise_schedule", type=str, default="discrete", help="for NoiseScheduleV2")
    parser.add_argument('--ts_range', nargs='+', type=int, default=[], help='timestep range, such as [0, 200]')
    parser.add_argument("--eta", type=float, default=0.0)

    args = parser.parse_args()

    # parse config file
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    new_config = dict2namespace(config)

    # setup logger
    logger = logging.getLogger()
    formatter = logging.Formatter("[%(asctime)s] %(message)s")
    handler1 = logging.StreamHandler(stream=sys.stdout)
    handler1.setFormatter(formatter)
    logger.addHandler(handler1)
    level = getattr(logging, 'INFO', None)
    if not isinstance(level, int):
        raise ValueError("log level {INFO} not supported")
    logger.setLevel(level)

    # add device
    gpu_ids = args.gpu_ids
    logging.info(f"gpu_ids : {gpu_ids}")
    args.device = torch.device(f"cuda:{gpu_ids[0]}") if torch.cuda.is_available() and gpu_ids else torch.device("cpu")
    new_config.device = args.device

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
    logging.info(f"host: {os.uname().nodename}")
    logging.info(f"args: {args}")

    try:
        if args.todo == 'sample' or args.todo == 'dpmSolver':
            logging.info(f"{args.todo} ===================================")
            od, st, sk = args.order_arr[0], args.steps_arr[0], args.skip_type_arr[0]
            runner = DiffusionDpmSolver(args, config, order=od, steps=st, skip_type=sk, device=config.device)
            runner.sample()
        elif args.todo == 'dpmSolver.ratios':
            logging.info(f"{args.todo} ===================================")
            runner = DiffusionDpmSolver(args, config, device=config.device)
            runner.sample_ratios()
        elif args.todo == 'sample_all':
            logging.info(f"{args.todo} ===================================")
            runner = DiffusionDpmSolver(args, config, device=config.device)
            runner.sample_all(args.order_arr, args.steps_arr, args.skip_type_arr, times=args.repeat_times)
        elif args.todo == 'sample_all_scheduled':
            logging.info(f"{args.todo} ===================================")
            runner = DiffusionDpmSolver(args, config, device=config.device)
            runner.sample_all_scheduled()
        elif args.todo == 'sample_ori_sch':
            from runners.diffusion_dpm_solver_predefined_trajectory import DiffusionDpmSolverPredefinedTrajectory
            runner = DiffusionDpmSolverPredefinedTrajectory(args, config, device=config.device)
            runner.sample_original_and_scheduled()
        elif args.todo == 'alpha_bar_all':
            logging.info(f"{args.todo} ===================================")
            runner = DiffusionDpmSolver(args, config, device=config.device)
            runner.alpha_bar_all()
        else:
            raise Exception(f"Invalid todo: {args.todo}")
    except RuntimeError:
        logging.error(traceback.format_exc())

    return 0


if __name__ == "__main__":
    sys.exit(main())
