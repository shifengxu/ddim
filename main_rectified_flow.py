import argparse
import traceback

import logging
import yaml
import sys
import os
import torch
import numpy as np
import torch.backends.cudnn as cudnn

from main_fid import calc_fid
from runners.diffusion_sampling_rf import DiffusionSamplingRectifiedFlow
from runners.diffusion_training_rf import DiffusionTrainingRectifiedFlow
from runners.kl_divergence import KullbackLeiblerDivergence
from schedule_rf.linear_interpreter import LinearInterpreter
from schedule_rf.scheduler_rf import ScheduleRectifiedFlow
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
    parser.add_argument("--predefined_ts_file", type=str, default="")
    # parser.add_argument("--predefined_ts_file", type=str, default="./output7_rectified_flow/predefined_ts_file.txt")
    parser.add_argument("--sample_order", type=int, default=1, help="1|2|3")
    parser.add_argument("--sample_order_arr", nargs='*', type=int, default=[], help="1|2|3")
    parser.add_argument("--sample_steps_arr", nargs='*', type=int, default=[])
    parser.add_argument("--sample_geometric_arr", nargs='*', type=float, default=[0.9])
    parser.add_argument("--sample_init_ts_arr", nargs='*', type=int, default=[940])

    # training
    parser.add_argument('--ema_flag', type=str2bool, default=True, help='EMA flag')
    parser.add_argument('--ema_rate', type=float, default=0.999, help='mu in EMA. 0 means using value from config')
    parser.add_argument('--ema_start_epoch', type=int, default=0, help='EMA start epoch')
    parser.add_argument("--resume_training", type=str2bool, default=False)
    parser.add_argument("--resume_ckpt", type=str, default="./exp/logs/doc/ckpt.pth")
    parser.add_argument("--loss_dual", type=str2bool, default=False, help="use dual loss")
    parser.add_argument("--loss_lambda", type=float, default=0.1, help="lambda when dual loss")

    # scheduler
    parser.add_argument('--sch_lr', type=float, default=0.000001)
    parser.add_argument('--sch_epochs', type=int, default=1000)
    parser.add_argument('--sch_max_ts_arr', nargs='+', type=float, default=[0.94], help='max timestep')
    parser.add_argument('--sch_loss_decay_arr', nargs='+', type=float, default=[0.4])
    parser.add_argument('--sch_delta_expo_arr', nargs='+', type=float, default=[0.5])
    parser.add_argument('--sch_steps', type=int, default=10, help='steps count for sampling')
    parser.add_argument('--sch_log_interval', type=int, default=50)
    parser.add_argument('--sch_mse_file', type=str, default='./output6_schedule_rf/mse_list_500.txt')
    parser.add_argument('--sch_output_file', type=str, default='./output6_schedule_rf/details_saved.txt')

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

def sample_all(args, config):
    steps_arr     = args.sample_steps_arr
    init_ts_arr   = args.sample_init_ts_arr
    geometric_arr = args.sample_geometric_arr or np.linspace(0.80, 1.10, 31)
    order_arr     = args.sample_order_arr or [1, 2, 3]
    basename = os.path.basename(args.sample_ckpt_path)
    stem, ext = os.path.splitext(basename)
    result_file = f"./sample_all_rf_{stem}.txt"
    logging.info(f"main_rectified_flow::sample_all()")
    logging.info(f"steps_arr    : {steps_arr}")
    logging.info(f"init_ts_arr  : {init_ts_arr}")
    logging.info(f"geometric_arr: {geometric_arr}")
    logging.info(f"order_arr    : {order_arr}")
    logging.info(f"result_file  : {result_file}")
    res_arr = []
    args.predefined_ts_file = None
    for steps in steps_arr:
        for init_ts in init_ts_arr:
            for geo in geometric_arr:
                for order in order_arr:
                    args.sample_order = order
                    runner = DiffusionSamplingRectifiedFlow(args, config, device=config.device)
                    runner.sample(ts_geometric=geo, steps=steps, init_ts=init_ts)
                    del runner  # to save GPU memory
                    fid = calc_fid(args.gpu_ids[0], True)
                    msg = f"FID: {fid:7.3f}. steps:{steps}, init_ts:{init_ts},  geo:{geo}, order:{order}"
                    res_arr.append(msg)
                    with open(result_file, 'w') as fptr: [fptr.write(f"{m}\n") for m in res_arr]
                    logging.info(msg)
                    logging.info("")
                    logging.info("")
                    logging.info("")
                # for
            # for
        # for
    # for

    # ts_file_arr = ['', './predefined_ts_file.txt']
    # for tf in ts_file_arr:
    #     args.predefined_ts_file = tf
    #     for order in order_arr:
    #         args.sample_order = order
    #         runner = DiffusionSamplingRectifiedFlow(args, config, device=config.device)
    #         runner.sample()
    #         fid = calc_fid(args.gpu_ids[0], True)
    #         msg = f"FFFFF. order:{order}, FID: {fid:7.4f}. predefined_ts_file: {tf}"
    #         res_arr.append(msg)
    #         with open(result_file, 'w') as fptr: [fptr.write(f"{m}\n") for m in res_arr]
    #         logging.info(msg)
    #         logging.info("")
    #         logging.info("")
    #         logging.info("")
    #     # for
    # # for
    [logging.info(f"{msg}") for msg in res_arr]

def schedule_sample(args, config):
    result_file = "./schedule_sample_result_rf.txt"
    order_arr = args.sample_order_arr or [1]
    logging.info(f"main_rectified_flow::schedule_sample()")
    logging.info(f"sch_max_ts_arr    : {args.sch_max_ts_arr}")
    logging.info(f"sch_loss_decay_arr: {args.sch_loss_decay_arr}")
    logging.info(f"sch_delta_expo_arr: {args.sch_delta_expo_arr}")
    logging.info(f"order_arr         : {order_arr}")
    res_arr = []
    t_zero = torch.zeros((1,), device=args.device)
    lip = LinearInterpreter(y_arr_file=args.sch_mse_file)
    for max_ts in args.sch_max_ts_arr:
        for loss_decay in args.sch_loss_decay_arr:
            for delta_expo in args.sch_delta_expo_arr:
                scheduler = ScheduleRectifiedFlow(weight_simulator=lip, max_ts=max_ts, loss_decay=loss_decay,
                                                  delta_expo=delta_expo, steps=args.sch_steps,
                                                  lr=args.sch_lr, n_epochs=args.sch_epochs,
                                                  log_interval=args.sch_log_interval,
                                                  output_file=args.sch_output_file, device=args.device)
                ts_arr = scheduler.train()
                ts_arr *= 1000  # change range from [0, 1] to [0, 1000]
                ts_arr = torch.concat([ts_arr, t_zero], dim=0)
                for order in order_arr:
                    args.sample_order = order
                    runner = DiffusionSamplingRectifiedFlow(args, config, device=config.device)
                    runner.sample(ts_arr)
                    fid = calc_fid(args.gpu_ids[0], True)
                    msg = f"FID:{fid:7.4f}: max_ts:{max_ts:.4f}, loss_decay:{loss_decay:.4f}, " \
                          f"delta_expo:{delta_expo:.4f}, order:{order}"
                    res_arr.append(msg)
                    with open(result_file, 'w') as fptr: [fptr.write(f"{m}\n") for m in res_arr]
                    logging.info(f"Saved: {result_file}")
                    logging.info(msg)
                    logging.info("")
                    logging.info("")
                    logging.info("")
                # for
            # for
        # for
    # for
    [logging.info(f"{msg}") for msg in res_arr]

def kl_div(args, config):
    # runner = KullbackLeiblerDivergence(args, config)

    args.sample_ckpt_path = "./output7_rectified_flow/ckpt_rf_E500_dual_loss0.1.pth"
    runner = KullbackLeiblerDivergence(args, config)
    # runner.calc_grad_similarity()
    # runner.save_image_gt_x0()
    # runner.save_image_gt_xt(ts2)
    # runner.predict_save_image(0.6, 0.4)
    # runner.predict_seq_save_image([0.6, 0.5, 0.4])
    dir1 = "./output4_kl_div/img_gt_x0.600"
    dir2 = "./output4_kl_div/img_pr_x0.800-0.600"
    runner.calc_kl_div(dir1, dir2)

def main():
    args, config = parse_args_and_config()
    logging.info(f"pid : {os.getpid()}")
    logging.info(f"cwd : {os.getcwd()}")
    logging.info(f"args: {args}")

    logging.info(f"main_rectified_flow -> {args.todo} ===================================")
    try:
        if args.todo == 'sample':
            runner = DiffusionSamplingRectifiedFlow(args, config, device=config.device)
            runner.sample()
        elif args.todo == 'sample_all':
            sample_all(args, config)
        elif args.todo == 'schedule_sample':
            schedule_sample(args, config)
        elif args.todo == 'train':
            runner = DiffusionTrainingRectifiedFlow(args, config, device=config.device)
            runner.train()
        elif args.todo == 'loss_stat':
            runner = DiffusionTrainingRectifiedFlow(args, config, device=config.device)
            runner.loss_stat()
        elif args.todo == 'kl_div':
            kl_div(args, config)
        else:
            raise Exception(f"Invalid todo: {args.todo}")
    except RuntimeError:
        logging.info(traceback.format_exc())

    return 0


if __name__ == "__main__":
    sys.exit(main())
