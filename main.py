import argparse
import traceback
import shutil
import logging
import yaml
import sys
import os
import torch
import numpy as np
import torch.utils.tensorboard as tb

from runners.diffusion import Diffusion

torch.set_printoptions(sci_mode=False)


def parse_args_and_config():
    parser = argparse.ArgumentParser(description=globals()["__doc__"])

    parser.add_argument("--config", type=str, default='cifar10.yml', help="Path to the config file")
    parser.add_argument('--gpu_ids', nargs='+', type=int, default=[0, 1, 2])
    parser.add_argument("--seed", type=int, default=1234, help="Random seed")
    parser.add_argument("--exp", type=str, default="exp", help="Path for saving running related data.")
    parser.add_argument("--doc", type=str, default='doc',
                        help="A string for documentation purpose. Will be the name of the log folder.")
    parser.add_argument("--data_dir", type=str, default="", help="dir of training/testing data.")
    parser.add_argument('--ts_range', nargs='+', type=int, default=[], help='timestep range, such as [0, 200]')
    parser.add_argument("--comment", type=str, default="", help="A string for experiment comment")
    parser.add_argument("--verbose", type=str, default="info",
                        help="Verbose level: info | debug | warning | critical")
    parser.add_argument("--test", action="store_true", help="Whether to test the model")
    parser.add_argument("--sample", action="store_true", default=False,
                        help="Whether to produce samples from the model")
    parser.add_argument("--fid", action="store_true", default=True)
    parser.add_argument("--interpolation", action="store_true")
    parser.add_argument("--resume_training", action="store_true", help="Whether to resume training")
    parser.add_argument("-i", "--image_folder", type=str, default="images",
                        help="The folder name of samples")
    parser.add_argument("--ni", action="store_true", default=True,
                        help="No interaction. Suitable for Slurm Job launcher")
    parser.add_argument("--use_pretrained", action="store_true")
    parser.add_argument("--sample_type", type=str, default="generalized",
                        help="sampling approach (generalized or ddpm_noisy)")
    parser.add_argument("--skip_type", type=str, default="uniform",
                        help="skip according to (uniform or quadratic)")
    parser.add_argument("--timesteps", type=int, default=1000, help="number of steps involved")
    parser.add_argument("--eta", type=float, default=0.0,
                        help="eta used to control the variances of sigma")
    parser.add_argument("--sequence", action="store_true")

    args = parser.parse_args()
    args.log_path = os.path.join(args.exp, "logs", args.doc)
    if not os.path.exists(args.log_path):
        os.makedirs(args.log_path)

    # parse config file
    with open(os.path.join("configs", args.config), "r") as f:
        config = yaml.safe_load(f)

    new_config = dict2namespace(config)
    tb_path = os.path.join(args.exp, "tensorboard", args.doc)

    # setup logger
    logger = logging.getLogger()
    formatter = logging.Formatter("%(levelname)s - %(filename)s - %(asctime)s - %(message)s")
    handler1 = logging.StreamHandler(stream=sys.stdout)
    handler1.setFormatter(formatter)
    logger.addHandler(handler1)
    level = getattr(logging, args.verbose.upper(), None)
    if not isinstance(level, int):
        raise ValueError("log level {} not supported".format(args.verbose))
    logger.setLevel(level)
    if not args.test and not args.sample:
        handler2 = logging.FileHandler(os.path.join(args.log_path, "stdout.txt"))
        handler2.setFormatter(formatter)
        logger.addHandler(handler2)
        if not args.resume_training:
            renew_log_dir(args, tb_path, new_config)

        new_config.tb_logger = tb.SummaryWriter(log_dir=tb_path)
    elif args.sample:
        os.makedirs(os.path.join(args.exp, "image_samples"), exist_ok=True)
        args.image_folder = os.path.join(args.exp, "image_samples", args.image_folder)
        if not os.path.exists(args.image_folder):
            os.makedirs(args.image_folder)
        else:
            if not (args.fid or args.interpolation):
                overwrite = False
                if args.ni:
                    overwrite = True
                else:
                    response = input(
                        f"Image folder {args.image_folder} already exists. Overwrite? (Y/N)"
                    )
                    if response.upper() == "Y":
                        overwrite = True

                if overwrite:
                    shutil.rmtree(args.image_folder)
                    os.makedirs(args.image_folder)
                else:
                    print("Output image folder exists. Program halted.")
                    sys.exit(0)

    # add device
    gpu_ids = args.gpu_ids
    logging.info(f"gpu_ids : {gpu_ids}")
    device = torch.device(f"cuda:{gpu_ids[0]}") if torch.cuda.is_available() else torch.device("cpu")
    new_config.device = device
    if not args.data_dir:
        args.data_dir = args.exp
    logging.info(f"data_dir: {args.data_dir}")
    logging.info(f"ts_range: {args.ts_range}")

    # set random seed
    logging.info(f"seed: {args.seed}")
    logging.info(f"  torch.manual_seed({args.seed})")
    logging.info(f"  np.random.seed({args.seed})")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        logging.info(f"  torch.cuda.manual_seed_all({args.seed})")
        torch.cuda.manual_seed_all(args.seed)

    torch.backends.cudnn.benchmark = True

    return args, new_config


def renew_log_dir(args, tb_path, new_config):
    if os.path.exists(args.log_path):
        overwrite = False
        if args.ni:
            overwrite = True
        else:
            response = input("Folder already exists. Overwrite? (Y/N)")
            if response.upper() == "Y":
                overwrite = True

        if overwrite:
            if os.path.exists(args.log_path):
                logging.info(f"remove: {args.log_path}")
                shutil.rmtree(args.log_path)
            if os.path.exists(tb_path):
                logging.info(f"remove: {tb_path}")
                shutil.rmtree(tb_path)
            logging.info(f"mkdir : {args.log_path}")
            os.makedirs(args.log_path)
        else:
            print("Folder exists. Program halted.")
            sys.exit(0)
    else:
        os.makedirs(args.log_path)

    with open(os.path.join(args.log_path, "config.yml"), "w") as f:
        yaml.dump(new_config, f, default_flow_style=False)


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


def main():
    args, config = parse_args_and_config()
    logging.info("Writing log file to {}".format(args.log_path))
    logging.info(f"pid : {os.getpid()}")
    logging.info(f"cwd : {os.getcwd()}")
    logging.info(f"args: {args}")

    try:
        runner = Diffusion(args, config, device=config.device)
        if args.sample:
            logging.info(f"sample ===================================")
            runner.sample()
        elif args.test:
            logging.info(f"test ===================================")
            runner.test()
        else:
            logging.info(f"train ===================================")
            runner.train()
    except Exception:
        logging.error(traceback.format_exc())

    return 0


if __name__ == "__main__":
    sys.exit(main())
