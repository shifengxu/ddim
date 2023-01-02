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

from runners.diffusion_lostats import DiffusionLostats
from runners.diffusion_sampling import DiffusionSampling
from runners.diffusion_sampling0 import DiffusionSampling0
from runners.diffusion_sampling2 import DiffusionSampling2
from runners.diffusion_samplingPrev import DiffusionSamplingPrev
from runners.diffusion_samplingByPhase import DiffusionSamplingByPhase
from runners.diffusion_training import DiffusionTraining
from runners.diffusion_testing import DiffusionTesting
from runners.diffusion_partial_sampling import DiffusionPartialSampling
from runners.diffusion_latent_sampling import DiffusionLatentSampling
from runners.diffusion_training0 import DiffusionTraining0
from runners.diffusion_training2 import DiffusionTraining2
from runners.diffusion_trainingPrev import DiffusionTrainingPrev

from utils import str2bool, dict2namespace

torch.set_printoptions(sci_mode=False)


def parse_args_and_config():
    parser = argparse.ArgumentParser(description=globals()["__doc__"])

    parser.add_argument("--config", type=str, default='./configs/cifar10.yml')
    parser.add_argument('--gpu_ids', nargs='+', type=int, default=[7])
    parser.add_argument("--data_dir", type=str, default="./exp")
    parser.add_argument("--test_data_dir", type=str, default="../vq-vae-2-python/image_dataset/FFHQ32x32_test")
    parser.add_argument("--test_per_epoch", type=int, default=10, help='calc loss on test dataset. 0 means no calc.')
    parser.add_argument('--lr', type=float, default=0., help="learning rate")
    parser.add_argument("--seed", type=int, default=1234, help="Random seed. 0 means ignore")
    parser.add_argument("--n_epochs", type=int, default=0, help="0 mean epoch number from config file")
    parser.add_argument("--batch_size", type=int, default=250, help="0 mean to use size from config file")
    parser.add_argument("--exp", type=str, default="exp", help="Path for saving running related data.")
    parser.add_argument("--doc", type=str, default='doc',
                        help="A string for documentation purpose. Will be the name of the log folder.")
    parser.add_argument('--ts_range', nargs='+', type=int, default=[], help='timestep range, such as [0, 200]')
    parser.add_argument('--ema_flag', type=str2bool, default=True, help='EMA flag')
    parser.add_argument('--ema_rate', type=float, default=0.99, help='mu in EMA. 0 means using value from config')
    parser.add_argument('--ema_start_epoch', type=int, default=50, help='EMA start epoch')
    parser.add_argument("--model_in_channels", type=int, default='0', help='model.in_channels')
    parser.add_argument("--data_resolution", type=int, default='0', help='data.resolution')
    parser.add_argument("--comment", type=str, default="", help="A string for experiment comment")
    parser.add_argument("--verbose", type=str, default="info",
                        help="Verbose level: info | debug | warning | critical")
    parser.add_argument("--todo", type=str, default='lsample', help="train|sample|psample|lsample")
    parser.add_argument("--sample_count", type=int, default='50000', help="sample image count")
    parser.add_argument("--sample_img_init_id", type=int, default='0', help="sample image init ID")
    parser.add_argument("--sample_ckpt_path", type=str, default='./exp/ema-cifar10-model-790000.ckpt')
    parser.add_argument("--sample_ckpt_path_x0", type=str, default='./output3_sampleByPhase/ckpt_E1000_x0.pth')
    parser.add_argument("--sample_x0_ts_cnt", type=int, default=500)
    parser.add_argument("--sample_ckpt_dir", type=str, default='./exp/model_S4E1000TSxxx')
    parser.add_argument("--sample_batch_size", type=int, default='500', help="0 mean from config file")
    parser.add_argument("--sample_output_dir", type=str, default="exp/image_sampled")
    parser.add_argument('--psample_ts_list', nargs='+', type=int, help='0 means x0',
                        default=[50, 150, 250, 350, 450, 550, 650, 750, 850, 950, 0, 1000])
    parser.add_argument('--psample_dir', type=str, default="./exp/partialSample")
    parser.add_argument('--psample_type', type=str, default='from_x0', help='from_x0|from_gn')
    parser.add_argument("--fid", action="store_true", default=True)
    parser.add_argument("--interpolation", action="store_true")
    parser.add_argument("--resume_training", type=str2bool, default=False)
    parser.add_argument("--resume_ckpt", type=str, default="./exp/logs/doc/ckpt.pth")
    parser.add_argument("--ni", action="store_true", default=True,
                        help="No interaction. Suitable for Slurm Job launcher")
    parser.add_argument("--use_pretrained", action="store_true")
    parser.add_argument("--sample_type", type=str, default="generalized",
                        help="sampling approach (generalized or ddpm_noisy)")
    parser.add_argument("--skip_type", type=str, default="uniform",
                        help="skip according to (uniform or quadratic)")
    parser.add_argument("--timesteps", type=int, default=1000, help="number of steps involved")
    parser.add_argument("--beta_schedule", type=str, default="cosine")
    parser.add_argument("--eta", type=float, default=0.0,
                        help="eta used to control the variances of sigma")
    parser.add_argument("--sequence", action="store_true")

    args = parser.parse_args()
    args.log_path = os.path.join(args.exp, "logs", args.doc)
    if not os.path.exists(args.log_path):
        os.makedirs(args.log_path)

    # parse config file
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    new_config = dict2namespace(config)
    tb_path = os.path.join(args.exp, "tensorboard", args.doc)

    # setup logger
    logger = logging.getLogger()
    # formatter = logging.Formatter("%(levelname)s - %(filename)s - %(asctime)s - %(message)s")
    formatter = logging.Formatter("%(asctime)s - %(message)s")
    handler1 = logging.StreamHandler(stream=sys.stdout)
    handler1.setFormatter(formatter)
    logger.addHandler(handler1)
    level = getattr(logging, args.verbose.upper(), None)
    if not isinstance(level, int):
        raise ValueError("log level {} not supported".format(args.verbose))
    logger.setLevel(level)
    if 'test' not in args.todo and 'sample' not in args.todo:
        handler2 = logging.FileHandler(os.path.join(args.log_path, "stdout.txt"))
        handler2.setFormatter(formatter)
        logger.addHandler(handler2)
        if not args.resume_training:
            renew_log_dir(args, tb_path, new_config)

        new_config.tb_logger = tb.SummaryWriter(log_dir=tb_path)
    elif args.todo == 'sample':
        # os.makedirs(os.path.join(args.exp, "image_samples"), exist_ok=True)
        if not os.path.exists(args.sample_output_dir):
            print(f"mkdir: {args.sample_output_dir}")
            os.makedirs(args.sample_output_dir)
        else:
            if not (args.fid or args.interpolation):
                overwrite = False
                if args.ni:
                    overwrite = True
                else:
                    response = input(f"Image folder {args.sample_output_dir} already exists. Overwrite? (Y/N)")
                    if response.upper() == "Y":
                        overwrite = True

                if overwrite:
                    print(f"rmtree: {args.sample_output_dir}")
                    shutil.rmtree(args.sample_output_dir)
                    print(f"mkdir : {args.sample_output_dir}")
                    os.makedirs(args.sample_output_dir)
                else:
                    print("Output image folder exists. Program halted.")
                    sys.exit(0)

    # add device
    gpu_ids = args.gpu_ids
    logging.info(f"gpu_ids : {gpu_ids}")
    device = torch.device(f"cuda:{gpu_ids[0]}") if torch.cuda.is_available() and gpu_ids else torch.device("cpu")
    new_config.device = device
    if not args.data_dir:
        args.data_dir = args.exp
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
    logging.info(f"final seed: torch.seed(): {torch.seed()}")

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


def main():
    args, config = parse_args_and_config()
    logging.info("Writing log file to {}".format(args.log_path))
    logging.info(f"pid : {os.getpid()}")
    logging.info(f"cwd : {os.getcwd()}")
    logging.info(f"args: {args}")

    try:
        if args.todo == 'psample':
            logging.info(f"partial sample ===========================")
            runner = DiffusionPartialSampling(args, config, device=config.device)
            runner.run()
        elif args.todo == 'lsample':
            logging.info(f"latent sample ===========================")
            runner = DiffusionLatentSampling(args, config, device=config.device)
            runner.sample()
        elif args.todo == 'sample':
            logging.info(f"sample ===================================")
            runner = DiffusionSampling(args, config, device=config.device)
            runner.sample()
        elif args.todo == 'test':
            logging.info(f"test ===================================")
            runner = DiffusionTesting(args, config, device=config.device)
            runner.test()
        elif args.todo == 'train':
            logging.info(f"train ===================================")
            runner = DiffusionTraining(args, config, device=config.device)
            runner.train()
        elif args.todo == 'train2':
            logging.info(f"train2 ===================================")
            runner = DiffusionTraining2(args, config, device=config.device)
            runner.train()
        elif args.todo == 'sample2':
            logging.info(f"sample2 ===================================")
            runner = DiffusionSampling2(args, config, device=config.device)
            runner.sample()
        elif args.todo == 'train0':
            logging.info(f"train0 ===================================")
            runner = DiffusionTraining0(args, config, device=config.device)
            runner.train()
        elif args.todo == 'sample0':
            logging.info(f"sample0 ===================================")
            runner = DiffusionSampling0(args, config, device=config.device)
            runner.sample()
        elif args.todo == 'trainPrev':
            logging.info(f"trainPrev ===================================")
            runner = DiffusionTrainingPrev(args, config, device=config.device)
            runner.train()
        elif args.todo == 'samplePrev':
            logging.info(f"samplePrev ===================================")
            runner = DiffusionSamplingPrev(args, config, device=config.device)
            runner.sample()
        elif args.todo.startswith('lostats'):
            logging.info(f"{args.todo} ===================================")
            runner = DiffusionLostats(args, config, device=config.device)
            runner.run()
        elif args.todo == 'sampleByPhase':
            logging.info(f"{args.todo} ===================================")
            runner = DiffusionSamplingByPhase(args, config, device=config.device)
            runner.sample()
        else:
            raise Exception(f"Invalid todo: {args.todo}")
    except Exception:
        logging.error(traceback.format_exc())

    return 0


if __name__ == "__main__":
    sys.exit(main())
