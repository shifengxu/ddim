import argparse
import logging
import random
import yaml
import sys
import os
import torch
import numpy as np
import torch.backends.cudnn as cudnn
import utils
from runners.diffusion_dpm_solver import DiffusionDpmSolver
from schedule.schedule_batch import ScheduleBatch
from utils import str2bool, dict2namespace

log_fn = utils.log_info

torch.set_printoptions(sci_mode=False)

class ScheduleSampleConfig:
    def __init__(self, lp=None, ts_int_flag=None, calo=None, improve_target=None):
        """
        :param lp: learning portion
        :param ts_int_flag:
        :param calo: calculating loss order. order when calculating loss during schedule
        :param improve_target: if FID reach the improve-target, then stop iteration
        """
        self.lp = lp
        self.ts_int_flag = ts_int_flag
        self.calo = calo
        self.improve_target = improve_target

    def parse(self, cfg_str):
        # cfg_str is like: "0.1   : False         : 0     : 20%"
        arr = cfg_str.strip().split(':')
        if len(arr) != 4: raise ValueError(f"Invalid cfg_str: {cfg_str}")
        self.lp             = float(arr[0].strip())
        self.ts_int_flag    = str2bool(arr[1].strip())
        self.calo           = int(arr[2].strip())
        self.improve_target = arr[3].strip()  # keep as string. Because it may be: 20% or 0.33 or empty
        return self

class ScheduleSampleResult:
    def __init__(self, ssc: ScheduleSampleConfig, key=None, fid=None, fid_base=0.):
        self.ssc = ssc
        self.key = key
        self.fid = fid
        self.notes = ''
        self.fid_base = fid_base  # base FID. new FID will compare with it.

def parse_args_and_config():
    parser = argparse.ArgumentParser(description=globals()["__doc__"])
    # ab: alpha_bar; ss: schedule & sample
    parser.add_argument("--todo", type=str, default='sample_scheduled')
    parser.add_argument("--config", type=str, default='./configs/celeba.yml')
    parser.add_argument('--gpu_ids', nargs='+', type=int, default=[3])
    parser.add_argument("--ts_type", type=str, default='continuous', help="discrete|continuous")
    parser.add_argument("--seed", type=int, default=1234, help="Random seed. 0 means ignore")
    parser.add_argument("--ss_plan_file", type=str, default="./output3_vubo_celeba_conti/vubo_ss_plan.txt")
    parser.add_argument("--fid_base_file", type=str, default="")
    parser.add_argument("--repeat_times", type=int, default=1, help='run XX times to get avg FID')
    parser.add_argument("--dpm_order", type=int, default=0, help='force DPM order to be XX. 0 means ignore.')
    parser.add_argument("--ts_int_flag", type=str2bool, default=False, help='timestep change to int type')
    parser.add_argument("--use_predefined_ts", type=str2bool, default=False)
    parser.add_argument("--steps_arr", nargs='+', type=int, default=[10])
    parser.add_argument("--order_arr", nargs='+', type=int, default=[1])
    parser.add_argument("--skip_type_arr", nargs='+', type=str, default=["logSNR"])
    parser.add_argument("--ab_original_dir", type=str, default='./output3_vubo_celeba_conti/phase1_ab_original')
    parser.add_argument("--ab_scheduled_dir", type=str, default='./output3_vubo_celeba_conti/phase2_ab_scheduled')
    parser.add_argument("--ab_summary_dir", type=str, default='./output3_vubo_celeba_conti/phase3_ab_summary')
    parser.add_argument("--sample_count", type=int, default='5', help="sample image count")
    parser.add_argument("--sample_ckpt_path", type=str, default="./exp/ema-celeba-ownpure-conti-E1000.ckpt")
    parser.add_argument("--sample_batch_size", type=int, default=5, help="0 mean from config file")
    parser.add_argument("--sample_output_dir", type=str, default="./output3_vubo_celeba_conti/generated")
    parser.add_argument("--fid_input1", type=str, default="./exp/datasets/celeba200K/50K")
    # parser.add_argument("--predefined_aap_file", type=str, default="./output7_vividvar/res_aacum_0020.txt")
    # parser.add_argument("--predefined_aap_file", type=str, default="geometric_ratio:1.07")
    # parser.add_argument("--predefined_aap_file", type=str, default="all_scheduled_dir:./exp/dpm_alphaBar.scheduled")
    parser.add_argument("--predefined_aap_file", type=str, default="")
    parser.add_argument("--resume_ckpt", type=str, default="./exp/logs/doc/ckpt.pth")
    parser.add_argument("--beta_schedule", type=str, default="cosine")
    parser.add_argument("--noise_schedule", type=str, default="cosine", help="for NoiseScheduleV2")
    parser.add_argument('--ts_range', nargs='+', type=int, default=[], help='timestep range, such as [0, 200]')
    parser.add_argument("--eta", type=float, default=0.0)

    # arguments for schedule_batch
    parser.add_argument("--n_epochs", type=int, default=1000)
    parser.add_argument("--log_interval", type=int, default=200)
    parser.add_argument('--lr', type=float, default=0.000001)
    parser.add_argument('--lp', type=float, default=0.01, help='learning_portion')
    parser.add_argument('--aa_low', type=float, default=0.0001, help="Alpha Accum lower bound")
    parser.add_argument("--aa_low_lambda", type=float, default=10000000)
    parser.add_argument("--weight_file", type=str, default='./output3_vubo_celeba_conti/res_eps_mse_arr.txt')
    parser.add_argument("--weight_power", type=float, default=1.0, help='change the weight value')
    args = parser.parse_args()
    args.output_dir    = args.ab_scheduled_dir  # only for class ScheduleBatch
    args.alpha_bar_dir = args.ab_original_dir   # only for class ScheduleBatch

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
        logging.info(f"  random.seed({seed})")
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
    if seed and torch.cuda.is_available():
        logging.info(f"  torch.cuda.manual_seed({seed})")
        logging.info(f"  torch.cuda.manual_seed_all({seed})")
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    logging.info(f"final seed: torch.initial_seed(): {torch.initial_seed()}")
    cudnn.benchmark = True
    return args, new_config

def schedule_and_sample(args, config):
    args = args
    sb = ScheduleBatch(args)
    ds = DiffusionDpmSolver(args, config, device=args.device)
    ab_dir = args.ab_original_dir
    log_fn(f"schedule_and_sample() *********************************")
    sum_dir = args.ab_summary_dir
    if not os.path.exists(sum_dir):
        logging.info(f"  os.makedirs({sum_dir})")
        os.makedirs(sum_dir)
    run_hist_file = os.path.join(sum_dir, "ss_run_hist.txt")
    fid_best_file = os.path.join(sum_dir, "ss_run_best.txt")
    plan_map = load_plans_from_file(args.ss_plan_file)
    fid_base_map = load_fid_from_file(args.fid_base_file)
    file_list = [f for f in os.listdir(ab_dir) if 'dpm_alphaBar' in f and f.endswith('.txt')]
    file_list = [os.path.join(ab_dir, f) for f in file_list]
    file_list = [f for f in file_list if os.path.isfile(f)]
    f_cnt = len(file_list)
    log_fn(f"  ab_original_dir  : {args.ab_original_dir}")
    log_fn(f"  ab_scheduled_dir : {args.ab_scheduled_dir}")
    log_fn(f"  ab file cnt      : {f_cnt}")
    log_fn(f"  sample_output_dir: {args.sample_output_dir}")
    log_fn(f"  fid_input1       : {args.fid_input1}")
    run_hist = []  # running history
    fid_best = []  # best FID for each key
    for idx, f_path in enumerate(sorted(file_list)):
        log_fn(f"{idx:03d}/{f_cnt}: {f_path} ----------------------------------")
        f_name = os.path.basename(f_path)  # dpm_alphaBar_1-010-time_quadratic.txt
        tmp = f_name.split('dpm_alphaBar_')[-1] # 1-010-time_quadratic.txt
        key = tmp.split('.')[0]                 # 1-010-time_quadratic
        ssc_arr = plan_map.get(key, plan_map['default'])
        ssr_best = None
        fid_base = fid_base_map.get(key, 0.)
        for ssc in ssc_arr:
            scheduled_file = sb.schedule_single(f_path, args.lr, ssc.lp, order=ssc.calo)
            args.predefined_aap_file = scheduled_file
            args.ts_int_flag = ssc.ts_int_flag
            fid_avg = ds.sample_times(args.repeat_times)
            ssr = ScheduleSampleResult(ssc, key, fid_avg, fid_base)
            rt_flag = reach_target(fid_base, fid_avg, ssc.improve_target)
            if rt_flag:
                ssr.notes = f"reach_target({fid_base:.5f}, {fid_avg:.5f}, {ssc.improve_target})"
                log_fn(ssr.notes)
            run_hist.append(ssr)
            output_ssr_list(run_hist, run_hist_file)
            if ssr_best is None or ssr_best.fid > fid_avg:
                ssr_best = ssr
            if rt_flag: break
            # if
        # for
        fid_best.append(ssr_best)
        output_ssr_list(fid_best, fid_best_file)
    # for

def sample_scheduled(args, config):
    args = args
    ds = DiffusionDpmSolver(args, config, device=args.device)
    log_fn(f"sample_scheduled() *********************************")
    sch_dir = args.ab_scheduled_dir
    sum_dir = args.ab_summary_dir
    log_fn(f"  sch_dir: {sch_dir}")
    log_fn(f"  sum_dir: {sum_dir}")
    if not os.path.exists(sum_dir):
        logging.info(f"  os.makedirs({sum_dir})")
        os.makedirs(sum_dir)
    run_hist_file = os.path.join(sum_dir, "ss_run_hist.txt")
    fid_best_file = os.path.join(sum_dir, "ss_run_best.txt")
    plan_map = load_plans_from_file(args.ss_plan_file)
    fid_base_map = load_fid_from_file(args.fid_base_file)
    file_list = [f for f in os.listdir(sch_dir) if 'dpm_alphaBar' in f and f.endswith('.txt')]
    file_list = [os.path.join(sch_dir, f) for f in file_list]
    file_list = [f for f in file_list if os.path.isfile(f)]
    f_cnt = len(file_list)
    log_fn(f"  ab file cnt      : {f_cnt}")
    log_fn(f"  fid_input1       : {args.fid_input1}")
    run_hist = []  # running history
    fid_best = []  # best FID for each key
    for idx, f_path in enumerate(sorted(file_list)):
        log_fn(f"{idx:03d}/{f_cnt}: {f_path} ----------------------------------")
        f_name = os.path.basename(f_path)  # dpm_alphaBar_1-010-time_quadratic.txt
        tmp = f_name.split('dpm_alphaBar_')[-1] # 1-010-time_quadratic.txt
        key = tmp.split('.')[0]                 # 1-010-time_quadratic
        ssc_arr = plan_map.get(key, plan_map['default'])
        ssr_best = None
        fid_base = fid_base_map.get(key, 0.)
        for ssc in ssc_arr:
            args.predefined_aap_file = f_path
            args.ts_int_flag = ssc.ts_int_flag
            fid_avg = ds.sample_times(args.repeat_times)
            ssr = ScheduleSampleResult(ssc, key, fid_avg, fid_base)
            rt_flag = reach_target(fid_base, fid_avg, ssc.improve_target)
            if rt_flag:
                ssr.notes = f"reach_target({fid_base:.5f}, {fid_avg:.5f}, {ssc.improve_target})"
                log_fn(ssr.notes)
            run_hist.append(ssr)
            output_ssr_list(run_hist, run_hist_file)
            if ssr_best is None or ssr_best.fid > fid_avg:
                ssr_best = ssr
            if rt_flag: break
            # if
        # for
        fid_best.append(ssr_best)
        output_ssr_list(fid_best, fid_best_file)
    # for

def output_ssr_list(ssr_list, f_path):
    log_fn(f"Save file: {f_path}")
    with open(f_path, 'w') as f_ptr:
        f_ptr.write(f"# FID_base=>   FID    : lp    :ts_int: calo: key                 : notes\n")
        for ssr in ssr_list:
            ssc = ssr.ssc
            f_ptr.write(f"{ssr.fid_base:9.5f} => {ssr.fid:9.5f}: {ssc.lp:.4f}: "
                        f"{'True ' if ssc.ts_int_flag else 'False'}: "
                        f"{ssc.calo:4d}: {ssr.key.ljust(20)}: {ssr.notes}\n")
        # for
    # with

def reach_target(fid_base, fid_new, improve_target):
    if improve_target is None or improve_target == '' or improve_target == 0:
        return False
    if fid_base is None or fid_base == '' or fid_base == 0:
        return False
    if fid_base <= fid_new:
        return False
    if '%' in improve_target: # 20%
        imp_flt = float(improve_target[:-1]) / 100.
        new_flt = (fid_base - fid_new) / fid_base
    else: # 0.2
        imp_flt = float(improve_target)
        new_flt = fid_base - fid_new
    return new_flt >= imp_flt

def load_plans_from_file(f_path):
    log_fn(f"load_plans_from_file(): {f_path}")
    with open(f_path, 'r') as f:
        lines = f.readlines()
    cnt_empty = 0
    cnt_comment = 0
    cnt_valid = 0
    plan_map = {}  # key is string, value is an array of ScheduleSampleConfig
    for line in lines:
        line = line.strip()
        if line == '':
            cnt_empty += 1
            continue
        if line.startswith('#'):
            cnt_comment += 1
            continue
        cnt_valid += 1
        key, cfg_str = line.strip().split(':', 1)
        key = key.strip()
        ssc = ScheduleSampleConfig().parse(cfg_str)
        if key in plan_map:
            plan_map[key].append(ssc)
        else:
            plan_map[key] = [ssc]
    log_fn(f"  cnt_empty  : {cnt_empty}")
    log_fn(f"  cnt_comment: {cnt_comment}")
    log_fn(f"  cnt_valid  : {cnt_valid}")
    log_fn(f"  cnt key    : {len(plan_map)}")
    for idx, key in enumerate(sorted(plan_map)):
        log_fn(f"  {idx:03d} {key}: {len(plan_map[key])}")
    log_fn(f"load_plans_from_file(): {f_path}... Done")
    if 'default' not in plan_map:
        raise ValueError(f"'default' must be in plan file: {f_path}")
    return plan_map

def load_fid_from_file(f_path):
    log_fn(f"load_fid_from_file(): {f_path}")
    if f_path is None or f_path == '':
        log_fn(f"  !!! fid_file empty. Will use 0.0 by default.")
        return {}
    with open(f_path, 'r') as f:
        lines = f.readlines()
    cnt_empty = 0
    cnt_comment = 0
    cnt_valid = 0
    fid_map = {}  # key is string, value is fid of float type
    for line in lines:
        line = line.strip()
        if line == '':
            cnt_empty += 1
            continue
        if line.startswith('#'):
            cnt_comment += 1
            continue
        cnt_valid += 1
        key, val = line.strip().split(':', 1)
        key, val = key.strip(), val.strip()
        fid_map[key] = float(val)
    log_fn(f"  cnt_empty  : {cnt_empty}")
    log_fn(f"  cnt_comment: {cnt_comment}")
    log_fn(f"  cnt_valid  : {cnt_valid}")
    log_fn(f"  cnt key    : {len(fid_map)}")
    log_fn(f"load_fid_from_file(): {f_path}... Done")
    return fid_map

def main():
    args, config = parse_args_and_config()
    logging.info(f"pid : {os.getpid()}")
    logging.info(f"cwd : {os.getcwd()}")
    logging.info(f"args: {args}")

    arr = args.todo.split(',')
    for a in arr:
        if a == 'alpha_bar_all':
            logging.info(f"alpha_bar_all() ===================================")
            runner = DiffusionDpmSolver(args, config, device=config.device)
            runner.alpha_bar_all()
        elif a == 'schedule_sample':
            logging.info(f"schedule_and_sample() ===================================")
            schedule_and_sample(args, config)
        elif a == 'schedule':
            logging.info(f"schedule_only() ===================================")
            sb = ScheduleBatch(args)
            sb.schedule_batch()
        elif a == 'sample_scheduled':
            logging.info(f"{a} ===================================")
            sample_scheduled(args, config)
        elif a == 'sample_baseline':
            logging.info(f"{a} ===================================")
            od, st, sk = args.order_arr[0], args.steps_arr[0], args.skip_type_arr[0]
            runner = DiffusionDpmSolver(args, config, order=od, steps=st, skip_type=sk, device=config.device)
            args.predefined_aap_file = ''
            runner.sample_times(1)
        else:
            raise Exception(f"Invalid todo: {args.todo}")
    # for
    return 0

if __name__ == "__main__":
    sys.exit(main())
