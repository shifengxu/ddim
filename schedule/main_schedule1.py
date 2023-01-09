"""
Find beta schedule.
For each timestep t, just make their variance equal to each other by
minimizing their variance.
"""
import argparse
import sys
import os
import time
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import optim
from torch.backends import cudnn
from torch.nn import DataParallel

import utils
from base import ScheduleBase, Schedule1Model

log_fn = utils.log_info

torch.set_printoptions(sci_mode=False)

def parse_args():
    parser = argparse.ArgumentParser(description=globals()["__doc__"])
    parser.add_argument('--gpu_ids', type=int, nargs='+', default=[7])
    parser.add_argument("--n_epochs", type=int, default=500000)
    parser.add_argument('--lr', type=float, default=0.2)
    parser.add_argument('--output_dir', type=str, default='./output7_schedule1')
    parser.add_argument('--aa_low', type=float, default=0.0001, help="Alpha Accum lower bound")
    parser.add_argument("--weight_file", type=str, default='./output7_schedule1/weight_loss_sqrt.txt')
    parser.add_argument("--load_ckpt_path", type=str, default='')
    parser.add_argument("--aacum0_lambda", type=float, default=100)
    args = parser.parse_args()

    # add device
    gpu_ids = args.gpu_ids
    log_fn(f"gpu_ids : {gpu_ids}")
    args.device = f"cuda:{gpu_ids[0]}" if torch.cuda.is_available() and gpu_ids else "cpu"
    cudnn.benchmark = True

    if not os.path.exists(args.output_dir):
        log_fn(f"os.makedirs({args.output_dir})")
        os.makedirs(args.output_dir)

    return args

def model_save(args, model, e_idx):
    real_model = model
    if isinstance(real_model, torch.nn.DataParallel):
        real_model = real_model.module
    states = {
        'model': real_model.state_dict(),
        'cur_epoch': e_idx
    }
    if e_idx % 1000 == 0:
        ek = e_idx // 1000
        fpath = os.path.join(args.output_dir, f"ckpt_E{ek:04d}K.pth")
    else:
        fpath = os.path.join(args.output_dir, f"ckpt_E{e_idx:07d}.pth")
    log_fn(f"save ckpt: {fpath}")
    torch.save(states, fpath)

def model_load(ckpt_path, model):
    log_fn(f"load ckpt: {ckpt_path}")
    states = torch.load(ckpt_path)
    log_fn(f"  model.load_state_dict(states['model'])")
    log_fn(f"  old epoch: {states['cur_epoch']}")
    model.load_state_dict(states['model'])
    return states['cur_epoch']

def model_generate(args):
    model = Schedule1Model()
    e_start = 0  # epoch start
    if args.load_ckpt_path:
        e_start = model_load(args.load_ckpt_path, model)

    log_fn(f"model.to({args.device})")
    model.to(args.device)
    if len(args.gpu_ids) > 1:
        log_fn(f"model = DataParallel(model, device_ids={args.gpu_ids})")
        model = DataParallel(model, device_ids=args.gpu_ids)
    return model, e_start

def main():
    args = parse_args()
    log_fn(f"pid : {os.getpid()}")
    log_fn(f"cwd : {os.getcwd()}")
    log_fn(f"args: {args}")
    if args.weight_file:
        weight_arr = ScheduleBase.load_floats(args.weight_file)
        weight_arr = weight_arr.to(args.device)
        str_coef = 'coefficient_wt'
    else:
        weight_arr = None
        str_coef = 'coefficient'

    o_dir = args.output_dir
    model, e_start = model_generate(args)
    lr = args.lr
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    e_cnt = args.n_epochs
    log_fn(f"lr      : {lr}")
    log_fn(f"e_cnt   : {e_cnt}")
    log_fn(f"e_start : {e_start}")
    log_fn(f"aa_low  : {args.aa_low}")
    log_fn(f"torch.seed() : {torch.seed()}")
    log_fn(f"aacum0_lambda: {args.aacum0_lambda}")
    model.train()
    aacum, alpha, coefficient = None, None, None
    start_time = time.time()
    for e_idx in range(e_start, e_cnt):
        optimizer.zero_grad()
        alpha, aacum, coefficient = model()
        if weight_arr is not None:
            coefficient = torch.mul(coefficient, weight_arr)

        loss_var = torch.var(coefficient)
        aa_min = aacum[-1]
        loss_min = torch.square(aa_min - args.aa_low) * args.aacum0_lambda
        loss = torch.add(loss_var, loss_min)
        loss.backward()
        optimizer.step()
        if e_idx % 2000 == 0 or e_idx == e_cnt - 1:
            elp, eta = utils.get_time_ttl_and_eta(start_time, e_idx - e_start, e_cnt - e_start)
            if loss < 0.000001 and loss_var < 0.000001 and loss_min < 0.000001:
                ls, loss, loss_var, loss_min = 'ls*M', loss * 1000000, loss_var * 1000000, loss_min * 1000000
            elif loss < 0.001 and loss_var < 0.001 and loss_min < 0.001:
                ls, loss, loss_var, loss_min = 'ls*K', loss * 1000, loss_var * 1000, loss_min * 1000
            else:
                ls = 'loss'
            log_fn(f"E{e_idx:05d}/{e_cnt} {ls}: {loss:.8f} {loss_var:.8f} {loss_min:.8f}."
                   f" aa:{aacum[0]:.12f}~{aacum[-1]:.8f}. elp:{elp}, eta:{eta}")

            fig, axs = plt.subplots()
            x_axis = np.arange(0, 1000)
            axs.plot(x_axis, coefficient.detach().cpu().numpy(), label=str_coef)
            axs.set_xlabel(f"timestep. Epoch: {e_idx}")
            plt.legend()
            fig.savefig(os.path.join(o_dir, f"et_{str_coef}.png"))
            plt.close()
            utils.save_list(aacum, f"aacum_{e_idx}", os.path.join(o_dir, 'et_aacum.txt'))
            utils.save_list(alpha, f"alpha_{e_idx}", os.path.join(o_dir, 'et_alpha.txt'))
            utils.save_list(coefficient, f"coef{e_idx}", os.path.join(o_dir, 'et_coef.txt'))

        if e_idx > 0 and e_idx % 50000 == 0 or e_idx == e_cnt - 1:
            model_save(args, model, e_idx)
    # for e_idx
    utils.output_list(aacum, 'alpha_aacum')
    utils.output_list(alpha, 'alpha')
    utils.output_list(coefficient, str_coef)
    return 0


if __name__ == "__main__":
    sys.exit(main())
