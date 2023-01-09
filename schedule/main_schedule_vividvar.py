"""
Find beta schedule by vivid variance. Also use final var as target
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
from var_simulator2 import VarSimulator2

log_fn = utils.log_info

torch.set_printoptions(sci_mode=False)

def parse_args():
    parser = argparse.ArgumentParser(description=globals()["__doc__"])
    parser.add_argument('--gpu_ids', type=int, nargs='+', default=[7])
    parser.add_argument("--n_epochs", type=int, default=500000)
    parser.add_argument('--lr', type=float, default=0.000001)
    parser.add_argument('--output_dir', type=str, default='./output7_schedule1')
    parser.add_argument('--aa_low', type=float, default=0.0001, help="Alpha Accum lower bound")
    parser.add_argument("--aacum0_lambda", type=float, default=100000000)
    parser.add_argument("--weight_file", type=str, default='./output7_schedule1/weight_loss.txt')
    parser.add_argument("--load_ckpt_path", type=str, default='')
    parser.add_argument("--beta_schedule", type=str, default="cosine")
    parser.add_argument("--vs_mode", type=str, default="vivid", help='var_simulator mode: vivid|static')
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

    log_fn(f"model: {type(model).__name__}")
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
    var_arr = ScheduleBase.load_floats(args.weight_file)
    vs = VarSimulator2(args.beta_schedule, var_arr, mode=args.vs_mode)
    ts_cnt = len(var_arr)
    x_arr = ScheduleBase.get_alpha_cumprod(args.beta_schedule, ts_cnt)
    y_arr = vs(x_arr)
    mse = ((var_arr - y_arr) ** 2).mean(axis=0)
    fig, axs = plt.subplots()
    x_axis = np.arange(0, 1000)
    axs.plot(x_axis, var_arr, label="original")
    axs.plot(x_axis, y_arr.numpy(), label=type(vs).__name__)
    axs.set_xlabel(f"timestep. mse={mse:.8f}")
    plt.legend()
    f_path = os.path.join(args.output_dir, f"var_simulator2.png")
    fig.savefig(f_path)
    plt.close()
    log_fn(f"saved: {f_path}")

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
    vs.to(args.device)
    start_time = time.time()
    loss_track = None
    for e_idx in range(e_start, e_cnt):
        optimizer.zero_grad()
        alpha, aacum, coefficient = model()
        new_weight_arr = vs(aacum)
        loss_var = ScheduleBase.accumulate_variance(alpha, aacum, new_weight_arr)
        aa_min = aacum[-1]
        loss_min = torch.square(aa_min - args.aa_low) * args.aacum0_lambda
        loss = torch.add(loss_var, loss_min)
        loss.backward()
        model.gradient_clip()
        optimizer.step()
        if e_idx % 2000 == 0 or e_idx == e_cnt - 1:
            elp, eta = utils.get_time_ttl_and_eta(start_time, e_idx - e_start, e_cnt - e_start)
            log_fn(f"E{e_idx:05d}/{e_cnt} loss: {loss:.8f} {loss_var:.8f} {loss_min:.8f}."
                   f" aa:{aacum[0]:.12f}~{aacum[-1]:.8f}. elp:{elp}, eta:{eta}")
            if loss_track is None or loss_track > loss.item():
                loss_track = loss.item()
                el_str = f"Epoch:{e_idx:06d}; loss:{loss:05.2f} {loss_var:05.2f} {loss_min:05.2f}"
                status_save(args, alpha, aacum, coefficient, new_weight_arr, el_str)

        if e_idx > 0 and e_idx % 50000 == 0 or e_idx == e_cnt - 1:
            model_save(args, model, e_idx)
    # for e_idx
    utils.output_list(aacum, 'alpha_aacum')
    utils.output_list(alpha, 'alpha')
    return 0

def status_save(args, alpha, aacum, coefficient, new_weight_arr, el_str):
    fig, axs = plt.subplots()
    x_axis = np.arange(0, 1000)
    std_d = coefficient.detach() * torch.sqrt(new_weight_arr)  # standard deviation
    axs.plot(x_axis, std_d.cpu().numpy(), label='std vividvar')
    axs.set_xlabel(f"timestep. {el_str}")
    plt.legend()
    o_dir = args.output_dir
    fig.savefig(os.path.join(o_dir, f"et_vividvar.png"))
    plt.close()
    utils.save_list(aacum, f"aacum", os.path.join(o_dir, f"et_aacum.txt"), msg=el_str)
    utils.save_list(alpha, f"alpha", os.path.join(o_dir, f"et_alpha.txt"), msg=el_str)
    utils.save_list(std_d, f"std_d", os.path.join(o_dir, f"et_std_d.txt"), msg=el_str)

if __name__ == "__main__":
    sys.exit(main())
