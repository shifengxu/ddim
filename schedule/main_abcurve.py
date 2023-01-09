"""
Given a model, we can get the predicted epsilon for each x_t and t.
For each step t, there is error between predicted epsilon and real epsilon.
And the error for all the step t will compose a curve.
Here we will simulate the curve by continuous functions:
    f = alpha_bar_t
    s = 1 - f
    error(s) = a*u^s + b*s^v + C
"""
import argparse
import sys
import os
import time
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn, optim
from torch.backends import cudnn
from torch.nn import DataParallel

import utils
from base import ScheduleBase

log_fn = utils.log_info

torch.set_printoptions(sci_mode=False)

class AbcurveModel(nn.Module):
    def __init__(self, out_channels=5):
        super().__init__()
        self.linear1 = torch.nn.Linear(5,  100, dtype=torch.float64)
        self.linear2 = torch.nn.Linear(100,  100, dtype=torch.float64)
        self.linear3 = torch.nn.Linear(100,  out_channels, dtype=torch.float64)

        # the seed. we choose value 0.5. And it is better than value 1.0
        ones_in = torch.mul(torch.ones((5,), dtype=torch.float64), 0.5)
        self.seed_in = torch.nn.Parameter(ones_in, requires_grad=False)

    def gradient_clip(self):
        if self.linear1.weight.grad is not None:
            self.linear1.weight.grad = torch.tanh(self.linear1.weight.grad)
            self.linear1.weight.grad = torch.tanh(self.linear1.weight.grad)
        if self.linear2.weight.grad is not None:
            self.linear2.weight.grad = torch.tanh(self.linear2.weight.grad)
            self.linear2.weight.grad = torch.tanh(self.linear2.weight.grad)
        if self.linear3.weight.grad is not None:
            self.linear3.weight.grad = torch.tanh(self.linear3.weight.grad)
            self.linear3.weight.grad = torch.tanh(self.linear3.weight.grad)

    def forward(self):

        output = self.linear1(self.seed_in)
        output = self.linear2(output)
        output = self.linear3(output)
        # output = torch.exp(output)  # make output positive
        return output

    @staticmethod
    def curve_fn(input, hypers):
        a, b, c, u, v = hypers
        # a*u^s + b*s^v + C
        res = torch.exp(a * input + b)
        # res = a * torch.pow(u, input) + c
        return res

def parse_args():
    parser = argparse.ArgumentParser(description=globals()["__doc__"])
    parser.add_argument('--gpu_ids', type=int, nargs='+', default=[7])
    parser.add_argument("--n_epochs", type=int, default=50000)
    parser.add_argument('--lr', type=float, default=0.000001)
    parser.add_argument("--beta_schedule", type=str, default="cosine")
    parser.add_argument('--output_dir', type=str, default='./output7_schedule1')
    parser.add_argument("--weight_file", type=str, default='./output7_schedule1/weight_file.txt')
    parser.add_argument("--load_ckpt_path", type=str, default='')
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
    model = AbcurveModel()
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
    o_dir = args.output_dir
    ts_cnt = 1000
    log_fn(f"pid : {os.getpid()}")
    log_fn(f"cwd : {os.getcwd()}")
    log_fn(f"args: {args}")
    target_y = ScheduleBase.load_floats(args.weight_file).to(args.device)
    target_y = torch.flip(target_y, dims=(0,))
    alpha_bar = ScheduleBase.get_alpha_cumprod(args.beta_schedule, ts_cnt=ts_cnt).to(args.device)
    input = 1 - alpha_bar  # if alpha_bar is cosine stype
    # input = torch.arange(1, 1001, device=args.device, dtype=torch.float64) / 1000
    input = input.to(torch.float64)
    utils.output_list(input, 'input')

    model, e_start = model_generate(args)
    lr = args.lr
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    loss_fn = torch.nn.MSELoss()
    e_cnt = args.n_epochs
    log_fn(f"lr      : {lr}")
    log_fn(f"e_cnt   : {e_cnt}")
    log_fn(f"e_start : {e_start}")
    log_fn(f"t.seed(): {torch.seed()}")
    log_fn(f"o_dir   : {o_dir}")
    log_fn(f"ts_cnt  : {ts_cnt}")
    model.train()
    start_time = time.time()
    for e_idx in range(e_start, e_cnt):
        optimizer.zero_grad()
        hypers = model()
        predicted_y = model.curve_fn(input, hypers)
        loss = loss_fn(predicted_y, target_y)
        loss.backward()
        optimizer.step()
        if e_idx % 1000 == 0 or e_idx == e_cnt - 1:
            h_str = f"[{hypers[0]:.4f} {hypers[1]:.4f} {hypers[2]:.4f} {hypers[3]:.4f} {hypers[4]:.4f}]"
            elp, eta = utils.get_time_ttl_and_eta(start_time, e_idx - e_start, e_cnt - e_start)
            log_fn(f"E{e_idx:05d}/{e_cnt} loss: {loss:.4f}. hypers:{h_str}. elp:{elp}, eta:{eta}")

            fig, axs = plt.subplots()
            x_axis = np.arange(0, ts_cnt)
            axs.plot(x_axis, predicted_y.detach().cpu().numpy(), label='predicted_y')
            axs.plot(x_axis, target_y.detach().cpu().numpy(), label='target_y')
            axs.set_xlabel(f"timestep. Epoch: {e_idx}")
            plt.legend()
            # plt.show()
            fig.savefig(os.path.join(o_dir, f"abcurve.png"))
            plt.close()

        if e_idx > 0 and e_idx % 50000 == 0 or e_idx == e_cnt - 1:
            model_save(args, model, e_idx)
    # for e_idx
    return 0


if __name__ == "__main__":
    sys.exit(main())
