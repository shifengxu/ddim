"""
Find beta schedule.
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
log_fn = utils.log_info

torch.set_printoptions(sci_mode=False)


class Schedule1Model(nn.Module):
    def __init__(self, out_channels=1000):
        super().__init__()
        self.linear1 = torch.nn.Linear(1,  50, dtype=torch.float64)
        self.linear2 = torch.nn.Linear(50,  500, dtype=torch.float64)
        self.linear3 = torch.nn.Linear(500,  out_channels, dtype=torch.float64)

    def forward(self, input_seed):
        output1 = self.linear1(input_seed)
        # output1 = output1 * torch.sigmoid(output1)
        output2 = self.linear2(output1)
        # output2 = output2 * torch.sigmoid(output2)
        output3 = self.linear3(output2)
        output = torch.softmax(output3, dim=0)
        return output


def parse_args():
    parser = argparse.ArgumentParser(description=globals()["__doc__"])
    parser.add_argument('--gpu_ids', type=int, nargs='+', default=[7])
    parser.add_argument("--n_epochs", type=int, default=1000000)
    parser.add_argument('--lr', type=float, default=0.2)
    parser.add_argument('--output_dir', type=str, default='./output7_schedule1')
    parser.add_argument('--aa_low', type=float, default=0., help="Alpha Accum lower bound")
    parser.add_argument('--aa_high', type=float, default=1., help="Alpha Accum upper bound")
    parser.add_argument("--loss_file", type=str, default='./output7_schedule1/loss_file.txt')
    parser.add_argument("--load_ckpt_path", type=str, default='./output7_schedule1/ckpt_E0800K.pth')
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
    return model

def main():
    args = parse_args()
    device = args.device
    log_fn(f"pid : {os.getpid()}")
    log_fn(f"cwd : {os.getcwd()}")
    log_fn(f"args: {args}")

    model = Schedule1Model()
    if args.load_ckpt_path:
        model = model_load(args.load_ckpt_path, model)

    log_fn(f"model.to({device})")
    model.to(device)
    if len(args.gpu_ids) > 1:
        log_fn(f"model = DataParallel(model, device_ids={args.gpu_ids})")
        model = DataParallel(model, device_ids=args.gpu_ids)
    lr = args.lr
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    e_cnt = args.n_epochs
    seed = torch.tensor((0.5,), dtype=torch.float64, device=device)
    log_fn(f"lr   : {lr}")
    log_fn(f"e_cnt: {e_cnt}")
    log_fn(f"seed : {seed}")
    model.train()
    aacum, alpha, coefficient = None, None, None
    start_time = time.time()
    for e_idx in range(0, e_cnt):
        optimizer.zero_grad()
        output = model(seed)
        aacum = torch.cumsum(output, dim=0)
        aacum0 = aacum[0]
        aacum = torch.flip(aacum, dims=(0,))
        aacum = aacum * 0.99999
        aa_prev = torch.cat([torch.ones(1).to(device), aacum[:-1]], dim=0)
        alpha = torch.div(aacum, aa_prev)

        # make sure alpha > aacum.
        # Or else, the 2nd epoch will have output: tensor([nan, nan, ,,,])
        alpha[0] += 1e-8

        coefficient = ((1-aacum).sqrt() - (alpha-aacum).sqrt()) / alpha.sqrt()
        loss_var = torch.var(coefficient)
        loss_min = torch.square(aacum0 - 0.0001)
        loss = loss_var + 100 * loss_min
        loss.backward()
        optimizer.step()
        e2 = e_idx + 1
        if e2 % 2000 == 0 or e2 == e_cnt:
            elp, eta = utils.get_time_ttl_and_eta(start_time, e2, e_cnt)
            log_fn(f"E{e2:05d}/{e_cnt} loss: {loss:10.8f} {loss_var:10.8f} {loss_min:.10f}."
                   f" {aacum0:10.8f}. elp:{elp}, eta:{eta}")
            fig, axs = plt.subplots()
            x_axis = np.arange(0, 1000)
            axs.plot(x_axis, coefficient.detach().cpu().numpy())
            axs.set_xlabel(f"timestep. Epoch: {e2}")
            axs.set_ylabel('et_coefficient')
            # plt.show()
            fig.savefig(os.path.join(args.output_dir, 'et_coefficient.png'))
            plt.close()
        if e2 % 50000 == 0 or e2 == e_cnt:
            model_save(args, model, e2)
    # for e_idx
    utils.output_list(aacum, 'alpha_aacum')
    utils.output_list(alpha, 'alpha')
    utils.output_list(coefficient, 'coefficient')
    return 0


if __name__ == "__main__":
    sys.exit(main())
