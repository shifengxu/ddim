"""
Find beta schedule.
This is final variance, not yet vivid variance.
"""
import argparse
import sys
import os
import time
import torch
from torch import nn, optim
from torch.backends import cudnn
from torch.nn import DataParallel

import utils
log_fn = utils.log_info

torch.set_printoptions(sci_mode=False)

class ScheduleFinalVarModel(nn.Module):
    def __init__(self, out_channels=1000):
        super().__init__()
        self.linear1 = torch.nn.Linear(1000,  2000, dtype=torch.float64)
        self.linear2 = torch.nn.Linear(2000,  2000, dtype=torch.float64)
        self.linear3 = torch.nn.Linear(2000,  out_channels, dtype=torch.float64)

        self.linearMax = torch.nn.Sequential(
            torch.nn.Linear(1, 100, dtype=torch.float64),
            torch.nn.Linear(100, 1, dtype=torch.float64),
        )
        # the seed. we choose value 0.5. And it is better than value 1.0
        ones_k = torch.mul(torch.ones((1000,), dtype=torch.float64), 0.5)
        self.seed_k = torch.nn.Parameter(ones_k, requires_grad=False)
        ones_1 = torch.mul(torch.ones((1,), dtype=torch.float64), 0.5)
        self.seed_1 = torch.nn.Parameter(ones_1, requires_grad=False)

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
        output = self.linear1(self.seed_k)
        output = self.linear2(output)
        output = self.linear3(output)
        output = torch.softmax(output, dim=0)
        aa_max = self.linearMax(self.seed_1)
        aa_max = torch.sigmoid(aa_max)
        return output, aa_max

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_ids', type=int, nargs='+', default=[7])
    parser.add_argument("--n_epochs", type=int, default=500000)
    parser.add_argument('--lr', type=float, default=0.000001)
    parser.add_argument('--output_dir', type=str, default='./output7_schedule1')
    parser.add_argument('--aa_low', type=float, default=0.0001, help="Alpha Accum lower bound")
    parser.add_argument('--aa_low_lambda', type=float, default=10000000)
    parser.add_argument("--weight_file", type=str, default='./output7_schedule1/weight_loss_smooth.txt')
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
    model = ScheduleFinalVarModel()
    e_start = 0  # epoch start
    if args.load_ckpt_path:
        e_start = model_load(args.load_ckpt_path, model)

    log_fn(f"model.to({args.device})")
    model.to(args.device)
    if len(args.gpu_ids) > 1:
        log_fn(f"model = DataParallel(model, device_ids={args.gpu_ids})")
        model = DataParallel(model, device_ids=args.gpu_ids)
    return model, e_start

def weight_load(f_path):
    if not os.path.exists(f_path):
        raise Exception(f"File not found: {f_path}")
    if not os.path.isfile(f_path):
        raise Exception(f"Not file: {f_path}")
    log_fn(f"Read file: {f_path}")
    with open(f_path, 'r') as f:
        lines = f.readlines()
    cnt_empty = 0
    cnt_comment = 0
    f_arr = []
    for line in lines:
        line = line.strip()
        if line == '':
            cnt_empty += 1
            continue
        if line.startswith('#'):
            cnt_comment += 1
            continue
        flt = float(line)
        f_arr.append(flt)
    log_fn(f"  cnt_empty  : {cnt_empty}")
    log_fn(f"  cnt_comment: {cnt_comment}")
    log_fn(f"  cnt_valid  : {len(f_arr)}")
    weights = torch.tensor(f_arr, dtype=torch.float64)
    log_fn(f"  weights first 5: {weights[:5].numpy()}")
    log_fn(f"  weights last 5 : {weights[-5:].numpy()}")
    return weights

def accumulate_variance(alpha, aacum, weight_arr):
    """
    accumulate variance from x_1000 to x_1.
    """
    coef = (1 - aacum).sqrt() - (alpha - aacum).sqrt()
    coef = coef ** 2
    numerator = torch.mul(coef, weight_arr)
    res = torch.div(numerator, aacum)
    res = torch.sum(res)
    return res

def main():
    args = parse_args()
    device = args.device
    log_fn(f"pid : {os.getpid()}")
    log_fn(f"cwd : {os.getcwd()}")
    log_fn(f"args: {args}")
    weight_arr = weight_load(args.weight_file)
    weight_arr = weight_arr.to(args.device)

    o_dir = args.output_dir
    model, e_start = model_generate(args)
    lr = args.lr
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    e_cnt = args.n_epochs
    log_fn(f"lr      : {lr}")
    log_fn(f"e_cnt   : {e_cnt}")
    log_fn(f"e_start : {e_start}")
    log_fn(f"aa_low  : {args.aa_low}")
    log_fn(f"aa_low_lambda  : {args.aa_low_lambda}")
    log_fn(f"torch_seed     : {torch.seed()}")
    model.train()
    aacum, alpha = None, None
    start_time = time.time()
    for e_idx in range(e_start, e_cnt):
        optimizer.zero_grad()
        output, out_max = model()
        aacum = torch.cumsum(output, dim=0)
        aacum0 = aacum[0]
        aacum = torch.flip(aacum, dims=(0,))
        aa_max = (0.9 + 0.0999999999*out_max)  # make sure aa_max is in (0.9, 1)
        aacum = aacum * aa_max
        aa_prev = torch.cat([torch.ones(1).to(device), aacum[:-1]], dim=0)
        alpha = torch.div(aacum, aa_prev)

        # make sure alpha > aacum.
        # Or else, the 2nd epoch will have output: tensor([nan, nan, ,,,])
        alpha[0] += 1e-12

        loss_var = accumulate_variance(alpha, aacum, weight_arr)
        loss_min = torch.square(aacum0 - args.aa_low) * args.aa_low_lambda
        loss = torch.add(loss_var, loss_min)
        loss.backward()
        model.gradient_clip()
        optimizer.step()
        if e_idx % 2000 == 0 or e_idx == e_cnt - 1:
            elp, eta = utils.get_time_ttl_and_eta(start_time, e_idx - e_start, e_cnt - e_start)
            msg = f"E{e_idx:05d}/{e_cnt} loss: {loss:.8f} {loss_var:.8f} {loss_min:.8f}." \
                  f" aa:{aacum[0]:.8f}~{aacum[-1]:.12f}. elp:{elp}, eta:{eta}"
            log_fn(msg)
            utils.save_list(aacum, f"aacum_{e_idx}", os.path.join(o_dir, 'et_aacum.txt'), msg)
            utils.save_list(alpha, f"alpha_{e_idx}", os.path.join(o_dir, 'et_alpha.txt'), msg)

        if e_idx > 0 and e_idx % 50000 == 0 or e_idx == e_cnt - 1:
            model_save(args, model, e_idx)
    # for e_idx
    utils.output_list(aacum, 'alpha_aacum')
    utils.output_list(alpha, 'alpha')
    return 0


if __name__ == "__main__":
    sys.exit(main())
