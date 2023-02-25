"""
Find beta schedule by vivid variance.
Vivid variance is also final variance by default.
"""
import argparse
import sys
import os
import time
import matplotlib.pyplot as plt
from torch import optim
from torch.backends import cudnn
from torch.nn import DataParallel

# add current dir and parent dir into python-path.
# this it to facilitate the Linux start.exe file.
cur_dir = os.path.dirname(__file__)
prt_dir = os.path.dirname(cur_dir)  # parent dir
if cur_dir not in sys.path:
    sys.path.append(cur_dir)
if prt_dir not in sys.path:
    sys.path.append(prt_dir)

from base import *
from var_simulator2 import VarSimulator2

log_fn = utils.log_info

torch.set_printoptions(sci_mode=False)
torch.autograd.set_detect_anomaly(True)

def parse_args():
    parser = argparse.ArgumentParser(description=globals()["__doc__"])
    parser.add_argument('--gpu_ids', type=int, nargs='+', default=[7])
    parser.add_argument('--todo', type=str, default='train')
    parser.add_argument("--n_epochs", type=int, default=500000)
    parser.add_argument('--log_interval', type=int, default=2000)
    parser.add_argument('--lr', type=float, default=0.000001)
    parser.add_argument('--output_dir', type=str, default='./output7_vividvar')
    parser.add_argument('--aa_low', type=float, default=0.0001, help="Alpha Accum lower bound")
    parser.add_argument("--aa_low_lambda", type=float, default=10000000)
    parser.add_argument("--weight_file", type=str, default='./output7_vividvar/res_mse_avg_list.txt')
    parser.add_argument("--load_ckpt_path", type=str, default='')
    parser.add_argument("--beta_schedule", type=str, default="linear")
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
        'out_channels': real_model.out_channels,
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
    out_ch = states['out_channels']
    log_fn(f"  model.load_state_dict(states['model'])")
    log_fn(f"  old epoch    : {states['cur_epoch']}")
    log_fn(f"  out_channels : {out_ch}")
    if out_ch != model.out_channels:
        msg = f"model out_channels mismatch. loaded: {out_ch}; defined: {model.out_channels}."
        msg += " The defined value should be from args parameters."
        raise Exception(msg)
    model.load_state_dict(states['model'])
    return states['cur_epoch']

def model_generate(args):
    model = ScheduleAlphaModel()
    e_start = 0  # epoch start
    if args.load_ckpt_path:
        e_start = model_load(args.load_ckpt_path, model)

    log_fn(f"model: {type(model).__name__}")
    log_fn(f"  out_channels: {model.out_channels}")
    log_fn(f"  model.to({args.device})")
    model.to(args.device)
    if len(args.gpu_ids) > 1:
        log_fn(f"model = DataParallel(model, device_ids={args.gpu_ids})")
        model = DataParallel(model, device_ids=args.gpu_ids)
    return model, e_start

def save_vs_plot(args, vs, var_arr):
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

def sample(args, model, e_idx, vs):
    log_fn(f"sample() =======================")
    model.eval()
    with torch.no_grad():
        alpha, aacum = model()
    new_weight_arr, idx_arr = vs(aacum, include_index=True)
    l_var = ScheduleBase.accumulate_variance(alpha, aacum, new_weight_arr)
    aa_min = aacum[-1]
    l_min = torch.square(aa_min - args.aa_low) * args.aa_low_lambda
    loss = torch.add(l_var, l_min)
    el_str = f"Sample by epoch:{e_idx:06d}; loss:{loss:05.2f} {l_var:05.2f} {l_min:05.2f}"
    status_save(args, alpha, aacum, idx_arr, new_weight_arr, el_str)
    log_fn(f" sampling done.")

def indexing(args, vs):
    log_fn(f"indexing() =======================")
    aacum = [# 0.000040,
             0.000108, 0.000275, 0.000664, 0.001527,
             0.003338, 0.006937, 0.013707, 0.025754, 0.046017,
             0.078191, 0.126356, 0.194200, 0.283887, 0.394732,
             0.522087, 0.656885, 0.786252, 0.895329, 0.970005, 0.999900]
    aacum.reverse()
    aacum = torch.tensor(aacum, device=args.device)
    aa_prev = torch.cat([torch.ones(1).to(args.device), aacum[:-1]], dim=0)
    alpha = torch.div(aacum, aa_prev)
    new_weight_arr, idx_arr = vs(aacum, include_index=True)
    loss_var, coef, numerator, sub_var = ScheduleBase.accumulate_variance(alpha, aacum, new_weight_arr, True)
    aa_min = aacum[-1]
    l_min = torch.square(aa_min - args.aa_low) * args.aa_low_lambda
    loss = torch.add(loss_var, l_min)
    m_arr2 = [f"indexing. loss:{loss:05.2f} = {loss_var:05.2f} + {l_min:05.2f}"]
    detail_save(args, alpha, aacum, idx_arr, new_weight_arr, loss_var, coef, numerator, sub_var, m_arr2)
    log_fn(f" indexing done.")

def main():
    args = parse_args()
    log_fn(f"pid : {os.getpid()}")
    log_fn(f"cwd : {os.getcwd()}")
    log_fn(f"args: {args}")
    var_arr = ScheduleBase.load_floats(args.weight_file)
    vs = VarSimulator2(args.beta_schedule, var_arr, mode=args.vs_mode)
    # save_vs_plot(args, vs, var_arr)
    vs.to(args.device)

    model, e_start = model_generate(args)
    if args.todo == 'sample':
        return sample(args, model, e_start, vs)
    if args.todo == 'indexing':
        return indexing(args, vs)

    lr = args.lr
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    e_cnt = args.n_epochs
    m_arr = [f"lr      : {lr}",
             f"e_cnt   : {e_cnt}",
             f"e_start : {e_start}",
             f"aa_low  : {args.aa_low}",
             f"aa_low_lambda: {args.aa_low_lambda}",
             f"beta_schedule: {args.beta_schedule}",
             f"torch.seed() : {torch.seed()}"]  # message array
    [log_fn(m) for m in m_arr]
    model.train()
    aacum, alpha = None, None
    start_time = time.time()
    loss_track = None
    for e_idx in range(e_start, e_cnt):
        optimizer.zero_grad()
        alpha, aacum = model()
        new_weight_arr, idx_arr = vs(aacum, include_index=True)
        loss_var, coef, numerator, sub_var = ScheduleBase.accumulate_variance(alpha, aacum, new_weight_arr, True)
        aa_min = aacum[-1]
        loss_min = torch.square(aa_min - args.aa_low) * args.aa_low_lambda
        loss = torch.add(loss_var, loss_min)
        loss.backward()
        model.gradient_clip()
        optimizer.step()
        if e_idx % args.log_interval == 0 or e_idx == e_cnt - 1:
            elp, eta = utils.get_time_ttl_and_eta(start_time, e_idx - e_start, e_cnt - e_start)
            log_fn(f"E{e_idx:05d}/{e_cnt} loss: {loss_var:.5f} {loss_min:.5f}."
                   f" a:{alpha[0]:.8f}~{alpha[-1]:.8f};"
                   f" aa:{aacum[0]:.8f}~{aacum[-1]:.5f}. elp:{elp}, eta:{eta}")
            if loss_track is None or loss_track > loss.item():
                loss_track = loss.item()
                m_arr2 = m_arr + [f"Epoch:{e_idx:06d}; loss:{loss:05.2f} = {loss_var:05.2f} + {loss_min:05.2f}"]
                # status_save(args, alpha, aacum, idx_arr, new_weight_arr, m_arr2)
                detail_save(args, alpha, aacum, idx_arr, new_weight_arr, loss_var, coef, numerator, sub_var, m_arr2)

        if e_idx > 0 and e_idx % 50000 == 0 or e_idx == e_cnt - 1:
            model_save(args, model, e_idx)
    # for e_idx
    utils.output_list(aacum, 'alpha_aacum')
    utils.output_list(alpha, 'alpha')
    return 0

def status_save(args, alpha, aacum, index, weight_arr, el_str):
    alpha[0] += 1e-12
    coef = ((1 - aacum).sqrt() - (alpha - aacum).sqrt()) / alpha.sqrt()
    fig, axs = plt.subplots()
    x_axis = np.arange(0, len(aacum))
    std_d = coef.detach() * torch.sqrt(weight_arr)  # standard deviation
    axs.plot(x_axis, std_d.detach().cpu().numpy(), label='std vividvar')
    axs.set_xlabel(f"timestep. {el_str}")
    plt.legend()
    o_dir = args.output_dir
    fig.savefig(os.path.join(o_dir, f"et_vividvar.png"))
    plt.close()
    combo = []
    for i in range(len(aacum)):
        combo.append("{:8.6f}: {:3d}".format(aacum[i], index[i]))
    m_arr = list(el_str) if el_str is list else [el_str]
    m_arr.append('')
    m_arr.append('aacum : timestep')
    utils.save_list(combo, '', os.path.join(o_dir, f"et_aacum.txt"), msg=m_arr, fmt="{}")
    utils.save_list(alpha, '', os.path.join(o_dir, f"et_alpha.txt"), msg=el_str)
    utils.save_list(std_d, '', os.path.join(o_dir, f"et_std_d.txt"), msg=el_str)

def detail_save(args, alpha, aacum, idx_arr, new_weight_arr, loss_var, coef, numerator, sub_var, m_arr):
    o_dir = args.output_dir
    combo = []
    for i in range(len(aacum)):
        s = f"{aacum[i]:8.6f}: {idx_arr[i]:3d}: {alpha[i]:8.6f};" \
            f" {coef[i]:8.6f}*{new_weight_arr[i]:10.6f}={numerator[i]:9.6f};" \
            f" {numerator[i]:9.6f}/{aacum[i]:8.6f}={sub_var[i]:10.6f}"
        combo.append(s)
    m_arr.append(f"loss_var: {loss_var:10.6f}")
    m_arr.append('aacum : ts : alpha   ; coef    * weight   =numerator; numerator/aacum   =sub_var')
    utils.save_list(combo, '', os.path.join(o_dir, f"et_detail.txt"), msg=m_arr, fmt="{}")

if __name__ == "__main__":
    sys.exit(main())
