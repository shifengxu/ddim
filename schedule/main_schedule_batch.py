"""
Find beta schedule by vivid variance, in batch.
Vivid variance is also final variance by default.
"""
import argparse
import sys
import os
import time

import torch
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
    parser.add_argument("--n_epochs", type=int, default=10000)
    parser.add_argument('--log_interval', type=int, default=2000)
    parser.add_argument('--lr', type=float, default=0.000001)
    parser.add_argument('--lp', type=float, default=0.01, help='learning_portion')
    parser.add_argument('--output_dir', type=str, default='./output7_vividvar')
    parser.add_argument('--alpha_bar_dir', type=str, default='./exp/dpm_alphaBar')
    parser.add_argument('--aa_low', type=float, default=0.0001, help="Alpha Accum lower bound")
    parser.add_argument("--aa_low_lambda", type=float, default=10000000)
    parser.add_argument("--weight_file", type=str, default='./output7_vividvar/res_mse_avg_list.txt')
    parser.add_argument("--beta_schedule", type=str, default="linear")
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

def model_generate(alpha_bar, lp, gpu_ids, device):
    model = ScheduleParamAlphaModel(alpha_bar=alpha_bar, learning_portion=lp)
    log_fn(f"model: {type(model).__name__}")
    log_fn(f"  out_channels: {model.out_channels}")
    log_fn(f"  model.to({device})")
    model.to(device)
    if len(gpu_ids) > 1:
        log_fn(f"model = DataParallel(model, device_ids={gpu_ids})")
        model = DataParallel(model, device_ids=gpu_ids)
    return model

def load_floats_from_file(f_path, c_arr):
    log_fn(f"load_floats_from_file() from file: {f_path}")
    with open(f_path, 'r') as f:
        lines = f.readlines()
    cnt_empty = 0
    cnt_comment = 0
    order = None
    f_arr = []
    for line in lines:
        line = line.strip()
        if line == '':
            cnt_empty += 1
            continue
        if line.startswith('#'):
            cnt_comment += 1
            c_arr.append(line)
            key, val = line[1:].split(':')
            if key.strip() == 'order':
                if order is None: order = val.strip()
                else: raise ValueError(f"duplicate order in {f_path}")
            continue
        flt = float(line)
        f_arr.append(flt)
    log_fn(f"  cnt_empty  : {cnt_empty}")
    log_fn(f"  cnt_comment: {cnt_comment}")
    log_fn(f"  cnt_valid  : {len(f_arr)}")
    if order is None: raise ValueError(f"Not found order in {f_path}")
    return f_arr, int(order)

def main():
    args = parse_args()
    log_fn(f"pid : {os.getpid()}")
    log_fn(f"cwd : {os.getcwd()}")
    log_fn(f"args: {args}")
    var_arr = ScheduleBase.load_floats(args.weight_file)
    vs = VarSimulator2(args.beta_schedule, var_arr)
    vs.to(args.device)
    ab_dir = args.alpha_bar_dir
    m_arr = [f"lr           : {args.lr}",
             f"n_epochs     : {args.n_epochs}",
             f"aa_low       : {args.aa_low}",
             f"aa_low_lambda: {args.aa_low_lambda}",
             f"beta_schedule: {args.beta_schedule}",
             f"torch.seed() : {torch.seed()}",
             f"alpha_bar_dir: {ab_dir}"]  # message array
    [log_fn(m) for m in m_arr]
    file_list = [f for f in os.listdir(ab_dir) if 'dpm_alphaBar' in f and f.endswith('.txt')]
    file_list = [os.path.join(ab_dir, f) for f in file_list]
    file_list = [f for f in file_list if os.path.isfile(f)]
    f_cnt = len(file_list)
    log_fn(f"alpha_bar file cnt: {f_cnt}")
    for idx, f_path in enumerate(sorted(file_list)):
        log_fn(f"{idx:03d}/{f_cnt}: {f_path} ****************************")
        c_arr = [f" Old comments in file {f_path}"]  # comment array
        alpha_bar, order = load_floats_from_file(f_path, c_arr)
        c_arr = [c[1:].strip() for c in c_arr]  # remove prefix '#'
        alpha_bar = alpha_bar[1:]  # ignore the first one, as it is for timestep 0
        _, idx_arr = vs(torch.tensor(alpha_bar, device=args.device), include_index=True)
        s_arr = [f"{alpha_bar[i]:.8f} {idx_arr[i]:4d}" for i in range(len(alpha_bar))]
        s_arr.insert(0, "Old alpha_bar and its timestep")
        new_arr = c_arr + [''] + s_arr + [''] + m_arr
        f_name = os.path.basename(f_path)
        train(args, vs, alpha_bar, order, new_arr, f_name)
    return 0

def calc_loss(alpha, aacum, weight_arr, order):
    if order == 1:
        loss_var, coef, numerator, sub_var = ScheduleBase.accumulate_variance(alpha, aacum, weight_arr, True)
        return loss_var, coef, weight_arr, numerator, sub_var
    elif order == 2:
        return calc_loss_order2(aacum, weight_arr)
    elif order == 3:
        return calc_loss_order3_v2(aacum, weight_arr)
    else:
        raise ValueError(f"Unsupported order {order}")

def calc_loss_order2(aacum, weight_arr):
    a_cnt = len(aacum)
    if a_cnt % 2 == 0:
        # aacum index and weight index. if a_cnt is 8, then index is 0 ~ 7
        idx_a = list(range(1, a_cnt, 2))  # [1, 3, 5, 7]
        idx_w = list(range(0, a_cnt, 2))  # [0, 2, 4, 6]
    else:
        # aacum index and weight index. if a_cnt is 9, then index is 0 ~ 8
        idx_a = list(range(0, a_cnt, 2))  # [0, 2, 4, 6, 8]
        idx_w = list(range(1, a_cnt, 2))  # [1, 3, 5, 7]
        idx_w = [0] + idx_w  # [0, 1, 3, 5, 7]
    idx_a = torch.tensor(idx_a, dtype=torch.long, device=aacum.device)
    idx_w = torch.tensor(idx_w, dtype=torch.long, device=aacum.device)
    aa2 = torch.index_select(aacum, dim=0, index=idx_a)  # new aacum
    tmp = [torch.ones((1,), device=aacum.device), aa2[:-1]]
    tmp = torch.cat(tmp, dim=0)
    al2 = aa2 / tmp  # new alpha
    wt1 = torch.index_select(weight_arr, dim=0, index=idx_a)
    wt2 = torch.index_select(weight_arr, dim=0, index=idx_w)  # new weight
    wt2 += 1.0 * wt1  # plus some weight
    loss_var, c, n, s = ScheduleBase.accumulate_variance(al2, aa2, wt2, True)
    coef = torch.zeros_like(aacum)
    numerator = torch.zeros_like(aacum)
    sub_var = torch.zeros_like(aacum)
    weight = torch.zeros_like(aacum)
    coef[idx_a] = c
    numerator[idx_a] = n
    sub_var[idx_a] = s
    weight[idx_a] = wt2
    return loss_var, coef, weight, numerator, sub_var

def calc_loss_order3(aacum, weight_arr):
    """
    For detailed explanation, please see the doc: Readme_DPM_Solver3_predefined.docx
    :param aacum:
    :param weight_arr:
    :return:
    """
    a_cnt = len(aacum)
    if a_cnt % 3 == 0:
        # if a_cnt is  9, then index will be 0 ~ 8.
        # Then inner jump size array is [3, 3, 2, 1]
        # aacum idx will be [0, 2, 5, 8]
        # weight index is complicated. For the case of jump size 3, each step involves 2 weights.
        # weight series 1 : [0, 1, 5, 8]
        # weight series 2:  [      3, 6]
        # aacum index and weight index.
        idx_a = [0, 2] + list(range(5, a_cnt, 3))  # [0, 2, 5, 8]
        idx_w = [0, 1] + list(range(5, a_cnt, 3))  # [0, 1, 5, 8] weight series 1
        idx_v = list(range(3, a_cnt, 3))           # [      3, 6] weight series 2
        idx_a = torch.tensor(idx_a, dtype=torch.long, device=aacum.device)
        idx_w = torch.tensor(idx_w, dtype=torch.long, device=aacum.device)
        idx_v = torch.tensor(idx_v, dtype=torch.long, device=aacum.device)
        aa3 = torch.index_select(aacum, dim=0, index=idx_a)       # new aacum
        wt3 = torch.index_select(weight_arr, dim=0, index=idx_w)  # new weight
        wtv = torch.index_select(weight_arr, dim=0, index=idx_v)
        wt3[2:] += wtv # append series 2 into series 1.
    elif a_cnt % 3 == 1:
        # if a_cnt is 10, then index will be 0 ~ 9.
        # Then inner jump size array [3, 3, 3, 1]
        # aacum idx will be [0, 3, 6, 9]
        # weight series 1 : [0, 3, 6, 9]
        # weight series 2:  [   1, 4, 7]
        # aacum index and weight index.
        idx_a = list(range(0, a_cnt, 3))  # [0, 3, 6, 9]
        idx_w = list(range(0, a_cnt, 3))  # [0, 3, 6, 9] weight series 1
        idx_v = list(range(1, a_cnt, 3))  # [   1, 4, 7] weight series 2
        idx_a = torch.tensor(idx_a, dtype=torch.long, device=aacum.device)
        idx_w = torch.tensor(idx_w, dtype=torch.long, device=aacum.device)
        idx_v = torch.tensor(idx_v, dtype=torch.long, device=aacum.device)
        aa3 = torch.index_select(aacum, dim=0, index=idx_a)       # new aacum
        wt3 = torch.index_select(weight_arr, dim=0, index=idx_w)  # new weight
        wtv = torch.index_select(weight_arr, dim=0, index=idx_v)
        wt3[1:] += wtv # append series 2 into series 1.
    else: # a_cnt % 3 == 2
        # If a_cnt is 11, then the index will be 0 ~ 10
        # Then inner jump size array [3, 3, 3, 2]
        # aacum idx will be [1, 4, 7, 10]
        # weight series 1 : [0, 4, 7, 10]
        # weight series 2:  [   2, 5,  8]
        # aacum index and weight index.
        idx_a = list(range(1, a_cnt, 3))        # [1, 4, 7, 10]
        idx_w = [0] + list(range(4, a_cnt, 3))  # [0, 4, 7, 10] weight series 1
        idx_v = list(range(2, a_cnt, 3))        # [   2, 5,  8] weight series 2
        idx_a = torch.tensor(idx_a, dtype=torch.long, device=aacum.device)
        idx_w = torch.tensor(idx_w, dtype=torch.long, device=aacum.device)
        idx_v = torch.tensor(idx_v, dtype=torch.long, device=aacum.device)
        aa3 = torch.index_select(aacum, dim=0, index=idx_a)       # new aacum
        wt3 = torch.index_select(weight_arr, dim=0, index=idx_w)  # new weight
        wtv = torch.index_select(weight_arr, dim=0, index=idx_v)
        wt3[1:] += wtv # append series 2 into series 1.
    tmp = [torch.ones((1,), device=aacum.device), aa3[:-1]]
    tmp = torch.cat(tmp, dim=0)
    al3 = aa3 / tmp  # new alpha
    loss_var, c, n, s = ScheduleBase.accumulate_variance(al3, aa3, wt3, True)
    coef = torch.zeros_like(aacum)
    numerator = torch.zeros_like(aacum)
    sub_var = torch.zeros_like(aacum)
    weight = torch.zeros_like(aacum)
    coef[idx_a] = c
    numerator[idx_a] = n
    sub_var[idx_a] = s
    weight[idx_a] = wt3
    return loss_var, coef, weight, numerator, sub_var

def calc_loss_order3_v2(aacum, weight_arr):
    """
    the original code follows the DPM Solver-3 formula, but not working well with time_uniform.
    So try the below code.
    :param aacum:
    :param weight_arr:
    :return:
    """
    a_cnt = len(aacum)
    if a_cnt % 3 == 0:
        # if a_cnt is  9, then index will be 0 ~ 8.
        idx_a = list(range(2, a_cnt, 3))  # [2, 5, 8]
        idx_w = list(range(2, a_cnt, 3))  # [2, 5, 8] weight series 1
        idx_v = list(range(0, a_cnt, 3))  # [0, 3, 6] weight series 2
        idx_u = list(range(1, a_cnt, 3))  # [1, 4, 7] weight series 3
        idx_a = torch.tensor(idx_a, dtype=torch.long, device=aacum.device)
        idx_w = torch.tensor(idx_w, dtype=torch.long, device=aacum.device)
        idx_v = torch.tensor(idx_v, dtype=torch.long, device=aacum.device)
        idx_u = torch.tensor(idx_u, dtype=torch.long, device=aacum.device)
        aa3 = torch.index_select(aacum, dim=0, index=idx_a)       # new aacum
        wt3 = torch.index_select(weight_arr, dim=0, index=idx_w)  # new weight
        wtv = torch.index_select(weight_arr, dim=0, index=idx_v)
        wtu = torch.index_select(weight_arr, dim=0, index=idx_u)
        wt3 += wtv # append series 2 into series 1.
        wt3 += wtu # append series 3 into series 1.
    elif a_cnt % 3 == 1:
        # if a_cnt is 10, then index will be 0 ~ 9.
        idx_a = list(range(0, a_cnt, 3))  # [0, 3, 6, 9]
        idx_w = list(range(0, a_cnt, 3))  # [0, 3, 6, 9] weight series 1
        idx_v = list(range(1, a_cnt, 3))  # [   1, 4, 7] weight series 2
        idx_u = list(range(2, a_cnt, 3))  # [   2, 5, 8] weight series 3
        idx_a = torch.tensor(idx_a, dtype=torch.long, device=aacum.device)
        idx_w = torch.tensor(idx_w, dtype=torch.long, device=aacum.device)
        idx_v = torch.tensor(idx_v, dtype=torch.long, device=aacum.device)
        idx_u = torch.tensor(idx_u, dtype=torch.long, device=aacum.device)
        aa3 = torch.index_select(aacum, dim=0, index=idx_a)       # new aacum
        wt3 = torch.index_select(weight_arr, dim=0, index=idx_w)  # new weight
        wtv = torch.index_select(weight_arr, dim=0, index=idx_v)
        wtu = torch.index_select(weight_arr, dim=0, index=idx_u)
        wt3[1:] += (wt3[1:] + wtv + wtu) / 3 # append series 2 & 3 into series 1.
    else: # a_cnt % 3 == 2
        # If a_cnt is 11, then the index will be 0 ~ 10
        idx_a = list(range(1, a_cnt, 3))        # [1, 4, 7, 10]
        idx_w = [0] + list(range(4, a_cnt, 3))  # [0, 4, 7, 10] weight series 1
        idx_v = list(range(2, a_cnt, 3))        # [   2, 5,  8] weight series 2
        idx_u = [1] + list(range(3, a_cnt, 3))  # [1, 3, 6,  9] weight series 3
        idx_a = torch.tensor(idx_a, dtype=torch.long, device=aacum.device)
        idx_w = torch.tensor(idx_w, dtype=torch.long, device=aacum.device)
        idx_v = torch.tensor(idx_v, dtype=torch.long, device=aacum.device)
        idx_u = torch.tensor(idx_u, dtype=torch.long, device=aacum.device)
        aa3 = torch.index_select(aacum, dim=0, index=idx_a)       # new aacum
        wt3 = torch.index_select(weight_arr, dim=0, index=idx_w)  # new weight
        wtv = torch.index_select(weight_arr, dim=0, index=idx_v)
        wtu = torch.index_select(weight_arr, dim=0, index=idx_u)
        wt3 += wtu # append series 2 & 3 into series 1.
        wt3[1:] += wtv
    tmp = [torch.ones((1,), device=aacum.device), aa3[:-1]]
    tmp = torch.cat(tmp, dim=0)
    al3 = aa3 / tmp  # new alpha
    loss_var, c, n, s = ScheduleBase.accumulate_variance(al3, aa3, wt3, True)
    coef = torch.zeros_like(aacum)
    numerator = torch.zeros_like(aacum)
    sub_var = torch.zeros_like(aacum)
    weight = torch.zeros_like(aacum)
    coef[idx_a] = c
    numerator[idx_a] = n
    sub_var[idx_a] = s
    weight[idx_a] = wt3
    return loss_var, coef, weight, numerator, sub_var

def train(args, vs, alpha_bar, order, m_arr, f_name):
    model = model_generate(alpha_bar, args.lp, args.gpu_ids, args.device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    model.train()
    start_time = time.time()
    loss_low = None
    loss_ori = None
    e_cnt = args.n_epochs
    f_path = os.path.join(args.output_dir, f"abDetail_{f_name}")
    for e_idx in range(0, e_cnt):
        optimizer.zero_grad()
        alpha, aacum = model()
        weight_arr, idx_arr = vs(aacum, include_index=True)
        loss_var, coef, weight_arr, numerator, sub_var = calc_loss(alpha, aacum, weight_arr, order)
        if loss_ori is None: loss_ori = loss_var.item()
        aa_min = aacum[-1]
        loss_min = torch.square(aa_min - args.aa_low) * args.aa_low_lambda
        loss = torch.add(loss_var, loss_min)
        loss.backward()
        model.gradient_clip()
        optimizer.step()
        if e_idx % args.log_interval == 0 or e_idx == e_cnt - 1:
            elp, eta = utils.get_time_ttl_and_eta(start_time, e_idx, e_cnt)
            log_fn(f"E{e_idx:05d}/{e_cnt} loss: {loss_var:.5f} {loss_min:.5f}."
                   f" a:{alpha[0]:.8f}~{alpha[-1]:.8f};"
                   f" aa:{aacum[0]:.8f}~{aacum[-1]:.5f}. elp:{elp}, eta:{eta}")
            if loss_low is None or loss_low > loss.item():
                loss_low = loss.item()
                mm = list(m_arr)
                mm.append(f"Epoch        : {e_idx:06d}; loss:{loss:05.6f} = {loss_var:05.6f} + {loss_min:05.6f}")
                mm.append(f"loss_var     : {loss_ori:10.6f} => {loss_var:10.6f}")
                mm.append(f"model.learning_portion: {model.learning_portion}")
                mm.append(f"model.out_channels    : {model.out_channels}")
                detail_save(f_path, alpha, aacum, idx_arr, weight_arr, coef, numerator, sub_var, mm)
            # if
        # if
    # for e_idx

def detail_save(f_path, alpha, aacum, idx_arr, weight_arr, coef, numerator, sub_var, m_arr):
    combo = []
    for i in range(len(aacum)):
        s = f"{aacum[i]:8.6f}: {idx_arr[i]:3d}: {alpha[i]:8.6f};" \
            f" {coef[i]:8.6f}*{weight_arr[i]:11.6f}={numerator[i]:9.6f};" \
            f" {numerator[i]:9.6f}/{aacum[i]:8.6f}={sub_var[i]:10.6f}"
        s = s.replace('0.000000', '0.0     ')
        combo.append(s)
    m_arr.append('aacum : ts : alpha   ; coef    *weight     =numerator; numerator/aacum   =sub_var')
    log_fn(f"Save file: {f_path}")
    with open(f_path, 'w') as f_ptr:
        [f_ptr.write(f"# {m}\n") for m in m_arr]
        [f_ptr.write(f"{s}\n") for s in combo]
    # with


if __name__ == "__main__":
    sys.exit(main())
