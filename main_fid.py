import argparse
import datetime
import os
import time

import torch_fidelity
import utils

log_fn = utils.log_info


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_ids', nargs='+', type=int, default=[7])
    parser.add_argument("--seed", type=int, default=0, help="Random seed. 0 means ignore")
    parser.add_argument('--input1', type=str, default="./exp/psmpl_S4E1000TSxxx")
    parser.add_argument('--input2', type=str, default="./exp/psmpl_from_x0")
    args = parser.parse_args()

    gpu_ids = args.gpu_ids
    if gpu_ids:
        log_fn(f"Old: os.environ['CUDA_VISIBLE_DEVICES']: {os.environ.get('CUDA_VISIBLE_DEVICES')}")
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(g) for g in gpu_ids)
        log_fn(f"New: os.environ['CUDA_VISIBLE_DEVICES']: {os.environ['CUDA_VISIBLE_DEVICES']}")
    # The CUDA_VISIBLE_DEVICES environment variable is read by the cuda driver.
    # So it needs to be set before the cuda driver is initialized. It is best
    # if you make sure it is set before importing torch
    import torch

    seed = args.seed  # if seed is 0. then ignore it.
    log_fn(f"gpu_ids   : {gpu_ids}")
    log_fn(f"args.seed : {seed}")
    if seed:
        log_fn(f"  torch.manual_seed({seed})")
        torch.manual_seed(seed)
    if seed and torch.cuda.is_available():
        log_fn(f"  torch.cuda.manual_seed_all({seed})")
        torch.cuda.manual_seed_all(seed)
    log_fn(f"final seed: torch.seed(): {torch.seed()}")
    return args

class FidDirCouple:
    def __init__(self, path1, path2, ts, fid=0.):
        self.path1 = path1
        self.path2 = path2
        self.ts = ts
        self.fid = fid


def init_fdc_arr(dir1, dir2, expected_fc=50000):
    fdc_arr = []  # FidDirCouple array
    for dn1 in os.listdir(dir1):
        path1 = os.path.join(dir1, dn1)
        if not os.path.isdir(path1):
            log_fn(f"!!! Not a dir: {path1}")
            continue
        dn2 = dn1.replace('_x0_', '_gn_') if '_x0_' in dn1 else dn1.replace('_gn_', '_x0_')
        path2 = os.path.join(dir2, dn2)
        if not os.path.exists(path2):
            log_fn(f"!!! Not found dir: {path2}")
            continue
        fc = len(os.listdir(path1))
        if fc != expected_fc:
            log_fn(f"!!! {path1} file count is {fc}, but expect {expected_fc}")
            continue
        fc = len(os.listdir(path2))
        if fc != expected_fc:
            log_fn(f"!!! {path2} file count is {fc}, but expect {expected_fc}")
            continue
        fdc = FidDirCouple(path1, path2, path1.split('_')[-1])
        fdc_arr.append(fdc)  # dn1: from_gn_ts_0000
    fdc_arr.sort(key=lambda x: x.ts)
    return fdc_arr

def main():
    args = parse_args()
    log_fn(f"args: {args}")

    dir1 = args.input1
    dir2 = args.input2
    fdc_arr = init_fdc_arr(dir1, dir2)

    dir_cnt = len(fdc_arr)
    log_fn(f"dir1   : {dir1}")
    log_fn(f"dir2   : {dir2}")
    log_fn(f"dir_cnt: {dir_cnt}")
    log_fn(f"torch_fidelity.calculate_metrics()...")
    time_start = time.time()
    for i in range(dir_cnt):
        fdc = fdc_arr[i]
        path1, path2 = fdc.path1, fdc.path2
        metrics_dict = torch_fidelity.calculate_metrics(
            input1=path1,
            input2=path2,
            cuda=True,
            isc=False,
            fid=True,
            kid=False,
            verbose=False,
        )
        fdc.fid = metrics_dict['frechet_inception_distance']
        elp, rmn = utils.get_time_ttl_and_eta(time_start, i+1, dir_cnt)
        log_fn(f"FID:{i: 3d}/{dir_cnt}. {fdc.ts}: {fdc.fid:.6f}. elp:{elp}, eta:{rmn}")
    ts = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
    with open(os.path.join(dir1, f"fid_{ts}.txt"), "w") as f:
        for fdc in fdc_arr:
            s = f"{fdc.ts}\t{fdc.fid:.6f}"
            log_fn(s)
            f.write(f"{s}\n")
    # with

    return 0


if __name__ == "__main__":
    main()
