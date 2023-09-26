import argparse
import datetime
import os
import time
import torch_fidelity
import utils
import torch
import torch.utils.data as data
import torchvision.transforms as TF
from torchvision.datasets import CIFAR10, ImageFolder

log_fn = utils.log_info


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_ids', nargs='+', type=int, default=[7])
    parser.add_argument("--seed", type=int, default=0, help="Random seed. 0 means ignore")
    parser.add_argument('--input1', type=str, default="./exp/psmpl_S4E1000TSxxx")
    parser.add_argument('--input2', type=str, default="./exp/psmpl_from_x0")
    parser.add_argument('--mode', type=str, default="simple")
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

def mode_massive(dir1, dir2):
    fdc_arr = init_fdc_arr(dir1, dir2)

    dir_cnt = len(fdc_arr)
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
            samples_find_deep=True,
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

def main():
    args = parse_args()
    log_fn(f"DDIM-FID ==================================================")
    log_fn(f"args: {args}")

    dir1 = args.input1
    dir2 = args.input2
    log_fn(f"dir1   : {dir1}")
    log_fn(f"dir2   : {dir2}")
    if args.mode == 'simple':
        metrics_dict = torch_fidelity.calculate_metrics(
            input1=dir1,
            input2=dir2,
            cuda=True,
            isc=False,
            fid=True,
            kid=False,
            verbose=False,
            samples_find_deep=True,
        )
        log_fn(f"FID: {metrics_dict['frechet_inception_distance']:.6f}")
        # log_fn(f"ISC mean: {metrics_dict['inception_score_mean']:.6f}")
        # log_fn(f"ISC std : {metrics_dict['inception_score_std']:.6f}")

    else:
        mode_massive(dir1, dir2)

    return 0

# ***************************************************************************** miscellaneous
def calc_mean_variance():
    args = parse_args()
    log_fn(f"DDIM-Mean-Variance ==================================================")
    log_fn(f"args: {args}")

    # cifar10
    # [2022-11-12 12:05:55.136] data_dir: ./exp/datasets/cifar10/
    # [2022-11-12 12:05:55.136] batch_size: 200; batch_cnt: 250
    # [2022-11-12 12:05:55.136] mean_avg: 0.491400, 0.482159, 0.446531
    # [2022-11-12 12:05:55.136] var_avg : 0.060943, 0.059208, 0.068305
    # [2022-11-12 12:05:55.137] std_avg : 0.246866, 0.243327, 0.261352
    # FFHQ32x32_train
    # [2022-11-12 12:05:15.631] data_dir: ../vq-vae-2-python/image_dataset/FFHQ32x32_train
    # [2022-11-12 12:05:15.631] batch_size: 200; batch_cnt: 300
    # [2022-11-12 12:05:15.632] mean_avg: 0.520456, 0.425227, 0.380257
    # [2022-11-12 12:05:15.632] var_avg : 0.070539, 0.057563, 0.058208
    # [2022-11-12 12:05:15.632] std_avg : 0.265593, 0.239924, 0.241263
    def gen_dataset():
        data_dir = args.data_dir
        # mean = [0.520456, 0.425227, 0.380257]
        # std = [0.265593, 0.239924, 0.241263]
        # tf = TF.Compose([TF.ToTensor(), TF.Normalize(mean, std)])
        tf = TF.Compose([TF.ToTensor()])
        if 'cifar10' in data_dir:
            # ./exp/datasets/cifar10/
            ds = CIFAR10(data_dir, train=True, download=True, transform=tf)
        elif 'FFHQ' in data_dir or 'ffhq' in data_dir:
            # ../vq-vae-2-python/image_dataset/FFHQ32x32_train
            ds = ImageFolder(root=args.data_dir, transform=tf)
        else:
            ds = ImageFolder(root=args.data_dir, transform=tf)
        return ds

    data_set = gen_dataset()
    data_loader = data.DataLoader(
        data_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
    )
    var_sum = torch.zeros(3).to(args.device)
    mean_sum = torch.zeros(3).to(args.device)
    b_cnt = len(data_loader)
    for i, (x, y) in enumerate(data_loader):
        x = x.to(args.device)
        var, mean = torch.var_mean(x, dim=(0, 2, 3))
        var_sum += var
        mean_sum += mean
        log_fn(f"B{i:03d}/{b_cnt}. var:{var[0]:9.6f} {var[1]:9.6f} {var[2]:9.6f};"
               f" mean: {mean[0]:9.6f} {mean[1]:9.6f} {mean[2]:9.6f}")
    mean_avg = mean_sum / b_cnt
    var_avg = var_sum / b_cnt
    std_avg = var_avg.pow(0.5)
    log_fn(f"data_dir: {args.data_dir}")
    log_fn(f"batch_size: {args.batch_size}; batch_cnt: {b_cnt}")
    log_fn(f"mean_avg:{mean_avg[0]:9.6f},{mean_avg[1]:9.6f},{mean_avg[2]:9.6f}")
    log_fn(f"var_avg :{var_avg[0]:9.6f},{var_avg[1]:9.6f},{var_avg[2]:9.6f}")
    log_fn(f"std_avg :{std_avg[0]:9.6f},{std_avg[1]:9.6f},{std_avg[2]:9.6f}")
    # for
    return 0

# ***************************************************************************** miscellaneous
def aggregate_files():
    """"""
    import os
    import random
    import torchvision.transforms as transforms
    import torchvision.utils as tvu
    from PIL import Image

    def iterate(folder, f_arr):
        for item in os.listdir(folder):
            path = os.path.join(folder, item)
            if os.path.isdir(path):
                iterate(path, f_arr)
            else:
                f_arr.append(path)
        # for
    # def end
    dir_root = './exp/datasets/lsun/bedroom'
    dir_arr = []
    iterate(dir_root, dir_arr)
    f_cnt = len(dir_arr)
    log_fn(f"dir_root   : {dir_root}")
    log_fn(f"found files: {f_cnt}")
    idx_list = list(range(f_cnt))
    random.shuffle(idx_list)

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(), ])

    cfg_list = [
        ['./exp/datasets/lsun/bedroom_val', 10000,
         './exp/datasets/lsun/bedroom_val_10k.ini'],
        ['./exp/datasets/lsun/bedroom_train', 50000,
         './exp/datasets/lsun/bedroom_train_50k.ini'],
        ['./exp/datasets/lsun/bedroom2_train', 50000,
         './exp/datasets/lsun/bedroom2_train_50k.ini'],
        ['./exp/datasets/lsun/bedroom3_train', 50000,
         './exp/datasets/lsun/bedroom3_train_50k.ini'],
        ['./exp/datasets/lsun/bedroom_test', 10000,
         './exp/datasets/lsun/bedroom_test_10k.ini'],
        ['./exp/datasets/lsun/bedroom_01k', 1000,
         './exp/datasets/lsun/bedroom_01k.ini'],
    ]
    start_idx = 0
    for dir_target, cnt, log_file in cfg_list:
        log_fn(f"to generate:{cnt}, start_idx:{start_idx}, dir_target:{dir_target} *************")
        if not os.path.exists(dir_target):
            log_fn(f"os.makedirs({dir_target})")
            os.makedirs(dir_target)
        with open(log_file, 'w') as f_ptr:
            for i in range(cnt):
                idx = idx_list[i+start_idx]
                old_f_path = dir_arr[idx]
                new_f_name = f"{i:05d}.png"
                new_f_path = os.path.join(dir_target, new_f_name)
                msg = f"{new_f_name} <= {idx:6d} {old_f_path}"
                f_ptr.write(msg+"\n")
                if i % 200 == 0 or i == cnt - 1:
                    log_fn(msg)
                image = Image.open(old_f_path).convert("RGB")
                image = transform(image)
                tvu.save_image(image, new_f_path)
            # for
        # with
        start_idx += cnt
    # for

# ***************************************************************************** miscellaneous
def resize_images():
    """"""
    import os
    import torchvision.transforms as transforms
    import torchvision.utils as tvu
    from PIL import Image
    dir_in  = './exp/datasets/celeba/celeba/img_align_celeba_ori'
    dir_out = './exp/datasets/celeba/celeba/img_align_celeba'
    fname_lst = os.listdir(dir_in)
    fname_lst.sort()
    f_cnt = len(fname_lst)
    log_fn(f"dir_in     : {dir_in}")
    log_fn(f"dir_out    : {dir_out}")
    log_fn(f"found files: {f_cnt}")

    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(), ])
    if not os.path.exists(dir_out):
        log_fn(f"os.makedirs({dir_out})")
        os.makedirs(dir_out)
    for idx, fname in enumerate(fname_lst):
        fpath_in  = os.path.join(dir_in, fname)
        nm, ext = os.path.splitext(fname)
        fpath_out = os.path.join(dir_out, f"{nm}.png")
        if idx % 1000 == 0 or idx + 1 == f_cnt:
            log_fn(f"{fpath_in} -> {fpath_out}")
        image = Image.open(fpath_in).convert("RGB")
        image = transform(image)
        tvu.save_image(image, fpath_out)
    # for

# ***************************************************************************** miscellaneous
def extract_model_ema_pure():
    from models.diffusion import Model
    from models.ema import EMAHelper
    from utils import dict2namespace
    import yaml
    path_in  = "./exp/ema-celeba-own-conti-E1000.ckpt"
    path_out = "./exp/ema-celeba-ownpure-conti-E1000.ckpt"
    path_cfg = "./configs/celeba.yml"
    log_fn(f"path_in : {path_in}")
    log_fn(f"path_out: {path_out}")
    log_fn(f"path_cfg: {path_cfg}")
    with open(path_cfg, "r") as f:
        config = yaml.safe_load(f)
    config = dict2namespace(config)
    states = torch.load(path_in, map_location='cuda:1')
    model = Model(config)
    model.load_state_dict(states['model'], strict=True)

    log_fn(f"ema_helper: EMAHelper()")
    ema_helper = EMAHelper()
    ema_helper.register(model)
    log_fn(f"ema_helper: load from states[ema_helper]")
    ema_helper.load_state_dict(states["ema_helper"])
    log_fn(f"ema_helper: apply to model {type(model).__name__}")
    ema_helper.ema(model)

    log_fn(f"save: {path_out}")
    torch.save(model.state_dict(), path_out)

if __name__ == "__main__":
    main()
