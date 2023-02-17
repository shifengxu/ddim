# Calculate the mean and variance of a dataset
import argparse
import torch
import torchvision.transforms as TF
import torch.utils.data as data
from torchvision.datasets import CIFAR10, ImageFolder
import utils

log_fn = utils.log_info

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_ids', nargs='+', type=int, default=[7])
    parser.add_argument('--batch_size', type=int, default=200)
    # parser.add_argument('--data_dir', type=str, default="../vq-vae-2-python/image_dataset/FFHQ32x32_train")
    parser.add_argument('--data_dir', type=str, default="./exp/datasets/cifar10/")
    args = parser.parse_args()
    gpu_ids = args.gpu_ids
    device = torch.device(f"cuda:{gpu_ids[0]}") if torch.cuda.is_available() and gpu_ids else torch.device("cpu")
    args.device = device
    return args

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
def gen_dataset(args):
    data_dir = args.data_dir
    mean = [0.520456, 0.425227, 0.380257]
    std  = [0.265593, 0.239924, 0.241263]
    # tf = TF.Compose([TF.ToTensor(), TF.Normalize(mean, std)])
    tf = TF.Compose([TF.ToTensor()])
    if 'cifar10' in data_dir:
        # ./exp/datasets/cifar10/
        data_set = CIFAR10(data_dir, train=True, download=True, transform=tf)
    elif 'FFHQ' in data_dir or 'ffhq' in data_dir:
        # ../vq-vae-2-python/image_dataset/FFHQ32x32_train
        data_set = ImageFolder(root=args.data_dir, transform=tf)
    else:
        data_set = ImageFolder(root=args.data_dir, transform=tf)
    return data_set

def main():
    args = parse_args()
    log_fn(f"DDIM-Mean-Variance ==================================================")
    log_fn(f"args: {args}")

    data_set = gen_dataset(args)
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

if __name__ == "__main__":
    # main()
    aggregate_files()
