"""
Kullback Leibler divergence
"""

import os
import time
import torch
import torch.utils.data as data
import torchvision.utils as tvu
import torchvision.transforms as T

from datasets import get_dataset, ImageDataset
import utils
from models.ema import EMAHelper
from utils import log_info
from models.diffusion import Model

class KullbackLeiblerDivergence:
    def __init__(self, args, config):
        self.args = args
        self.config = config
        self.device = args.device
        self.model = None
        log_info(f"KullbackLeiblerDivergence()")
        log_info(f"  device   : {self.device}")

    def save_image_gt_x0(self):
        """
        Save image: ground truth of x0
        :return:
        """
        args = self.args
        train_loader, test_loader = self._get_data_loaders() # data loaders
        b_cnt = len(train_loader)
        b_sz = args.batch_size
        img_dir = args.sample_output_dir
        img_dir += "0.000"
        if not os.path.exists(img_dir):
            log_info(f"os.makedirs({img_dir})")
            os.makedirs(img_dir)
        time_start = time.time()
        log_info(f"KullbackLeiblerDivergence::save_image_gt_x0()")
        log_info(f"  img_dir : {img_dir}")
        log_info(f"  b_cnt   : {b_cnt}")
        log_info(f"  b_sz    : {b_sz}")
        for b_idx, (x, y) in enumerate(train_loader):
            x = x.to(self.device)
            last_img_path = self._save_image_batch(x, b_idx, b_sz, img_dir, False, y=y)
            elp, eta = utils.get_time_ttl_and_eta(time_start, b_idx, b_cnt)
            log_info(f"B{b_idx:03d}/{b_cnt}. {last_img_path} elp:{elp}, eta:{eta}.")
        # for

    def save_image_gt_xt(self, t):
        """
        save image of ground truth, for x_t
        :param t: t in [0, 1]
        :return:
        """
        args, config = self.args, self.config
        train_loader, test_loader = self._get_data_loaders() # data loaders
        b_cnt = len(train_loader)
        b_sz = args.batch_size
        img_dir = args.sample_output_dir
        img_dir += f"{t:.3f}"
        if not os.path.exists(img_dir):
            log_info(f"os.makedirs({img_dir})")
            os.makedirs(img_dir)
        log_info(f"KullbackLeiblerDivergence::save_image_gt_xt()")
        log_info(f"  t      : {t}")
        log_info(f"  img_dir: {img_dir}")
        log_info(f"  b_cnt  : {b_cnt}")
        log_info(f"  b_sz   : {b_sz}")
        log_info(f"  seed   : {args.seed}")
        time_start = time.time()
        for b_idx, (x0, y) in enumerate(train_loader):
            x0 = x0.to(self.device)
            x0 = x0 * 2 - 1
            epsilon = torch.randn_like(x0)
            x_t = x0 * (1 - t) + epsilon * t
            last_img_path = self._save_image_batch(x_t, b_idx, b_sz, img_dir, True)
            elp, eta = utils.get_time_ttl_and_eta(time_start, b_idx, b_cnt)
            log_info(f"B{b_idx:03d}/{b_cnt}. {last_img_path} elp:{elp}, eta:{eta}.")

    def predict_save_image(self, t_a, t_b):
        """
        predict x_b from x_a; and save x_b
        1 >= t_a > t_b >= 0
        :param t_a: timestep a
        :param t_b: timestep b
        :return:
        """
        self._load_model()
        img_dir = self.args.sample_output_dir
        img_dir += f"{t_a:.3f}-{t_b:.3f}"
        if not os.path.exists(img_dir):
            log_info(f"os.makedirs({img_dir})")
            os.makedirs(img_dir)
        train_loader, test_loader = self._get_data_loaders()  # data loaders
        b_sz = self.args.batch_size
        b_cnt = len(train_loader)
        log_info(f"KullbackLeiblerDivergence::predict_save_image()...")
        log_info(f"  img_dir           : {img_dir}")
        log_info(f"  t_a               : {t_a}")
        log_info(f"  t_b               : {t_b}")
        log_info(f"  batch_size        : {b_sz}")
        log_info(f"  batch_count       : {b_cnt}")
        time_start = time.time()
        with torch.no_grad():
            for b_idx, (x0, y) in enumerate(train_loader):
                x0 = x0.to(self.device)
                x0 = x0 * 2 - 1
                epsilon = torch.randn_like(x0, device=self.device)
                xa = t_a * epsilon + (1 - t_a) * x0
                t_a_expand = torch.ones(x0.size(0), device=x0.device) * t_a * 1000
                grad = self.model(xa, t_a_expand)
                delta = t_a - t_b
                delta = (torch.ones(1, device=self.device) * delta).view(-1, 1, 1, 1)
                pred_xb = xa - grad * delta
                last_img_path = self._save_image_batch(pred_xb, b_idx, b_sz, img_dir, True)
                elp, eta = utils.get_time_ttl_and_eta(time_start, b_idx, b_cnt)
                log_info(f"B{b_idx:03d}/{b_cnt}. {last_img_path} elp:{elp}, eta:{eta}.")
            # for b_idx
        # with

    def predict_seq_save_image(self, ts_arr):
        """
        predict by sequence.
        Assume ta, tb, tc = ts_arr
        1 >= ta > tb > tc >= 0
        :return:
        """
        self._load_model()
        img_dir = self.args.sample_output_dir
        img_dir += '-'.join([f"{ts:.3f}" for ts in ts_arr])
        if not os.path.exists(img_dir):
            log_info(f"os.makedirs({img_dir})")
            os.makedirs(img_dir)
        train_loader, test_loader = self._get_data_loaders()  # data loaders
        b_sz = self.args.batch_size
        b_cnt = len(train_loader)
        log_info(f"KullbackLeiblerDivergence::predict_seq_save_image()...")
        log_info(f"  img_dir           : {img_dir}")
        log_info(f"  ts_arr            : {ts_arr}")
        log_info(f"  batch_size        : {b_sz}")
        log_info(f"  batch_count       : {b_cnt}")
        ts_cnt = len(ts_arr)
        with torch.no_grad():
            time_start = time.time()
            for b_idx, (x0, y) in enumerate(train_loader):
                x0 = x0.to(self.device)
                x0 = x0 * 2 - 1
                epsilon = torch.randn_like(x0, device=self.device)
                ta = ts_arr[0]
                xa = ta * epsilon + (1 - ta) * x0
                for idx in range(1, ts_cnt):
                    tb = ts_arr[idx]
                    ta_expand = torch.ones(x0.size(0), device=x0.device) * ta * 1000
                    ga = self.model(xa, ta_expand)
                    delta = ta - tb
                    if b_idx == 0:
                        log_info(f"ta:{ta:.3f}, tb:{tb:.3f}, delta:{delta:.3f}")
                    # delta = (torch.ones(1, device=self.device) * delta).view(-1, 1, 1, 1)
                    pred_xb = xa - ga * delta
                    xa = pred_xb
                    ta = tb
                # for
                last_img_path = self._save_image_batch(xa, b_idx, b_sz, img_dir, True)
                elp, eta = utils.get_time_ttl_and_eta(time_start, b_idx, b_cnt)
                log_info(f"B{b_idx:03d}/{b_cnt}. {last_img_path} elp:{elp}, eta:{eta}.")
            # for b_idx
        # with

    def calc_mean_var(self, image_dir=None, image_size=32, num_workers=4):
        args = self.args
        image_dir = image_dir or args.data_dir
        batch_size = args.batch_size
        log_info(f"KullbackLeiblerDivergence::calc_mean_var()")
        log_info(f"  image_dir  : {image_dir}")
        log_info(f"  image_size : {image_size}")
        log_info(f"  num_workers: {num_workers}")
        log_info(f"  batch_size : {batch_size}")
        tf = T.Compose([T.Resize(image_size), T.ToTensor()])
        ds = ImageDataset(image_dir, classes=None, transform=tf)
        dl = data.DataLoader(ds, batch_size, shuffle=False, num_workers=num_workers)
        b_cnt = len(dl)
        mean_sum = None
        time_start = time.time()
        for b_idx, (x, y) in enumerate(dl):
            x = x.to(self.device)
            mean = x.mean(dim=0)
            if mean_sum is None:
                mean_sum = mean
            else:
                mean_sum += mean
            elp, eta = utils.get_time_ttl_and_eta(time_start, b_idx, b_cnt)
            log_info(f"R1B{b_idx:03d}/{b_cnt}. elp:{elp}, eta:{eta}.")
        # for
        mean = mean_sum / b_cnt
        mean_batch = mean.unsqueeze(0)
        # so far we got the global mean value
        var_sum = None
        for b_idx, (x, y) in enumerate(dl):
            x = x.to(self.device)
            x -= mean_batch
            var = torch.square(x)
            var = var.mean(dim=0)
            if var_sum is None:
                var_sum = var
            else:
                var_sum += var
            elp, eta = utils.get_time_ttl_and_eta(time_start, b_idx, b_cnt)
            log_info(f"R2B{b_idx:03d}/{b_cnt}. elp:{elp}, eta:{eta}.")
        # for
        var = var_sum / b_cnt
        # so fare we got mean and variance, with the same dimensions as single image
        log_info(f"  image_dir  : {image_dir}")
        log_info(f"  image_size : {image_size}")
        log_info(f"  mean_dim   : {list(mean.shape)}")
        log_info(f"  var_dim    : {list(var.shape)}")
        log_info(f"  mean_avg   : {mean.mean():.8f}")
        log_info(f"  var_avg    : {var.mean():.8f}")
        return mean, var

    def calc_kl_div(self, image_dir1, image_dir2, image_size=32, num_workers=4):
        """
        calculate KL divergence
        mean and var both have the same dimension as image
        :return:
        """
        mean1, var1 = self.calc_mean_var(image_dir1, image_size, num_workers)
        mean2, var2 = self.calc_mean_var(image_dir2, image_size, num_workers)
        sigma1, sigma2 = var1.sqrt(), var2.sqrt()
        log_info(f"KullbackLeiblerDivergence::calc_kl_div()")
        log_info(f"  image_size : {image_size}")
        log_info(f"  num_workers: {num_workers}")
        log_info(f"  image_dir1 : {image_dir1}")
        log_info(f"    mean1_dim: {list(mean1.shape)}")
        log_info(f"    var1_dim : {list(var1.shape)}")
        log_info(f"    mean1_avg: {mean1.mean():.8f}")
        log_info(f"    var1_avg : {var1.mean():.8f}")
        log_info(f"    sigma1   : {sigma1.mean():.8f}")
        log_info(f"  image_dir2 : {image_dir2}")
        log_info(f"    mean2_dim: {list(mean2.shape)}")
        log_info(f"    var2_dim : {list(var2.shape)}")
        log_info(f"    mean2_avg: {mean2.mean():.8f}")
        log_info(f"    var2_avg : {var2.mean():.8f}")
        log_info(f"    sigma2   : {sigma2.mean():.8f}")


        part1 = torch.log(sigma1 / sigma2)
        part2 = (var2 + (mean2 - mean1).square()) / (2 * var1)
        kl_div1 = part1 + part2 - 0.5
        kl_div1 = kl_div1.mean()
        log_info(f"Div_KL(dir2||dir1) ----------")
        log_info(f"  part1 :{part1.mean():11.8f}")
        log_info(f"  part2 :{part2.mean():11.8f}")
        log_info(f"  KL_div:{kl_div1:11.8f}")

        part1 = torch.log(sigma2 / sigma1)
        part2 = (var1 + (mean1 - mean2).square()) / (2 * var2)
        kl_div2 = part1 + part2 - 0.5
        kl_div2 = kl_div2.mean()
        log_info(f"Div_KL(dir1||dir2) ----------")
        log_info(f"  part1 :{part1.mean():11.8f}")
        log_info(f"  part2 :{part2.mean():11.8f}")
        log_info(f"  KL_div:{kl_div2:11.8f}")
        return (kl_div1 + kl_div2) / 2, mean1, var1, mean2, var2

    def calc_grad_similarity(self):
        args, config = self.args, self.config
        self._load_model()
        train_loader, test_loader = self._get_data_loaders() # data loaders
        data_loader = test_loader
        b_cnt = len(data_loader)
        b_sz = args.batch_size
        ts_arr = list(range(1000, 99, -100))
        ts_cnt = len(ts_arr)
        log_info(f"KullbackLeiblerDivergence::calc_grad_similarity()")
        log_info(f"  ts_arr : {ts_arr}")
        log_info(f"  b_cnt  : {b_cnt}")
        log_info(f"  b_sz   : {b_sz}")
        log_info(f"  seed   : {args.seed}")
        sim_arr = [0.] * ts_cnt
        cs = torch.nn.CosineSimilarity(dim=0)
        time_start = time.time()
        with torch.no_grad():
            for b_idx, (x0, y) in enumerate(data_loader):
                x0 = x0.to(self.device)
                x0 = x0 * 2 - 1
                n = x0.size(0)
                epsilon = torch.randn_like(x0)
                for i in range(1, ts_cnt):
                    ta, tb = ts_arr[i-1], ts_arr[i] # timestep: 1 ~ 1000
                    ca, cb = ta / 1000, tb / 1000   # cursor:   0 ~ 1.0
                    xa = x0 * (1 - ca) + epsilon * ca
                    ta_expand = torch.ones(n, device=self.device) * ta
                    grad_a = self.model(xa, ta_expand)
                    # xb = xa - grad_a * (ca - cb)
                    xb = x0 * (1 - cb) + epsilon * cb
                    tb_expand = torch.ones(n, device=self.device) * tb
                    grad_b = self.model(xb, tb_expand)
                    grad_a1d, grad_b1d = grad_a.view(-1), grad_b.view(-1)
                    sim = cs(grad_a1d, grad_b1d)
                    sim_arr[i] += sim
                    if b_idx == 0:
                        log_info(f"{ta:4d}~{tb:4d}: {sim:.12f}")
                # for
                elp, eta = utils.get_time_ttl_and_eta(time_start, b_idx, b_cnt)
                sim1, sim2 = sim_arr[1] / (b_idx + 1), sim_arr[-1] / (b_idx + 1)
                log_info(f"B{b_idx:03d}/{b_cnt}. sim:{sim1:.8f}~{sim2:.8f}; elp:{elp}, eta:{eta}.")
            # for
        # with
        sim_arr = [s / b_cnt for s in sim_arr]
        for ts, sim in zip(ts_arr, sim_arr):
            log_info(f"{ts:3d}: {sim:.8f}")

    @staticmethod
    def _save_image_batch(x, b_idx, b_sz, img_dir, rescale, y=None):
        if rescale:
            x = (x + 1) / 2
            x = torch.clamp(x, 0, 1)
        img_cnt = len(x)
        img_path = None
        for i in range(img_cnt):
            img_id = b_idx * b_sz + i
            f_name = f"{img_id:05d}_{y[i]}.png" if y else f"{img_id:05d}.png"
            img_path = os.path.join(img_dir, f_name)
            tvu.save_image(x[i], img_path)
        return img_path  # return the path of last saved image

    def _get_data_loaders(self):
        args, config = self.args, self.config
        batch_size = args.batch_size
        dataset, test_dataset = get_dataset(args, config)
        train_loader = data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=config.data.num_workers,
        )
        log_info(f"train dataset and data loader:")
        log_info(f"  root       : {dataset.root}")
        log_info(f"  split      : {dataset.split}") if hasattr(dataset, 'split') else None
        log_info(f"  len        : {len(dataset)}")
        log_info(f"  batch_cnt  : {len(train_loader)}")
        log_info(f"  batch_size : {batch_size}")
        log_info(f"  shuffle    : False")
        log_info(f"  num_workers: {config.data.num_workers}")

        test_loader = data.DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=config.data.num_workers,
        )
        log_info(f"test dataset and loader:")
        log_info(f"  root          : {test_dataset.root}")
        log_info(f"  len           : {len(test_dataset)}")
        log_info(f"  batch_cnt     : {len(test_loader)}")
        log_info(f"  batch_size    : {batch_size}")
        log_info(f"  shuffle       : False")
        log_info(f"  num_workers   : {config.data.num_workers}")
        return train_loader, test_loader

    def _load_model(self):
        """"""
        def apply_ema():
            log_info(f"  ema_helper: EMAHelper()")
            ema_helper = EMAHelper()
            ema_helper.register(model)
            k = "ema_helper"
            log_info(f"  ema_helper: load from states[{k}]")
            ema_helper.load_state_dict(states[k])
            log_info(f"  ema_helper: apply to model {type(model).__name__}")
            ema_helper.ema(model)

        if self.model is not None:
            return
        ckpt_path = self.args.sample_ckpt_path
        log_info(f"load ckpt: {ckpt_path}")
        states = torch.load(ckpt_path, map_location=self.config.device)
        model = Model(self.config)
        model.load_state_dict(states['model'], strict=True)
        if not hasattr(self.args, 'ema_flag'):
            log_info(f"  !!! Not found ema_flag in args. Assume it is true.")
            apply_ema()
        else:
            log_info(f"  Found args.ema_flag: {self.args.ema_flag}.")
            if self.args.ema_flag:
                apply_ema()

        log_info(f"  model = model.to({self.device})")
        model = model.to(self.device)
        model.eval()
        if len(self.args.gpu_ids) > 1:
            log_info(f"  torch.nn.DataParallel(model, device_ids={self.args.gpu_ids})")
            model = torch.nn.DataParallel(model, device_ids=self.args.gpu_ids)
        log_info(f"  self.model = model")
        self.model = model

# class
