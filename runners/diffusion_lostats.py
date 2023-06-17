import logging
import os
import time
import torch
import utils
import torch.utils.data as data
import torchvision.utils as tvu

from torch.backends import cudnn
from datasets import get_dataset, data_transform, inverse_data_transform
from functions.losses import x0_estimation_loss2
from models.diffusion import Model
from runners.diffusion import Diffusion
from utils import count_parameters

class DiffusionLostats(Diffusion):
    """Loss Stats"""
    def __init__(self, args, config, device=None):
        super().__init__(args, config, device)

    def get_data_loaders(self):
        args, config = self.args, self.config
        dataset, test_dataset = get_dataset(args, config)
        batch_size = args.batch_size or config.training.batch_size
        train_loader = data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=config.data.num_workers,
        )
        test_loader = data.DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=config.data.num_workers,
        )
        logging.info(f"train data:")
        logging.info(f"  root       : {dataset.root}")
        logging.info(f"  split      : {dataset.split}") if hasattr(dataset, 'split') else None
        logging.info(f"  len        : {len(dataset)}")
        logging.info(f"  batch_cnt  : {len(train_loader)}")
        logging.info(f"  batch_size : {batch_size}")
        logging.info(f"  shuffle    : True")
        logging.info(f"  num_workers: {config.data.num_workers}")

        logging.info(f"test data:")
        logging.info(f"  root          : {test_dataset.root}")
        logging.info(f"  len           : {len(test_dataset)}")
        logging.info(f"  test_data_dir : {args.test_data_dir}")
        logging.info(f"  batch_cnt     : {len(test_loader)}")
        logging.info(f"  batch_size    : {batch_size}")
        logging.info(f"  shuffle       : False")
        logging.info(f"  num_workers   : {config.data.num_workers}")
        return train_loader, test_loader

    def get_model(self):
        args, config = self.args, self.config
        in_ch = args.model_in_channels
        out_ch = args.model_in_channels
        resolution = args.data_resolution
        model = Model(config, in_channels=in_ch, out_channels=out_ch, resolution=resolution)
        cnt, str_cnt = count_parameters(model, log_fn=None)
        model_name = type(model).__name__
        logging.info(f"model: {model_name} ===================")
        logging.info(f"  config type: {config.model.type}")
        logging.info(f"  param size : {cnt} => {str_cnt}")
        logging.info(f"  ema_flag   : {self.args.ema_flag}")
        logging.info(f"  lr         : {self.args.lr}")
        logging.info(f"  model.to({self.device})")
        model.to(self.device)

        ckpt_path = self.args.sample_ckpt_path
        logging.info(f"  load model: {ckpt_path}")
        states = torch.load(ckpt_path, map_location=self.device)
        if self.args.todo == 'lostats:x0':
            if 'model_x0' in states:
                model.load_state_dict(states['model_x0'])
                logging.info(f"  load x0 model from states")
            else:
                model.load_state_dict(states)
                logging.info(f"  load pure model as states")
        else:
            if 'model' in states:
                model.load_state_dict(states['model'])
                logging.info(f"  load et model from states")
            else:
                model.load_state_dict(states)
                logging.info(f"  load pure model as states")

        if len(args.gpu_ids) > 1:
            logging.info(f"  torch.nn.DataParallel(device_ids={args.gpu_ids})")
            model = torch.nn.DataParallel(model, device_ids=args.gpu_ids)
        cudnn.benchmark = True
        return model

    def run(self):
        """Focus on MSE of epsilon"""
        train_loader, test_loader = self.get_data_loaders()
        model = self.get_model()
        model.eval()

        jump = self.args.timesteps
        h_cnt = 1000 // jump        # hit count
        b_cnt = len(train_loader)   # batch count
        hb_cnt = h_cnt * b_cnt      # hit * batch
        b_sz = self.args.batch_size or self.config.training.batch_size
        dmu_ttl_arr = [0.] * 1000  # mean
        var_ttl_arr = [0.] * 1000  # variance
        mse_ttl_arr = [0.] * 1000  # mse
        dmu_avg_arr = [0.] * 1000
        var_avg_arr = [0.] * 1000
        mse_avg_arr = [0.] * 1000
        cal_cnt_arr = [0] * 1000   # calculation count
        data_start = time.time()
        logging.info(f"h_cnt  : {h_cnt}")
        logging.info(f"jump   : {jump}")
        logging.info(f"b_cnt  : {b_cnt}")
        logging.info(f"b_sz   : {b_sz}")
        logging.info(f"hb_cnt : {hb_cnt}")
        with torch.no_grad():
            for b_idx, (x, y) in enumerate(train_loader):
                x = x.to(self.device)
                x = data_transform(self.config, x)
                b_sz = x.size(0)
                for ts in range(0, 1000, jump):
                    t = torch.ones((b_sz,), dtype=torch.int, device=self.device)
                    t *= ts
                    e = torch.randn_like(x)
                    if self.args.todo == 'lostats:x0':
                        mse, _ = x0_estimation_loss2(model, x, t, e, self.alphas_cumprod)
                    else:
                        at = self.alphas_cumprod.index_select(0, t).view(-1, 1, 1, 1)  # alpha_t
                        xt = x * at.sqrt() + e * (1.0 - at).sqrt()
                        output = model(xt, t.float())
                        mse = (e - output).square().sum(dim=(1, 2, 3)).mean(dim=0)
                        var, mean = torch.var_mean(e - output)
                        dmu_ttl_arr[ts] += mean.item()
                        var_ttl_arr[ts] += var.item() * 3072
                    mse_ttl_arr[ts] += mse.item()
                    cal_cnt_arr[ts] += 1

                    ts_idx = ts // jump
                    if ts_idx % 10 == 0 or ts_idx == h_cnt - 1:
                        elp, eta = utils.get_time_ttl_and_eta(data_start, b_idx * h_cnt + ts_idx, hb_cnt)
                        logging.info(f"B{b_idx:03d}/{b_cnt}.ts{ts:03d} mse:{mse:10.5f}; elp:{elp}, eta:{eta}")
                # for ts
                for i in range(len(var_avg_arr)):
                    if cal_cnt_arr[i] > 0:
                        var_avg_arr[i] = var_ttl_arr[i] / cal_cnt_arr[i]
                        dmu_avg_arr[i] = dmu_ttl_arr[i] / cal_cnt_arr[i]
                        mse_avg_arr[i] = mse_ttl_arr[i] / cal_cnt_arr[i]
                msg = f"B{b_idx:03d}/{b_cnt}"
                utils.save_list(var_avg_arr, '', "./res_eps_var_arr.txt", msg, "{:10.5f}")
                utils.save_list(dmu_avg_arr, '', "./res_eps_dmu_arr.txt", msg, "{:10.5f}")
                utils.save_list(mse_avg_arr, '', "./res_eps_mse_arr.txt", msg, "{:10.5f}")
                utils.save_list(cal_cnt_arr, '', "./res_eps_cnt_arr.txt", msg, "{:4.0f}")
            # for loader
        # with
        # for
        # utils.output_list(var_ttl_arr, 'loss_ttl')
        # utils.output_list(cal_cnt_arr, 'loss_cnt')
        # utils.output_list(var_avg_arr, 'loss_avg')
    # run(self)

    def run_xt(self):
        """
        Focus on MSE of xt.
        Conclusion: the real x_{t-1} and the predicted x_{t-1} do not match.
        In fact, given x_t and t, we can predict epsilon_t. but seems not able to find the real x_{t-1}.
        """
        _, test_loader = self.get_data_loaders()
        model = self.get_model()
        model.eval()

        jump = self.args.timesteps
        h_cnt = 1000 // jump        # hit count
        b_cnt = len(test_loader)    # batch count
        hb_cnt = h_cnt * b_cnt      # hit * batch
        b_sz = self.args.batch_size or self.config.training.batch_size
        dmu_ttl_arr = [0.] * 1000  # delta mu
        var_ttl_arr = [0.] * 1000  # variance
        mse_ttl_arr = [0.] * 1000  # mse
        dmu_avg_arr = [0.] * 1000
        var_avg_arr = [0.] * 1000
        mse_avg_arr = [0.] * 1000
        cal_cnt_arr = [0] * 1000   # calculation count
        data_start = time.time()
        logging.info(f"h_cnt  : {h_cnt}")
        logging.info(f"jump   : {jump}")
        logging.info(f"b_cnt  : {b_cnt}")
        logging.info(f"b_sz   : {b_sz}")
        logging.info(f"hb_cnt : {hb_cnt}")
        with torch.no_grad():
            for b_idx, (x, y) in enumerate(test_loader):
                logging.info(f"Batch:{b_idx:03d}/{b_cnt} **************************************")
                x = x.to(self.device)
                x = data_transform(self.config, x)
                b_sz = x.size(0)
                self.save_images(self.config, './img_ori', x, b_idx, b_sz)
                ab_T = self.alphas_cumprod[-1]  # alpha_bar_T
                eps = torch.randn_like(x)
                x_t = x * ab_T.sqrt() + eps * (1 - ab_T).sqrt()
                for ts in reversed(range(0, 1000, jump)):
                    t = torch.ones((b_sz,), dtype=torch.int, device=self.device)
                    t *= ts
                    output = model(x_t, t.float())
                    ab_t = self.alphas_cumprod.index_select(0, t.long())  # alpha_bar_t
                    ab_s = self.alphas_cumproq.index_select(0, t.long())  # alpha_bar_{t-1}
                    ab_t4d = ab_t.view(-1, 1, 1, 1)
                    ab_s4d = ab_s.view(-1, 1, 1, 1)
                    a_t4d = ab_t4d / ab_s4d  # alpha_t
                    xt_next = (x_t - (1 - ab_t4d).sqrt() * output) / a_t4d.sqrt() + (1 - ab_s4d).sqrt() * output
                    x_t = xt_next

                    real_xt = x * ab_s4d.sqrt() + eps * (1 - ab_s4d).sqrt()
                    mse = (real_xt - x_t).square().sum(dim=(1, 2, 3)).mean(dim=0)
                    var, mean = torch.var_mean(real_xt - x_t)
                    dmu_ttl_arr[ts] += mean.item()
                    var_ttl_arr[ts] += var.item() * 3072
                    mse_ttl_arr[ts] += mse.item()
                    cal_cnt_arr[ts] += 1

                    ts_idx = ts // jump
                    if ts_idx % 10 == 0 or ts_idx == h_cnt - 1:
                        elp, eta = utils.get_time_ttl_and_eta(data_start, b_idx * h_cnt + ts_idx, hb_cnt)
                        logging.info(f"B{b_idx:03d}/{b_cnt}.ts{ts:03d} mse:{mse:12.7f}; elp:{elp}, eta:{eta}")
                # for ts
                self.save_images(self.config, './img_gen', x_t, b_idx, b_sz)
                for i in range(len(var_avg_arr)):
                    if cal_cnt_arr[i] > 0:
                        var_avg_arr[i] = var_ttl_arr[i] / cal_cnt_arr[i]
                        dmu_avg_arr[i] = dmu_ttl_arr[i] / cal_cnt_arr[i]
                        mse_avg_arr[i] = mse_ttl_arr[i] / cal_cnt_arr[i]
                msg = f"B{b_idx:03d}/{b_cnt}"
                utils.save_list(var_avg_arr, '', "./res_xt_var_arr.txt", msg, "{:10.5f}")
                utils.save_list(dmu_avg_arr, '', "./res_xt_dmu_arr.txt", msg, "{:10.5f}")
                utils.save_list(mse_avg_arr, '', "./res_xt_mse_arr.txt", msg, "{:10.5f}")
                utils.save_list(cal_cnt_arr, '', "./res_xt_cnt_arr.txt", msg, "{:4.0f}")
            # for loader
        # with
        # utils.output_list(var_ttl_arr, 'loss_ttl')
        # utils.output_list(cal_cnt_arr, 'loss_cnt')
        # utils.output_list(var_avg_arr, 'loss_avg')
    # run_xt(self)

    def run_delta(self):
        """Focus on delta between predicted noise and ground truth noise"""
        train_loader, test_loader = self.get_data_loaders()
        model = self.get_model()
        model.eval()

        ts_list = [0, 99, 199, 299, 399, 499, 599, 699, 799, 899, 999]
        b_cnt = len(train_loader)   # batch count
        b_sz = self.args.batch_size or self.config.training.batch_size
        logging.info(f"b_cnt  : {b_cnt}")
        logging.info(f"b_sz   : {b_sz}")
        with torch.no_grad():
            for ts in ts_list:
                delta_list = []
                for b_idx, (x, y) in enumerate(train_loader):
                    x = x.to(self.device)
                    x = data_transform(self.config, x)
                    b_sz = x.size(0)
                    t = torch.ones((b_sz,), dtype=torch.int, device=self.device)
                    t *= ts
                    e = torch.randn_like(x)
                    at = self.alphas_cumprod.index_select(0, t).view(-1, 1, 1, 1)  # alpha_t
                    xt = x * at.sqrt() + e * (1.0 - at).sqrt()
                    output = model(xt, t.float())
                    delta = output - e
                    delta = delta.view(b_sz, -1)  # [batch_size, c*h*w]
                    for i in range(b_sz):
                        delta_list.append(delta[i][2000])

                    logging.info(f"B{b_idx:03d}/{b_cnt}.ts{ts:03d}")
                # for loader
                f_path = f"./output0_lostats/ts{ts:03d}.txt"
                logging.info(f"write {len(delta_list)} lines to {f_path}")
                with open(f_path, 'w') as fptr:
                    [fptr.write(f"{d:10.7f}\r\n") for d in delta_list]
                # with
            # for ts
        # with
    # run_delta(self)

    @staticmethod
    def save_images(config, img_dir, x, r_idx, b_sz):
        if not os.path.exists(img_dir):
            logging.info(f"os.makedirs({img_dir})")
            os.makedirs(img_dir)
        x = inverse_data_transform(config, x)
        img_cnt = len(x)
        img_path = None
        for i in range(img_cnt):
            img_id = r_idx * b_sz + i
            img_path = os.path.join(img_dir, f"{img_id:05d}.png")
            tvu.save_image(x[i], img_path)
        logging.info(f"Saved {img_cnt} images: {img_path}")

    def lostats_sample_image(self, x_T, model):
        seq = range(0, self.num_timesteps, self.args.timesteps)
        msg = f"DiffusionLostats::seq=[{seq[-1]}~{seq[0]}], len={len(seq)}"
        b_sz = len(x_T)
        xt = x_T
        for i in reversed(seq):
            t = (torch.ones(b_sz) * i).to(self.device)   # [999., 999.]
            ab_t = self.alphas_cumprod.index_select(0, t.long()) # alpha_bar_t
            ab_s = self.alphas_cumproq.index_select(0, t.long()) # alpha_bar_{t-1}
            e_t = model(xt, ab_t)  # epsilon_t
            if i % 50 == 0: logging.info(f"lostats_sample_image(): {msg}; i={i}")
            # simplified version of the formula.
            ab_t4d = ab_t.view(-1, 1, 1, 1)
            ab_s4d = ab_s.view(-1, 1, 1, 1)
            a_t4d = ab_t4d / ab_s4d  # alpha_t
            xt_next = (xt - (1 - ab_t4d).sqrt() * e_t) / a_t4d.sqrt() + (1 - ab_s4d).sqrt() * e_t
            xt = xt_next
        # for
        return xt

# class
