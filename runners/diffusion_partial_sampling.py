import logging
import os
import time

import numpy as np
import torch
import torch.utils.data as data
import torchvision.utils as tvu

import utils
from datasets import get_dataset, data_transform, inverse_data_transform
from models.diffusion import ModelStack, Model
from runners.diffusion import Diffusion
from utils import count_parameters, extract_ts_range


"""
 Partial Sampling.
 Usually, we sample from x1000 (aka x_T) to x0. and x0 is clean image, and x1000 is Gaussian Noise.
 But we may want to:
  1, sample from x0, to x200, we call it x200. This is by diffusing;
  2, sample from x1000 to x200, we call it x200'. this is by denoising.
 By doing so, we have chance to compare the x200 and x200'.
"""
class DiffusionPartialSampling(Diffusion):
    def __init__(self, args, config, device=None):
        super().__init__(args, config, device)
        self.sample_count = args.sample_count
        self.sample_img_init_id = args.sample_img_init_id
        self.time_start = None
        betas = torch.cat([torch.zeros(1).to(self.betas.device), self.betas], dim=0)
        self.cumprod_1001 = (1 - betas).cumprod(dim=0)  # has 1001 elements

    def run(self):
        args, config = self.args, self.config
        logging.info(f"psample: {args.psample}")
        logging.info(f"  psample_dir    : {args.psample_dir}")
        logging.info(f"  psample_ts_list: {args.psample_ts_list}")
        if args.psample == 'from_x0':
            return self.gen_xt_from_x0(args, config)
        elif args.psample == 'from_gn':  # from Gaussian Noise
            return self.gen_xt_from_gn()
        else:
            raise NotImplementedError(f"Invalid psample: {args.psample}")

    # ************************************************************************* gen xt from gaussian noise
    def gen_xt_from_gn(self):
        config = self.config
        ckpt_path_arr = self.get_ckpt_path_arr()
        if len(ckpt_path_arr) > 1:
            model = ModelStack(config, len(ckpt_path_arr))
            model = self.model_stack_load_from_local(model, ckpt_path_arr)
        else:
            model = Model(config)
            model = self.model_load_from_local(model)
        model.eval()
        logging.info(f"gen_xt_from_gn(self, {type(model).__name__})...")
        logging.info(f"  args.sample_output_dir : {self.args.sample_output_dir}")
        logging.info(f"  args.sample_type       : {self.args.sample_type}")
        logging.info(f"  args.skip_type         : {self.args.skip_type}")
        logging.info(f"  args.timesteps         : {self.args.timesteps}")
        logging.info(f"  num_timesteps          : {self.num_timesteps}")
        logging.info(f"  sample_count           : {self.sample_count}")
        logging.info(f"  sample_img_init_id     : {self.sample_img_init_id}")
        b_sz = self.args.sample_batch_size or config.sampling.batch_size
        r_cnt = (self.sample_count - 1) // b_sz + 1  # get the ceiling
        logging.info(f"  batch_size             : {b_sz}")
        logging.info(f"  round_cnt              : {r_cnt}")
        img_dir = self.args.psample_dir
        if not os.path.exists(img_dir):
            logging.info(f"mkdir: {img_dir}")
            os.makedirs(img_dir)

        self.time_start = time.time()
        with torch.no_grad():
            for r_idx in range(r_cnt):
                n = b_sz if r_idx + 1 < r_cnt else self.sample_count - r_idx * b_sz
                logging.info(f"round: {r_idx}/{r_cnt}. to generate: {n}."
                             f" init:{self.sample_img_init_id}; cnt:{self.sample_count}")
                x_t = torch.randn(  # normal distribution with mean 0 and variance 1
                    n,
                    config.data.channels,
                    config.data.image_size,
                    config.data.image_size,
                    device=self.device,
                )
                self.sample_image(x_t, model, r_cnt=r_cnt, r_idx=r_idx, b_sz=b_sz)
                real_model = model.module if isinstance(model, torch.nn.DataParallel) else model
                if isinstance(real_model, ModelStack):
                    logging.info(f"ModelStack brick hit counter: {real_model.brick_hit_counter}")
            # for r_idx
        # with

    def get_ckpt_path_arr(self):
        root_dir = self.args.sample_ckpt_dir
        ckpt_path_arr = []
        for fname in os.listdir(root_dir):
            if fname.startswith("ckpt") and fname.endswith(".pth"):
                ckpt_path_arr.append(os.path.join(root_dir, fname))
        ckpt_path_arr.sort()
        return ckpt_path_arr

    def model_load_from_local(self, model):
        if self.args.sample_ckpt_path:
            ckpt_path = self.args.sample_ckpt_path
        elif getattr(self.config.sampling, "ckpt_id", None) is None:
            ckpt_path = os.path.join(self.args.log_path, "ckpt.pth")
        else:
            ckpt_path = os.path.join(self.args.log_path, f"ckpt_{self.config.sampling.ckpt_id}.pth")
        logging.info(f"load ckpt: {ckpt_path}")
        states = torch.load(ckpt_path, map_location=self.config.device)
        if isinstance(states, dict):
            model.load_state_dict(states['model'], strict=True)
        else:
            model.load_state_dict(states[0], strict=True)
        model = model.to(self.device)
        model = torch.nn.DataParallel(model, device_ids=self.args.gpu_ids)
        logging.info(f"model({type(model).__name__})")
        logging.info(f"  model.to({self.device})")
        logging.info(f"  torch.nn.DataParallel(model, device_ids={self.args.gpu_ids})")
        return model

    def model_stack_load_from_local(self, model: ModelStack, ckpt_path_arr):
        cnt, str_cnt = count_parameters(model, log_fn=None)
        logging.info(f"Loading ModelStack(stack_sz: {model.stack_sz})")
        logging.info(f"  config type  : {self.config.model.type}")
        logging.info(f"  param size   : {cnt} => {str_cnt}")
        logging.info(f"  ckpt_path_arr: {len(ckpt_path_arr)}")
        ms = model.model_stack
        for i, ckpt_path in enumerate(ckpt_path_arr):
            model.tsr_stack[i] = extract_ts_range(ckpt_path)
            logging.info(f"  load ckpt {i: 2d} : {ckpt_path}. ts: {model.tsr_stack[i]}")
            states = torch.load(ckpt_path, map_location=self.config.device)
            if isinstance(states, dict):
                # This is for backward-compatibility. As previously, the states is a list ([]).
                # And that was not convenient for adding or dropping items.
                # Therefore, we change to use dict.
                ms[i].load_state_dict(states['model'], strict=True)
            else:
                ms[i].load_state_dict(states[0], strict=True)
        # for
        model = model.to(self.device)
        if len(self.args.gpu_ids) > 0:
            model = torch.nn.DataParallel(model, device_ids=self.args.gpu_ids)
        logging.info(f"  model.to({self.device})")
        logging.info(f"  torch.nn.DataParallel(model, device_ids={self.args.gpu_ids})")
        return model

    def save_xt(self, x, ts, r_idx, b_sz):
        x = x.clone()
        x = inverse_data_transform(self.config, x)
        img_cnt = len(x)
        ts_dir = os.path.join(self.args.psample_dir, f"from_gn_ts_{ts:04d}")
        if not os.path.exists(ts_dir):
            logging.info(f"mkdir: {ts_dir}")
            os.makedirs(ts_dir)
        img_first = img_last = None
        for i in range(img_cnt):
            img_id = r_idx * b_sz + i + self.sample_img_init_id
            img_path = os.path.join(ts_dir, f"{img_id:05d}.png")
            tvu.save_image(x[i], img_path)
            if i == 0: img_first = f"{img_id:05d}.png"
            if i == img_cnt - 1: img_last = f"{img_id:05d}.png"
        logging.info(f"  saved {img_cnt} images to: {ts_dir}/: {img_first} ~ {img_last}.")

    def sample_image(self, x, model, last=True, **kwargs):
        if self.args.sample_type == "generalized":
            if self.args.skip_type == "uniform":
                skip = self.num_timesteps // self.args.timesteps
                seq = range(0, self.num_timesteps, skip)
            elif self.args.skip_type == "quad":
                seq = (np.linspace(0, np.sqrt(self.num_timesteps * 0.8), self.args.timesteps) ** 2)
                seq = [int(s) for s in list(seq)]
            else:
                raise NotImplementedError

            kwargs['eta'] = self.args.eta
            x = self.generalized_steps(x, seq, model, **kwargs)
        elif self.args.sample_type == "ddpm_noisy":
            if self.args.skip_type == "uniform":
                skip = self.num_timesteps // self.args.timesteps
                seq = range(0, self.num_timesteps, skip)
            elif self.args.skip_type == "quad":
                seq = np.linspace(0, np.sqrt(self.num_timesteps * 0.8), self.args.timesteps) ** 2
                seq = [int(s) for s in list(seq)]
            else:
                raise NotImplementedError
            from functions.denoising import ddpm_steps

            x = ddpm_steps(x, seq, model, self.betas)
            if last:
                x = x[0][-1]
        else:
            raise NotImplementedError
        return x

    def generalized_steps(self, x_T, seq, model, **kwargs):
        """
        Original paper: Denoising Diffusion Implicit Models. ICLR. 2021
        :param x_T: x_T in formula; it has initial Gaussian Noise
        :param seq:
        :param model:
        :param kwargs:
        :return:
        """
        eta   = kwargs.get("eta", 0)
        r_cnt = kwargs['r_cnt']
        r_idx = kwargs['r_idx']
        b_sz  = kwargs['b_sz']
        ts_list = self.args.psample_ts_list
        ts_min = min(ts_list)
        if 1000 in ts_list:
            self.save_xt(x_T, 1000, r_idx, b_sz)
        msg = f"seq=[{seq[-1]}~{seq[0]}], len={len(seq)}"
        xt = x_T
        xt = xt.to(x_T.device)
        img_cnt = x_T.size(0)
        seq_next = [-1] + list(seq[:-1])
        for i, j in zip(reversed(seq), reversed(seq_next)):
            if i % 50 == 0:
                itr = 1000 - i
                elp, rmn = utils.get_time_ttl_and_eta(self.time_start, r_idx * 1000 + itr, r_cnt * 1000)
                logging.info(f"generalized_steps(): {msg}; i={i}. round:{r_idx}/{r_cnt}, elp:{elp}, eta:{rmn}")
            if i < ts_min:
                break
            t = (torch.ones(img_cnt) * i).to(x_T.device)  # [999., 999.]
            q = (torch.ones(img_cnt) * j).to(x_T.device)  # [998., 998.]. next t; can assume as t-1
            at = self.cumprod_1001.index_select(0, t.long() + 1).view(-1, 1, 1, 1)  # alpha_t
            aq = self.cumprod_1001.index_select(0, q.long() + 1).view(-1, 1, 1, 1)  # alpha_{t-1}
            et = model(xt, t)  # epsilon_t
            x0 = (xt - et * (1 - at).sqrt()) / at.sqrt()  # formula (9)
            sigma_t = eta * ((1 - at / aq) * (1 - aq) / (1 - at)).sqrt()  # formula (16)
            c2 = (1 - aq - sigma_t ** 2).sqrt()
            xt_next = aq.sqrt() * x0 + c2 * et + sigma_t * torch.randn_like(x_T)  # formula (12)
            xt = xt_next
            if i in ts_list:
                self.save_xt(xt, i, r_idx, b_sz)

        return xt

    # ************************************************************************* gen xt from x0
    def gen_xt_from_x0(self, args, config):
        b_sz = self.args.sample_batch_size or config.sampling.batch_size
        n_wk = config.data.num_workers
        config.data.random_flip = False  # hard code. disable flip.
        dataset, _ = get_dataset(args, config)
        logging.info(f"dataset:")
        logging.info(f"  root : {dataset.root}")
        logging.info(f"  split: {dataset.split}") if hasattr(dataset, 'split') else None
        logging.info(f"  len  : {len(dataset)}")
        logging.info(f"data_loader:")
        logging.info(f"  batch_size : {b_sz}")
        logging.info(f"  num_workers: {n_wk}")
        train_loader = data.DataLoader(dataset, batch_size=b_sz, shuffle=False, num_workers=n_wk)
        img_dir = self.args.psample_dir
        if not os.path.exists(img_dir):
            logging.info(f"mkdir: {img_dir}")
            os.makedirs(img_dir)

        for ts in args.psample_ts_list:
            ts_dir = os.path.join(img_dir, f"from_x0_ts_{ts:04d}")
            if not os.path.exists(ts_dir):
                logging.info(f"mkdir: {ts_dir}")
                os.makedirs(ts_dir)
            for b_idx, (x0, y) in enumerate(train_loader):
                xt = self.calc_xt_by_x0(x0, ts)
                img_first = img_last = None
                img_cnt = len(x0)
                for i in range(img_cnt):
                    img_name = f"{b_idx * b_sz + i:05d}_{y[i]}.png"
                    img_path = os.path.join(ts_dir, img_name)
                    tvu.save_image(xt[i], img_path)
                    if i == 0: img_first = img_name
                    if i == img_cnt - 1: img_last = img_name
                # for image
                logging.info(f"saved {img_cnt} images to: {ts_dir}: {img_first} ~ {img_last}")
            # for batch
        # for ts

    def calc_xt_by_x0(self, x0, ts):
        """
        Get x_t from x_0 and timestep.
        :param x0:
        :param ts: timestep. If 0, means x0 itself.
        :return:
        """
        if ts == 0:
            return x0
        ts -= 1
        img_cnt = len(x0)
        x0 = x0.to(self.device)
        x0 = data_transform(self.config, x0)  # transform to [-1, 1]
        ep = torch.randn_like(x0)
        tt = torch.ones(img_cnt, dtype=torch.int) * ts  # tensor dimension as x0
        tt = tt.to(self.device)
        at = self.alphas_cumprod.index_select(0, tt).view(-1, 1, 1, 1)  # alpha_t
        # self.alphas_cumprod.shape : [1000]
        # self.alphas_cumprod[0]    : 0.9999
        # at.shape                  : [500, 1, 1, 1]
        xt = x0 * at.sqrt() + ep * (1.0 - at).sqrt()
        xt = inverse_data_transform(self.config, xt)  # inverse transform
        return xt
# class
