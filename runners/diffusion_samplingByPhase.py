import os
import time
import logging
import numpy as np
import torch
import utils
import torchvision.utils as tvu
from datasets import inverse_data_transform
from models.diffusion import Model
from models.ema import EMAHelper
from runners.diffusion import Diffusion


class DiffusionSamplingByPhase(Diffusion):
    def __init__(self, args, config, device=None):
        super().__init__(args, config, device)
        self.sample_count = args.sample_count    # sample image count
        self.sample_img_init_id = args.sample_img_init_id
        self.model_et = None  # the model predicting epsilon_t
        self.model_x0 = None  # the model predicting x0
        self.x0_ts_cnt = args.sample_x0_ts_cnt

        skip = self.num_timesteps // self.args.timesteps
        self.seq = range(0, self.num_timesteps, skip)

    def sample(self):
        config = self.config
        args = self.args
        path_et = args.sample_ckpt_path
        path_x0 = args.sample_ckpt_path_x0
        self.model_et = Model(config)
        self.model_x0 = Model(config)
        self.model_et = self.load_from_local(self.model_et, path_et, 'et')
        self.model_x0 = self.load_from_local(self.model_x0, path_x0, 'x0')
        self.model_et.eval()
        self.model_x0.eval()
        logging.info(f"DiffusionSamplingByPhase::sample()")
        logging.info(f"  model_et               : {type(self.model_et).__name__}")
        logging.info(f"  x0_ts_cnt              : {self.x0_ts_cnt}")
        logging.info(f"  args.sample_output_dir : {self.args.sample_output_dir}")
        logging.info(f"  args.sample_type       : {self.args.sample_type}")
        logging.info(f"  args.skip_type         : {self.args.skip_type}")
        logging.info(f"  args.timesteps         : {self.args.timesteps}")
        logging.info(f"  num_timesteps          : {self.num_timesteps}")
        logging.info(f"  sample_count           : {self.sample_count}")
        logging.info(f"  sample_img_init_id     : {self.sample_img_init_id}")
        b_sz = self.args.sample_batch_size or config.sampling.batch_size
        n_rounds = (self.sample_count - 1) // b_sz + 1  # get the ceiling
        logging.info(f"  batch_size             : {b_sz}")
        logging.info(f"  n_rounds               : {n_rounds}")
        logging.info(f"  Generating image samples for FID evaluation")
        time_start = time.time()
        with torch.no_grad():
            for r_idx in range(n_rounds):
                n = b_sz if r_idx + 1 < n_rounds else self.sample_count - r_idx * b_sz
                logging.info(f"round: {r_idx}/{n_rounds}. to generate: {n}")
                x_t = torch.randn(  # normal distribution with mean 0 and variance 1
                    n,
                    config.data.channels,
                    config.data.image_size,
                    config.data.image_size,
                    device=self.device,
                )
                self.sample_fid_vanilla(x_t, config, time_start, n_rounds, r_idx, b_sz)
            # for r_idx
        # with

    def sample_fid_vanilla(self, xt, config, time_start, n_rounds, r_idx, b_sz):
        if self.args.skip_type == "uniform":
            skip = self.num_timesteps // self.args.timesteps
            seq = range(0, self.num_timesteps, skip)
        elif self.args.skip_type == "quad":
            seq = (np.linspace(0, np.sqrt(self.num_timesteps * 0.8), self.args.timesteps) ** 2)
            seq = [int(s) for s in list(seq)]
        else:
            raise NotImplementedError
        seq_x0 = list(seq[:self.x0_ts_cnt])
        seq_et = list(seq[self.x0_ts_cnt:])
        seq_x0.reverse()
        seq_et.reverse()
        xm = self.backward_steps_et(xt, seq_et, eta=self.args.eta)  # x_middle
        x0 = self.backward_steps_x0(xm, seq_x0, eta=self.args.eta)  # x_0

        elp, eta = utils.get_time_ttl_and_eta(time_start, r_idx+1, n_rounds)
        msg = f"round:{r_idx}/{n_rounds}, elp:{elp}, eta:{eta}"
        x = inverse_data_transform(config, xm)
        self.save_image(x, './generated_mid', r_idx, b_sz, msg)
        x = inverse_data_transform(config, x0)
        self.save_image(x, self.args.sample_output_dir, r_idx, b_sz, msg)

    def backward_steps_et(self, x_T, re_seq, **kwargs):
        """
        Original paper: Denoising Diffusion Implicit Models. ICLR. 2021
        :param x_T: x_T in formula; it has initial Gaussian Noise
        :param re_seq:   reversed sequence of timestep t
        :param kwargs:
        :return:
        """
        eta = kwargs.get("eta", 0)
        msg = f"backward_steps_et()re_seq=[{re_seq[0]}~{re_seq[-1]}], len={len(re_seq)}"
        b_sz = x_T.size(0)
        xt = x_T
        for i in re_seq:
            t = (torch.ones(b_sz) * i).to(x_T.device)   # [999., 999.]
            at = self.alphas_cumprod.index_select(0, t.long()).view(-1, 1, 1, 1) # alpha_t
            aq = self.alphas_cumproq.index_select(0, t.long()).view(-1, 1, 1, 1) # alpha_{t-1}
            mt = self.alphas.index_select(0, t.long()).view(-1, 1, 1, 1)
            et = self.model_et(xt, t)               # epsilon_t
            if eta == 0:
                if i % 50 == 0: logging.info(f"{msg}; eta=0, i={i}")
                # simplified version of the formula.
                # when at is too small, divide by at may have issue.
                xt_next = (xt - (1 - at).sqrt() * et) / mt.sqrt() + (1 - aq).sqrt() * et
            else:
                if i % 50 == 0: logging.info(f"{msg}; eta={eta}; i={i}")
                x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()  # formula (9)
                sigma_t = eta * ((1 - at / aq) * (1 - aq) / (1 - at)).sqrt()  # formula (16)
                c2 = (1 - aq - sigma_t ** 2).sqrt()
                xt_next = aq.sqrt() * x0_t + c2 * et + sigma_t * torch.randn_like(x_T)  # formula (12)
            xt = xt_next
        # for
        return xt

    def backward_steps_x0(self, x_T, re_seq, **kwargs):
        eta = kwargs.get("eta", 0)
        msg = f"backward_steps_x0()re_seq=[{re_seq[0]}~{re_seq[-1]}], len={len(re_seq)}"
        b_sz = x_T.size(0)
        xt = x_T
        for i in re_seq:
            t = (torch.ones(b_sz) * i).to(x_T.device)   # [999., 999.]
            at = self.alphas_cumprod.index_select(0, t.long()).view(-1, 1, 1, 1)  # alpha_t
            aq = self.alphas_cumproq.index_select(0, t.long()).view(-1, 1, 1, 1)  # alpha_{t-1}
            x0 = self.model_x0(xt, t)
            if eta == 0:
                if i % 50 == 0: logging.info(f"{msg}; eta=0, i={i}")
                xt_next = aq.sqrt() * x0 + ((1-aq) / (1-at)).sqrt() * (xt - at.sqrt()*x0)
                xt = xt_next
            else:
                if i % 50 == 0: logging.info(f"{msg}; eta={eta}; i={i}")
                sigma_t = eta * ((1 - at / aq) * (1 - aq) / (1 - at)).sqrt()  # formula (16)
                c2 = ((1 - aq - sigma_t) / (1 - at)).sqrt() * (xt - at.sqrt()*x0)
                xt_next = aq.sqrt() * x0 + c2 + sigma_t * torch.randn_like(x_T)  # formula (12)
                xt = xt_next
        # for
        return xt

    def save_image(self, x, img_dir, r_idx, b_sz, msg=None):
        if not os.path.exists(img_dir):
            logging.info(f"Making dir: {img_dir}")
            os.makedirs(img_dir)
        img_cnt = len(x)
        logging.info(f"save {img_cnt} images to: {img_dir}. {msg}")
        img_first = img_last = None
        for i in range(img_cnt):
            img_id = r_idx * b_sz + i + self.sample_img_init_id
            img_path = os.path.join(img_dir, f"{img_id:05d}.png")
            tvu.save_image(x[i], img_path)
            if i == 0: img_first = f"{img_id:05d}.png"
            if i == img_cnt - 1: img_last = f"{img_id:05d}.png"
        logging.info(f"image generated: {img_first} ~ {img_last}")

    def load_from_local(self, model, ckpt_path, m_type):
        """
        Local model from local
        :param model:
        :param ckpt_path:
        :param m_type: model type, et or x0
        :return:
        """
        logging.info(f"load_from_local: {ckpt_path}")
        logging.info(f"  m_type: {m_type}")
        states = torch.load(ckpt_path, map_location=self.config.device)
        if m_type == 'et':
            if 'model' in states:
                logging.info(f"  model.load_state_dict(states['model'])")
                model.load_state_dict(states['model'])
            else:
                logging.info(f"  model.load_state_dict(states)")
                model.load_state_dict(states)
            ema_key = 'ema_helper'
        elif m_type == 'x0':
            if 'model_x0' in states:
                logging.info(f"  model.load_state_dict(states['model_x0'])")
                model.load_state_dict(states['model_x0'])
            else:
                logging.info(f"  model.load_state_dict(states)")
                model.load_state_dict(states)
            ema_key = 'ema_x0'
        else:
            raise Exception(f"Unknown m_type: {m_type}")
        if self.args.ema_flag and ema_key in states:
            logging.info(f"  ema_helper: EMAHelper(mu={self.args.ema_rate})")
            ema_helper = EMAHelper(mu=self.args.ema_rate)
            ema_helper.register(model)
            logging.info(f"  ema_helper: load from states[{ema_key}]")
            ema_helper.load_state_dict(states[ema_key])
            logging.info(f"  ema_helper: apply to model {type(model).__name__}")
            ema_helper.ema(model)
        elif self.args.ema_flag:
            logging.info(f"  !!! not found key: {ema_key}")

        logging.info(f"  model({type(model).__name__}).to({self.device})")
        model = model.to(self.device)
        if len(self.args.gpu_ids) > 1:
            logging.info(f"  torch.nn.DataParallel(model, device_ids={self.args.gpu_ids})")
            model = torch.nn.DataParallel(model, device_ids=self.args.gpu_ids)
        return model

# class
