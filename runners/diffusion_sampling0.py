import os
import time
import logging
import torch
import utils
import torchvision.utils as tvu
from datasets import inverse_data_transform
from models.diffusion import Model, ModelStack
from runners.diffusion import Diffusion


class DiffusionSampling0(Diffusion):
    """
    Sampling by model who predicts x0.
    This corresponds to DiffusionTraining0 in diffusion_training0.py
    """
    def __init__(self, args, config, device=None):
        super().__init__(args, config, device)
        self.sample_count = 50000    # sample image count
        self.sample_img_init_id = 0  # sample image init ID. useful when generate image in parallel.
        self.model_x0 = None
        skip = self.num_timesteps // self.args.timesteps
        self.seq = range(0, self.num_timesteps, skip)


    def sample(self):
        config = self.config
        args = self.args
        ckpt_path_arr = self.get_ckpt_path_arr()
        self.sample_count = args.sample_count
        self.sample_img_init_id = args.sample_img_init_id
        if len(ckpt_path_arr) > 1:
            self.model_x0 = ModelStack(config, len(ckpt_path_arr))
            self.model_stack_load(ckpt_path_arr)
        else:
            self.model_x0 = Model(config)
            self.model_load()
        self.model_x0.eval()

        logging.info(f"DiffusionSampling0.sample(self, {type(self.model_x0).__name__})...")
        if os.path.exists(args.sample_output_dir):
            logging.info(f"  args.sample_output_dir : {args.sample_output_dir}")
        else:
            logging.info(f"  args.sample_output_dir : {args.sample_output_dir}, mkdir...")
            os.makedirs(args.sample_output_dir)
        logging.info(f"  args.sample_type       : {args.sample_type}")
        logging.info(f"  args.skip_type         : {args.skip_type}")
        logging.info(f"  args.timesteps         : {args.timesteps}")
        logging.info(f"  num_timesteps          : {self.num_timesteps}")
        logging.info(f"  sample_count           : {self.sample_count}")
        logging.info(f"  sample_img_init_id     : {self.sample_img_init_id}")
        b_sz = args.sample_batch_size or config.sampling.batch_size
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
                self.sample_fid_vanilla(x_t, time_start, n_rounds, r_idx, b_sz)
                if isinstance(self.model_x0, torch.nn.DataParallel):
                    real_model = self.model_x0.module
                else:
                    real_model = self.model_x0
                if isinstance(real_model, ModelStack):
                    logging.info(f"ModelStack brick hit counter: {real_model.brick_hit_counter}")
            # for r_idx
        # with

    def sample_fid_vanilla(self, x_t, time_start, n_rounds, r_idx, b_sz):
        x = self.sample_image(x_t, time_start, n_rounds, r_idx)
        x = inverse_data_transform(self.config, x)
        img_cnt = len(x)
        img_dir = self.args.sample_output_dir
        logging.info(f"save {img_cnt} images to: {img_dir}. round:{r_idx}/{n_rounds}")
        img_first = img_last = None
        for i in range(img_cnt):
            img_id = r_idx * b_sz + i + self.sample_img_init_id
            img_path = os.path.join(img_dir, f"{img_id:05d}.png")
            tvu.save_image(x[i], img_path)
            if i == 0: img_first = f"{img_id:05d}.png"
            if i == img_cnt - 1: img_last = f"{img_id:05d}.png"
        logging.info(f"image generated: {img_first} ~ {img_last}. "
                     f"init:{self.sample_img_init_id}; cnt:{self.sample_count}")

    def sample_image(self, x_T, time_start, r_cnt, r_idx):
        seq = self.seq
        i_cnt = len(seq)        # iteration count
        ir_cnt = i_cnt * r_cnt  # total iterations in all rounds
        msg = f"sampling0::sample_image():seq_len={len(seq)}"
        b_sz = x_T.size(0)
        xt = x_T
        for i in reversed(seq):
            t = (torch.ones(b_sz) * i).to(x_T.device)   # [999., 999.]
            at = self.alphas_cumprod.index_select(0, t.long()).view(-1, 1, 1, 1)  # alpha_t
            aq = self.alphas_cumproq.index_select(0, t.long()).view(-1, 1, 1, 1)  # alpha_{t-1}
            x0 = self.model_x0(xt, t)
            if i % 50 == 0:
                elp, eta = utils.get_time_ttl_and_eta(time_start, r_idx*i_cnt+(i_cnt-i), ir_cnt)
                logging.info(f"{msg}; i={i}, elp:{elp}, eta:{eta}")
            xt_next = aq.sqrt() * x0 + ((1-aq) / (1-at)).sqrt() * (xt - at.sqrt()*x0)
            xt = xt_next
        # for
        return xt

    def model_stack_load(self, ckpt_path_arr):
        cnt, str_cnt = utils.count_parameters(self.model_x0, log_fn=None)
        logging.info(f"Loading ModelStack(stack_sz: {self.model_x0.stack_sz})")
        logging.info(f"  config type  : {self.config.model.type}")
        logging.info(f"  param size   : {cnt} => {str_cnt}")
        logging.info(f"  ckpt_path_arr: {len(ckpt_path_arr)}")
        ms_x0 = self.model_x0.model_stack
        for i, ckpt_path in enumerate(ckpt_path_arr):
            tsr = utils.extract_ts_range(ckpt_path)
            self.model_x0.tsr_stack[i] = tsr
            logging.info(f"  load ckpt {i: 2d} : {ckpt_path}. ts: {self.model_x0.tsr_stack[i]}")
            states = torch.load(ckpt_path, map_location=self.config.device)
            ms_x0[i].load_state_dict(states['model_x0'], strict=True)
        # for
        self.model_x0 = self.model_x0.to(self.device)
        logging.info(f"  model_x0.to({self.device})")
        if len(self.args.gpu_ids) > 0:
            self.model_x0 = torch.nn.DataParallel(self.model_x0, device_ids=self.args.gpu_ids)
            logging.info(f"  torch.nn.DataParallel(model_x0, device_ids={self.args.gpu_ids})")
        return True

    def model_load(self):
        ckpt_path = self.args.sample_ckpt_path
        logging.info(f"load ckpt0: {ckpt_path}")
        states = torch.load(ckpt_path, map_location=self.config.device)
        self.model_x0.load_state_dict(states['model_x0'], strict=True)
        self.model_x0 = self.model_x0.to(self.device)
        logging.info(f"  model_x0.to({self.device})")
        if len(self.args.gpu_ids) > 1:
            logging.info(f"  torch.nn.DataParallel(model_x0, device_ids={self.args.gpu_ids})")
            self.model_x0 = torch.nn.DataParallel(self.model_x0, device_ids=self.args.gpu_ids)
        return True

# class
