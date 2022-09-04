import os
import time
import logging

import numpy as np
import torch

from datasets import inverse_data_transform
from functions.ckpt_util import get_ckpt_path
from models.diffusion import Model, ModelStack
from runners.diffusion import Diffusion

import torchvision.utils as tvu

from utils import count_parameters


class DiffusionSampling(Diffusion):
    def __init__(self, args, config, device=None):
        super().__init__(args, config, device)
        self.sample_count = 50000    # sample image count
        self.sample_img_init_id = 0  # sample image init ID. useful when generate image in parallel.

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

    def model_stack_load_from_local(self, model: ModelStack):
        root_dir = self.args.sample_ckpt_dir
        if self.args.sample_stack_size == 10:
            ckpt_path_arr = [
                os.path.join(root_dir, "ckpt_000-100.pth"),
                os.path.join(root_dir, "ckpt_100-200.pth"),
                os.path.join(root_dir, "ckpt_200-300.pth"),
                os.path.join(root_dir, "ckpt_300-400.pth"),
                os.path.join(root_dir, "ckpt_400-500.pth"),
                os.path.join(root_dir, "ckpt_500-600.pth"),
                os.path.join(root_dir, "ckpt_600-700.pth"),
                os.path.join(root_dir, "ckpt_700-800.pth"),
                os.path.join(root_dir, "ckpt_800-900.pth"),
                os.path.join(root_dir, "ckpt_900-1000.pth"),
            ]
        elif self.args.sample_stack_size == 4:
            ckpt_path_arr = [
                os.path.join(root_dir, "ckpt_000-250.pth"),
                os.path.join(root_dir, "ckpt_250-500.pth"),
                os.path.join(root_dir, "ckpt_500-750.pth"),
                os.path.join(root_dir, "ckpt_750-1000.pth"),
            ]
        else:  # other cases, need manual handling.
            ckpt_path_arr = [
                # f"{root_dir}/exp/model_S10E200/ckpt_000-100.pth",
                # f"{root_dir}/exp/model_S10E200/ckpt_100-200.pth",
                # f"{root_dir}/exp/model_S10E200/ckpt_200-300.pth",
                # f"{root_dir}/exp/model_S10E200/ckpt_300-400.pth",
                # f"{root_dir}/exp/model_S10E200/ckpt_400-500.pth",
                # f"{root_dir}/exp/model_S10E200/ckpt_500-600.pth",
                # f"{root_dir}/exp/model_S10E200/ckpt_600-700.pth",
                # f"{root_dir}/exp/model_S10E200/ckpt_700-800.pth",
                # f"{root_dir}/exp/model_S10E200/ckpt_800-900.pth",
                # f"{root_dir}/exp/model_S10E200/ckpt_900-1000.pth",
                f"{root_dir}/exp/model_stack_epoch500/ckpt_000-250.pth",
                f"{root_dir}/exp/model_stack_epoch500/ckpt_250-500.pth",
                f"{root_dir}/exp/model_stack_epoch500/ckpt_500-750.pth",
                f"{root_dir}/exp/model_stack_epoch500/ckpt_750-1000.pth",
            ]
        cnt, str_cnt = count_parameters(model, log_fn=None)
        logging.info(f"Loading ModelStack(stack_sz: {model.stack_sz})")
        logging.info(f"  config type  : {self.config.model.type}")
        logging.info(f"  param size   : {cnt} => {str_cnt}")
        logging.info(f"  ckpt_path_arr: {len(ckpt_path_arr)}")
        ms = model.model_stack
        for i, ckpt_path in enumerate(ckpt_path_arr):
            logging.info(f"  load ckpt {i: 2d} : {ckpt_path}")
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
        model = torch.nn.DataParallel(model, device_ids=self.args.gpu_ids)
        logging.info(f"  model.to({self.device})")
        logging.info(f"  torch.nn.DataParallel(model, device_ids={self.args.gpu_ids})")
        return model

    def sample(self):
        if self.args.sample_stack_size > 1:
            model = ModelStack(self.config, self.args.sample_stack_size)
        else:
            model = Model(self.config)

        if not self.args.use_pretrained:
            div_conquer = True
            if isinstance(model, ModelStack) and div_conquer:
                model = self.model_stack_load_from_local(model)
            else:
                model = self.model_load_from_local(model)
        else:
            # This used the pretrained DDPM model, see https://github.com/pesser/pytorch_diffusion
            if self.config.data.dataset == "CIFAR10":
                name = "cifar10"
            elif self.config.data.dataset == "LSUN":
                name = f"lsun_{self.config.data.category}"
            else:
                raise ValueError
            ckpt = get_ckpt_path(f"ema_{name}")
            print("Loading checkpoint {}".format(ckpt))
            model.load_state_dict(torch.load(ckpt, map_location=self.device))
            model.to(self.device)
            model = torch.nn.DataParallel(model, device_ids=self.args.gpu_ids)

        model.eval()

        if self.args.fid:
            self.sample_fid(model)            # Sampling from the generalized model for FID evaluation
        elif self.args.interpolation:
            self.sample_interpolation(model)  # Sampling from the model for image inpainting
        elif self.args.sequence:
            self.sample_sequence(model)       # Sampling from the sequence of images that lead to the sample
        else:
            raise NotImplementedError("Sample procedeure not defined")

    def sample_fid(self, model):
        config = self.config
        self.sample_count = self.args.sample_count
        self.sample_img_init_id = self.args.sample_img_init_id
        logging.info(f"sample_fid(self, {type(model).__name__})...")
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
                use_x0_for_xt = False  # hard coded switch
                if use_x0_for_xt:
                    x_t = self.gen_xt_from_x0(n, model)
                else:
                    x_t = torch.randn(  # normal distribution with mean 0 and variance 1
                        n,
                        config.data.channels,
                        config.data.image_size,
                        config.data.image_size,
                        device=self.device,
                    )
                if use_x0_for_xt:
                    self.sample_fid_decrease_avg(x_t, model, config)
                else:
                    self.sample_fid_vanilla(x_t, model, config, time_start, n_rounds, r_idx, b_sz)
                real_model = model.module if isinstance(model, torch.nn.DataParallel) else model
                if isinstance(real_model, ModelStack):
                    logging.info(f"ModelStack brick hit counter: {real_model.brick_hit_counter}")
            # for r_idx
        # with

    def sample_fid_vanilla(self, x_t, model, config, time_start, n_rounds, r_idx, b_sz):
        x = self.sample_image(x_t, model)
        x = inverse_data_transform(config, x)
        img_cnt = len(x)
        elp, eta = self.get_time_ttl_and_eta(time_start, r_idx+1, n_rounds)
        img_dir = self.args.sample_output_dir
        logging.info(f"save {img_cnt} images to: {img_dir}. round:{r_idx}/{n_rounds}, elp:{elp}, eta:{eta}")
        img_first = img_last = None
        for i in range(img_cnt):
            img_id = r_idx * b_sz + i + self.sample_img_init_id
            img_path = os.path.join(img_dir, f"{img_id:05d}.png")
            tvu.save_image(x[i], img_path)
            if i == 0: img_first = f"{img_id:05d}.png"
            if i == img_cnt - 1: img_last = f"{img_id:05d}.png"
        logging.info(f"image generated: {img_first} ~ {img_last}. "
                     f"init:{self.sample_img_init_id}; cnt:{self.sample_count}")

    def sample_fid_decrease_avg(self, x_t, model, config):
        x_t_ori = x_t.clone()
        avg_t = torch.mean(x_t, dim=0)
        limit = 10
        for k in range(0, limit + 1):
            logging.info(f"k: {k}/{limit} ================")
            x_t = x_t_ori - avg_t * (float(k) / limit)
            if k == limit:
                x_t += 1e-9  # epsilon. make it positive
            x = self.sample_image(x_t, model)
            x = inverse_data_transform(config, x)

            for i in range(len(x)):
                avg = torch.mean(x_t[i])
                var = torch.var(x_t[i], unbiased=False)
                img_path = os.path.join(self.args.sample_output_dir, f"{i:04d}_avg{avg:.4f}_var{var:.4f}.png")
                logging.info(f"save file: {img_path}")
                tvu.save_image(x[i], img_path)
            # for i
        # for k

    def sample_sequence(self, model):
        config = self.config

        x = torch.randn(
            8,
            config.data.channels,
            config.data.image_size,
            config.data.image_size,
            device=self.device,
        )

        # NOTE: This means that we are producing each predicted x0, not x_{t-1} at timestep t.
        with torch.no_grad():
            _, x = self.sample_image(x, model, last=False)

        x = [inverse_data_transform(config, y) for y in x]

        for i in range(len(x)):
            for j in range(x[i].size(0)):
                tvu.save_image(
                    x[i][j], os.path.join(self.args.sample_output_dir, f"{j}_{i}.png")
                )

    def sample_interpolation(self, model):
        config = self.config

        def slerp(z1, z2, alpha):
            theta = torch.acos(torch.sum(z1 * z2) / (torch.norm(z1) * torch.norm(z2)))
            return (
                torch.sin((1 - alpha) * theta) / torch.sin(theta) * z1
                + torch.sin(alpha * theta) / torch.sin(theta) * z2
            )

        z1 = torch.randn(
            1,
            config.data.channels,
            config.data.image_size,
            config.data.image_size,
            device=self.device,
        )
        z2 = torch.randn(
            1,
            config.data.channels,
            config.data.image_size,
            config.data.image_size,
            device=self.device,
        )
        alpha = torch.arange(0.0, 1.01, 0.1).to(z1.device)
        z_ = []
        for i in range(alpha.size(0)):
            z_.append(slerp(z1, z2, alpha[i]))

        x = torch.cat(z_, dim=0)
        xs = []

        # Hard coded here, modify to your preferences
        with torch.no_grad():
            for i in range(0, x.size(0), 8):
                xs.append(self.sample_image(x[i : i + 8], model))
        x = inverse_data_transform(config, torch.cat(xs, dim=0))
        for i in range(x.size(0)):
            tvu.save_image(x[i], os.path.join(self.args.sample_output_dir, f"{i}.png"))

    def gen_xt_from_x0(self, n, model, img_path_arr=None):
        from PIL import Image
        import torchvision.transforms as T
        import torchvision.transforms.functional as F
        cx, cy = 89, 121
        x1 = cy - 64
        x2 = cy + 64
        y1 = cx - 64
        y2 = cx + 64
        crop = lambda img: F.crop(img, x1, y1, x2 - x1, y2 - y1)
        config = self.config
        transform = T.Compose([crop, T.Resize(config.data.image_size), T.ToTensor()])
        if not img_path_arr:  # for testing
            img_path_arr = [f"./exp/{i:06d}.jpg" for i in range(1, 11)]
        x_arr = []
        for img_path in img_path_arr:
            logging.info(f"load file: {img_path}")
            image = Image.open(img_path)
            x = transform(image)    # shape [3, 64, 64]. value range [0, 1]
            x = x.unsqueeze(0)      # shape [1, 3, 64, 64]
            x_arr.append(x)
        x0 = torch.cat(x_arr).to(self.device)

        # save x0
        for i in range(len(x0)):
            img_path = os.path.join(self.args.sample_output_dir, f"{i:04d}.ori.png")
            logging.info(f"save file: {img_path}")
            tvu.save_image(x0[i], img_path)

        # apply diffusion
        b_seq = self.betas              # beta sequence
        m_seq = 1 - b_seq               # mean sequence. m = 1 - b
        a_seq = m_seq.cumprod(dim=0)    # alpha sequence
        a_t = a_seq[-1]
        xt = torch.randn(  # normal distribution with mean 0 and variance 1
            n,
            config.data.channels,
            config.data.image_size,
            config.data.image_size,
            device=self.device,
        )
        gen_by_model = True  # hard coded switch
        if gen_by_model:
            xt = self.gen_xt_from_x0_by_model(xt, x0, b_seq, m_seq, a_seq, model)
        else:
            xt = self.gen_xt_from_x0_by_formula(xt, x0, a_t)
        return xt

    @staticmethod
    def gen_xt_from_x0_by_model(xt, x0, b_seq, m_seq, a_seq, model):
        x = x0.clone()
        x0_sz = len(x)      # batch size
        seq_sz = len(b_seq) # sequence size
        for i in range(seq_sz):
            if i % 100 == 0:
                logging.info(f"gen_xt_from_x0_by_model(): {i:4d}/{seq_sz}")
            t = torch.ones(x0_sz, device=x0.device) * i  # [999., 999.]
            et = model(x, t)  # epsilon_t
            x = m_seq[i].sqrt() * x + ((1 - a_seq[i]).sqrt() - (m_seq[i] - a_seq[i]).sqrt()) * et
        for i in range(len(xt)):
            xt[i, :] = x[i % x0_sz, :]
        # epsilon = torch.randn_like(xt, device=xt.device)
        # xt = xt + epsilon  # if add epsilon, the generated image is messy and meaningless.
        return xt

    def gen_xt_from_x0_by_formula(self, xt, x0, a_t):
        # interpolation between two xt
        epsilon1 = torch.randn_like(x0[0], device=self.device)
        epsilon2 = torch.randn_like(x0[0], device=self.device)
        xt[0, :]  = a_t.sqrt() * x0[0] + (1 - a_t).sqrt() * epsilon1
        xt[-1, :] = a_t.sqrt() * x0[0] + (1 - a_t).sqrt() * epsilon2
        for i in range(1, len(xt) - 1):
            k = float(i) / (len(xt) - 1)
            epsilon = k * epsilon2 + (1 - k) * epsilon1
            xt[i, :] = a_t.sqrt() * x0[0] + (1 - a_t).sqrt() * epsilon
        return xt

    def sample_image(self, x, model, last=True):
        if self.args.sample_type == "generalized":
            if self.args.skip_type == "uniform":
                skip = self.num_timesteps // self.args.timesteps
                seq = range(0, self.num_timesteps, skip)
            elif self.args.skip_type == "quad":
                seq = (np.linspace(0, np.sqrt(self.num_timesteps * 0.8), self.args.timesteps) ** 2)
                seq = [int(s) for s in list(seq)]
            else:
                raise NotImplementedError
            from functions.denoising import generalized_steps

            xs = generalized_steps(x, seq, model, self.betas, eta=self.args.eta)
            x = xs
        elif self.args.sample_type == "ddpm_noisy":
            if self.args.skip_type == "uniform":
                skip = self.num_timesteps // self.args.timesteps
                seq = range(0, self.num_timesteps, skip)
            elif self.args.skip_type == "quad":
                seq = (
                    np.linspace(
                        0, np.sqrt(self.num_timesteps * 0.8), self.args.timesteps
                    )
                    ** 2
                )
                seq = [int(s) for s in list(seq)]
            else:
                raise NotImplementedError
            from functions.denoising import ddpm_steps

            x = ddpm_steps(x, seq, model, self.betas)
        else:
            raise NotImplementedError
        if last:
            x = x[0][-1]
        return x

# class
