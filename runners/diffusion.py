import os
import logging
import time
import glob

import numpy as np
import tqdm
import torch
import torch.utils.data as data
import torch.backends.cudnn as cudnn

from models.diffusion import Model
from models.ema import EMAHelper
from functions import get_optimizer
from functions.losses import loss_registry
from datasets import get_dataset, data_transform, inverse_data_transform
from functions.ckpt_util import get_ckpt_path

import torchvision.utils as tvu


def torch2hwcuint8(x, clip=False):
    if clip:
        x = torch.clamp(x, -1, 1)
    x = (x + 1.0) / 2.0
    return x


def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
    def sigmoid(x):
        return 1 / (np.exp(-x) + 1)

    if beta_schedule == "quad":
        betas = (
            np.linspace(
                beta_start ** 0.5,
                beta_end ** 0.5,
                num_diffusion_timesteps,
                dtype=np.float64,
            )
            ** 2
        )
    elif beta_schedule == "linear":
        betas = np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "const":
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "jsd":  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1.0 / np.linspace(
            num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "sigmoid":
        betas = np.linspace(-6, 6, num_diffusion_timesteps)
        betas = sigmoid(betas) * (beta_end - beta_start) + beta_start
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas


class Diffusion(object):
    def __init__(self, args, config, device=None):
        self.args = args
        self.config = config
        if device is None:
            device = (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
        self.device = device

        self.model_var_type = config.model.var_type
        betas = get_beta_schedule(
            beta_schedule=config.diffusion.beta_schedule,
            beta_start=config.diffusion.beta_start,
            beta_end=config.diffusion.beta_end,
            num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps,
        )
        betas = self.betas = torch.from_numpy(betas).float().to(self.device)
        self.num_timesteps = betas.shape[0]

        alphas = 1.0 - betas
        alphas_cumprod = alphas.cumprod(dim=0)
        alphas_cumprod_prev = torch.cat(
            [torch.ones(1).to(device), alphas_cumprod[:-1]], dim=0
        )
        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        if self.model_var_type == "fixedlarge":
            self.logvar = betas.log()
            # torch.cat(
            # [posterior_variance[1:2], betas[1:]], dim=0).log()
        elif self.model_var_type == "fixedsmall":
            self.logvar = posterior_variance.clamp(min=1e-20).log()

    def train(self):
        args, config = self.args, self.config
        tb_logger = self.config.tb_logger
        dataset, test_dataset = get_dataset(args, config)
        logging.info(f"dataset:")
        logging.info(f"  root : {dataset.root}")
        logging.info(f"  split: {dataset.split}") if hasattr(dataset, 'split') else None
        logging.info(f"  len  : {len(dataset)}")
        logging.info(f"data_loader:")
        logging.info(f"  batch_size : {config.training.batch_size}")
        logging.info(f"  shuffle    : True")
        logging.info(f"  num_workers: {config.data.num_workers}")
        train_loader = data.DataLoader(
            dataset,
            batch_size=config.training.batch_size,
            shuffle=True,
            num_workers=config.data.num_workers,
        )
        model = Model(config)

        model = model.to(self.device)
        model = torch.nn.DataParallel(model, device_ids=args.gpu_ids)
        cudnn.benchmark = True
        logging.info(f"model: {config.model.type}")
        logging.info(f"  model.to({self.device})")
        logging.info(f"  torch.nn.DataParallel(device_ids={args.gpu_ids})")

        optimizer = get_optimizer(self.config, model.parameters())

        if self.config.model.ema:
            ema_helper = EMAHelper(mu=self.config.model.ema_rate)
            ema_helper.register(model)
        else:
            ema_helper = None
        logging.info(f"ema_helper: {type(ema_helper).__name__}")

        start_epoch, step = 0, 0
        if self.args.resume_training:
            logging.info(f"resume_training: {self.args.log_path}")
            states = torch.load(os.path.join(self.args.log_path, "ckpt.pth"))
            model.load_state_dict(states[0])

            states[1]["param_groups"][0]["eps"] = self.config.optim.eps
            optimizer.load_state_dict(states[1])
            start_epoch = states[2]
            step = states[3]
            logging.info(f"  resume optimizer  : eps={states[1]['param_groups'][0]['eps']}")
            logging.info(f"  resume start_epoch: {start_epoch}")
            logging.info(f"  resume step       : {step}")
            if self.config.model.ema:
                ema_helper.load_state_dict(states[4])
                logging.info(f"  resume ema_helper : ...")

        e_cnt = self.config.training.n_epochs
        b_cnt = len(train_loader)
        eb_cnt = e_cnt * b_cnt      # epoch * batch
        data_start = time.time()
        for epoch in range(start_epoch, self.config.training.n_epochs):
            for i, (x, y) in enumerate(train_loader):
                n = x.size(0)
                model.train()
                step += 1

                x = x.to(self.device)
                x = data_transform(self.config, x)
                e = torch.randn_like(x)
                b = self.betas

                # antithetic sampling
                t = torch.randint(
                    low=0, high=self.num_timesteps, size=(n // 2 + 1,), device=self.device
                )
                t = torch.cat([t, self.num_timesteps - t - 1], dim=0)[:n]
                loss = loss_registry[config.model.type](model, x, t, e, b)

                tb_logger.add_scalar("loss", loss, global_step=step)

                if i % 5 == 0 or i == b_cnt - 1:
                    elp, eta = self.get_time_ttl_and_eta(data_start, epoch * b_cnt + i, eb_cnt)
                    logging.info(f"E:{epoch}/{e_cnt}, B:{i:02d}/{b_cnt}, loss:{loss.item():.4f}; elp:{elp}; eta:{eta}")

                optimizer.zero_grad()
                loss.backward()

                try:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), config.optim.grad_clip
                    )
                except Exception:
                    pass
                optimizer.step()

                if self.config.model.ema:
                    ema_helper.update(model)

                if step % self.config.training.snapshot_freq == 0 or step == 1:
                    states = [
                        model.state_dict(),
                        optimizer.state_dict(),
                        epoch,
                        step,
                    ]
                    if self.config.model.ema:
                        states.append(ema_helper.state_dict())

                    fpath = os.path.join(self.args.log_path, "ckpt_{}.pth".format(step))
                    logging.info(f"save: {fpath}")
                    torch.save(states, fpath)
                    fpath = os.path.join(self.args.log_path, "ckpt.pth")
                    logging.info(f"save: {fpath}")
                    torch.save(states, fpath)

    @staticmethod
    def get_time_ttl_and_eta(time_start, elapsed_iter, total_iter):
        """
        Get estimated total time and ETA time.
        :param time_start:
        :param elapsed_iter:
        :param total_iter:
        :return:
        """
        def sec_to_str(sec):
            val = int(sec)      # seconds in int type
            s = val % 60
            val = val // 60     # minutes
            m = val % 60
            val = val // 60     # hours
            h = val % 24
            d = val // 24       # days
            return f"{d}-{h:02d}:{m:02d}:{s:02d}"

        elapsed_time = time.time() - time_start  # seconds elapsed
        elp = sec_to_str(elapsed_time)
        if elapsed_iter == 0:
            eta = 'NA'
        else:
            # seconds
            eta = elapsed_time * (total_iter - elapsed_iter) / elapsed_iter
            eta = sec_to_str(eta)
        return elp, eta

    def sample(self):
        model = Model(self.config)

        if not self.args.use_pretrained:
            if getattr(self.config.sampling, "ckpt_id", None) is None:
                states = torch.load(
                    os.path.join(self.args.log_path, "ckpt.pth"),
                    map_location=self.config.device,
                )
            else:
                states = torch.load(
                    os.path.join(self.args.log_path, f"ckpt_{self.config.sampling.ckpt_id}.pth"),
                    map_location=self.config.device,
                )
            model = model.to(self.device)
            model = torch.nn.DataParallel(model, device_ids=self.args.gpu_ids)
            model.load_state_dict(states[0], strict=True)

            if self.config.model.ema:
                ema_helper = EMAHelper(mu=self.config.model.ema_rate)
                ema_helper.register(model)
                ema_helper.load_state_dict(states[-1])
                ema_helper.ema(model)
            else:
                ema_helper = None
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
        # img_id = len(glob.glob(f"{self.args.image_folder}/*"))
        img_id = 0
        total_n_samples = 10
        n_new = total_n_samples - img_id
        logging.info(f"image_folder   : {self.args.image_folder}")
        logging.info(f"init img_id    : {img_id}")
        logging.info(f"total_n_samples: {total_n_samples}")
        logging.info(f"n_new          : {n_new}")
        if n_new <= 0:
            logging.warning(f"Will not sample images due to n_new = {n_new}")
            return
        b_sz = config.sampling.batch_size
        n_rounds = (n_new - 1) // b_sz + 1  # get the ceiling
        logging.info(f"batch_size     : {b_sz}")
        logging.info(f"n_rounds       : {n_rounds}")
        logging.info(f"Generating image samples for FID evaluation")
        with torch.no_grad():
            for r_idx in range(n_rounds):
                logging.info(f"round: {r_idx}/{n_rounds}")
                n = b_sz if r_idx + 1 < n_rounds else n_new - r_idx * b_sz
                use_x0_for_xt = True
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

                x_t_ori = x_t.clone()
                avg_t = torch.mean(x_t, dim=0)
                limit = 10
                for k in range(0, limit+1):
                    logging.info(f"k: {k}/{limit} ================")
                    x_t = x_t_ori - avg_t * (float(k) / limit)
                    if k == limit:
                        x_t += 1e-9  # epsilon. make it positive
                    x = self.sample_image(x_t, model)
                    x = inverse_data_transform(config, x)

                    for i in range(len(x)):
                        avg = torch.mean(x_t[i])
                        var = torch.var(x_t[i], unbiased=False)
                        img_path = os.path.join(self.args.image_folder, f"{i:04d}_avg{avg:.4f}_var{var:.4f}.png")
                        logging.info(f"save file: {img_path}")
                        tvu.save_image(x[i], img_path)
                        img_id += 1
                    # for i
                # for k
            # for r_idx
        # with

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
                    x[i][j], os.path.join(self.args.image_folder, f"{j}_{i}.png")
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
            tvu.save_image(x[i], os.path.join(self.args.image_folder, f"{i}.png"))

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
            img_path = os.path.join(self.args.image_folder, f"{i:04d}.ori.png")
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
        gen_by_model = True
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
            xt[i] = x[i % x0_sz]
        # epsilon = torch.randn_like(xt, device=xt.device)
        # xt = xt + epsilon  # if add epsilon, the generated image is messy and meaningless.
        return xt

    def gen_xt_from_x0_by_formula(self, xt, x0, a_t):
        # interpolation between two xt
        epsilon1 = torch.randn_like(x0[0], device=self.device)
        epsilon2 = torch.randn_like(x0[0], device=self.device)
        xt[0]  = a_t.sqrt() * x0[0] + (1 - a_t).sqrt() * epsilon1
        xt[-1] = a_t.sqrt() * x0[0] + (1 - a_t).sqrt() * epsilon2
        for i in range(1, len(xt) - 1):
            k = float(i) / (len(xt) - 1)
            epsilon = k * epsilon2 + (1 - k) * epsilon1
            xt[i] = a_t.sqrt() * x0[0] + (1 - a_t).sqrt() * epsilon
        return xt

    def sample_image(self, x, model, last=True):
        if self.args.sample_type == "generalized":
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

    def test(self):
        pass
