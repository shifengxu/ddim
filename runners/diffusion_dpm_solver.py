import os
import time
import logging

import torch
import torch_fidelity

import utils
from datasets import inverse_data_transform
from models.diffusion import Model
from runners.diffusion import Diffusion
from models.dpm_solver2_pytorch import model_wrapper, DPM_Solver
from models.noise_schedule2 import NoiseScheduleVP2

import torchvision.utils as tvu


class DiffusionDpmSolver(Diffusion):
    def __init__(self, args, config, device=None):
        super().__init__(args, config, device)
        self.sample_count = 50000    # sample image count
        self.sample_img_init_id = 0  # sample image init ID. useful when generate image in parallel.
        self.dpm_solver = None
        self.noise_schedule = None
        self.t_start = None
        self.t_end = None
        self.order = 3
        self.steps = 20
        self.skip_type = 'time_uniform'

    def save_images(self, config, x, time_start, r_idx, n_rounds, b_sz):
        x = inverse_data_transform(config, x)
        img_cnt = len(x)
        elp, eta = utils.get_time_ttl_and_eta(time_start, r_idx+1, n_rounds)
        img_dir = self.args.sample_output_dir
        img_path = None
        for i in range(img_cnt):
            img_id = r_idx * b_sz + i + self.sample_img_init_id
            img_path = os.path.join(img_dir, f"{img_id:05d}.png")
            tvu.save_image(x[i], img_path)
        logging.info(f"  saved {img_cnt} images: {img_path}. elp:{elp}, eta:{eta}")

    def sample_ratios(self, r_start=1., r_end=1.2, step=0.001):
        logging.info(f"DiffusionDpmSolver::sample_ratios({r_start:.4f}, {r_end:.4f}, {step:.4f})")
        args = self.args
        ratio_arr = []
        fid_arr = []
        _r = r_start  # ratio
        # add epsilon to r_end to avoid float point issue (last r might be 1.9900000000000009)
        eps = 1e-8
        while _r <= r_end + eps:
            r, _r = _r, _r+step
            aap_file = f"geometric_ratio:{r:.4f}"
            logging.info(f"{aap_file} *******************************************")
            args.predefined_aap_file = aap_file
            self.sample()
            metrics_dict = torch_fidelity.calculate_metrics(
                input1='cifar10-train',
                input2=args.sample_output_dir,
                cuda=True,
                isc=False,
                fid=True,
                kid=False,
                verbose=False,
                samples_find_deep=True,
            )
            logging.info(f"Calculating FID...")
            fid = metrics_dict['frechet_inception_distance']
            logging.info(f"{aap_file} => FID: {fid:.6f}")
            ratio_arr.append(r)
            fid_arr.append(fid)
            with open('./ratio_and_fid.txt', 'w') as f_ptr:
                f_ptr.write(f"# r_start:{r_start}, r_end:{r_end}, step:{step}\n")
                f_ptr.write(f"# current ratio: {r}\n")
                f_ptr.write(f"# ratio: fid\n")
                for i in range(len(ratio_arr)):
                    f_ptr.write(f"{ratio_arr[i]:.4f}: {fid_arr[i]:7.4f}\n")
                # for
            # with
        # while

    def sample_all(self):
        args = self.args
        args.predefined_aap_file = ""
        logging.info(f"DiffusionDpmSolver::sample_all()")
        logging.info(f"  args.predefined_aap_file: '{args.predefined_aap_file}'")
        order_arr = [1, 2, 3]
        steps_arr = [10, 15, 20, 25, 50, 100]
        skip_arr = ['time_uniform', 'logSNR', 'time_quadratic']
        times = 5
        logging.info(f"  order_arr: {order_arr}")
        logging.info(f"  steps_arr: {steps_arr}")
        logging.info(f"  skip_arr : {skip_arr}")
        logging.info(f"  times    : {times}")
        fid_dict = {}
        for order in order_arr:
            self.order = order
            for steps in steps_arr:
                self.steps = steps
                for skip_type in skip_arr:
                    self.skip_type = skip_type
                    fid_avg = self.sample_times(times)
                    key = f"{order}-{steps}-{skip_type}"
                    fid_dict[key] = fid_avg
                    logging.info(f"fid_avg: {key}: {fid_avg:8.5f}")
                # for
            # for
        # for
        logging.info(f"fid_dict:")
        for key in sorted(fid_dict):
            logging.info(f"  {key}: {fid_dict[key]:8.5f}")

    def alpha_bar_all(self):
        def save_ab_file(file_path):
            ab_map = self.noise_schedule.alpha_bar_map
            ab_list = list(ab_map)
            ab_list.sort(reverse=True)
            if len(ab_list) != self.steps + 1:
                raise Exception(f"alpha_bar count {len(ab_list)} not match steps {self.steps}")
            with open(file_path, 'w') as f_ptr:
                f_ptr.write(f"# order     : {self.order}\n")
                f_ptr.write(f"# steps     : {self.steps}\n")
                f_ptr.write(f"# skip_type : {self.skip_type}\n")
                f_ptr.write(f"# data_type : alpha_bar\n")
                [f_ptr.write(f"{ab_map[k]}\n") for k in ab_list]
            # with
        # def
        args = self.args
        args.predefined_aap_file = ""
        args.sample_count = 5  # sample is not purpose. So make it small
        logging.info(f"DiffusionDpmSolver::alpha_bar_all()")
        logging.info(f"  args.predefined_aap_file: '{args.predefined_aap_file}'")
        logging.info(f"  args.sample_count       : '{args.sample_count}'")
        order_arr = [1, 2, 3]
        steps_arr = [10, 15, 20, 25, 50, 100]
        skip_arr = ['logSNR', 'time_quadratic', 'time_uniform']
        times = 5
        logging.info(f"  order_arr: {order_arr}")
        logging.info(f"  steps_arr: {steps_arr}")
        logging.info(f"  skip_arr : {skip_arr}")
        logging.info(f"  times    : {times}")
        for order in order_arr:
            self.order = order
            for steps in steps_arr:
                self.steps = steps
                for skip_type in skip_arr:
                    self.skip_type = skip_type
                    self.sample()
                    f_path = f"dpm_alphaBar_{order}-{steps:03d}-{skip_type}.txt"
                    save_ab_file(f_path)
                    logging.info(f"File saved: {f_path}")
                # for
            # for
        # for

    def sample_times(self, times: int):
        args, order, steps, skip_type = self.args, self.order, self.steps, self.skip_type
        fid_arr = []
        for i in range(times):
            self.sample()
            logging.info(f"Calculating FID...")
            metrics_dict = torch_fidelity.calculate_metrics(
                input1='cifar10-train',
                input2=args.sample_output_dir,
                cuda=True,
                isc=False,
                fid=True,
                kid=False,
                verbose=False,
                samples_find_deep=True,
            )
            fid = metrics_dict['frechet_inception_distance']
            logging.info(f"{order}-{steps}-{skip_type}-{i} => FID: {fid:.6f}")
            fid_arr.append(fid)
        # for
        fid_sum = 0.
        for fid in fid_arr:
            fid_sum += fid
        avg = fid_sum / len(fid_arr)
        return avg

    def sample(self):
        config, args = self.config, self.args
        model = Model(config)
        model = self.model_load_from_local(model)
        model.eval()

        self.sample_count = args.sample_count
        self.sample_img_init_id = args.sample_img_init_id
        logging.info(f"DiffusionDpmSolver::sample(self, {type(model).__name__})...")
        logging.info(f"  args.sample_output_dir : {args.sample_output_dir}")
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
        if not os.path.exists(args.sample_output_dir):
            logging.info(f"  os.makedirs({args.sample_output_dir})")
            os.makedirs(args.sample_output_dir)
        time_start = time.time()
        d = config.data
        params = self.create_dpm_solver(model, self.device)
        with torch.no_grad():
            for r_idx in range(n_rounds):
                n = b_sz if r_idx + 1 < n_rounds else self.sample_count - r_idx * b_sz
                logging.info(f"DiffusionDpmSolver::round: {r_idx}/{n_rounds}. to generate: {n}")
                x_t = torch.randn(n, d.channels, d.image_size, d.image_size, device=self.device)
                x = self.sample_by_dpm_solver(x_t, *params)
                self.save_images(config, x, time_start, r_idx, n_rounds, b_sz)
            # for r_idx
        # with

    def save_noise_schedule_data(self, aap_file, noise_schedule):
        f_path = f"./noise_schedule_predefined_ts.txt"
        logging.info(f"Saving file: {f_path}")
        with open(f_path, 'w') as f_ptr:
            f_ptr.write(f"# noise_schedule: {type(noise_schedule).__name__}\n")
            f_ptr.write(f"# aap_file: {aap_file}\n")
            f_ptr.write(f"# aap: alpha accumulated product\n")
            f_ptr.write(f"# aap   \ttimestep\n")
            aap = noise_schedule.predefined_aap
            ts = noise_schedule.predefined_ts
            for i in range(len(aap)):
                f_ptr.write(f"{aap[i]:8.6f}\t{ts[i]: 4d}\n")
            # for
        # with
        f_path = f"./noise_schedule_predefined_aap.txt"
        logging.info(f"Saving file: {f_path}")
        with open(f_path, 'w') as f_ptr:
            f_ptr.write(f"# noise_schedule: {type(noise_schedule).__name__}\n")
            f_ptr.write(f"# beta_schedule: {self.beta_schedule}\n")
            f_ptr.write(f"# aap_file: {aap_file}\n")
            f_ptr.write(f"# aap: alpha accumulated product\n")
            aap = noise_schedule.alphas_cumprod
            for i in range(len(aap)):
                f_ptr.write(f"{aap[i]:8.6f}\n")
            # for
        # with

    def create_dpm_solver(self, model, device):
        # You need to firstly define your model and the extra inputs of your model,
        # And initialize an `x_T` from the standard normal distribution.
        # `model` has the format: model(x_t, t_input, **model_kwargs).
        # If your model has no extra inputs, just let model_kwargs = {}.
        #
        # If you use discrete-time DPMs, you need to further define the
        # beta arrays for the noise schedule.
        #
        # model = ....
        # model_kwargs = {...}
        # x_T = ...
        # betas = ....

        # 1. Define the noise schedule.
        aap_file = self.args.predefined_aap_file
        if aap_file:
            noise_schedule = NoiseScheduleVP2(schedule='predefined',
                                              alphas_cumprod=self.alphas_cumprod,
                                              predefined_aap_file=aap_file)
            noise_schedule.to(device)
            self.save_noise_schedule_data(aap_file, noise_schedule)
        else:
            noise_schedule = NoiseScheduleVP2(schedule='discrete', alphas_cumprod=self.alphas_cumprod)
        if self.args.todo.endswith('.alpha_bar_all'):
            noise_schedule.alpha_bar_map = {}  # only init it for specific args.todo

        # 2. Convert your discrete-time `model` to the continuous-time
        # noise prediction model. Here is an example for a diffusion model
        # `model` with the noise prediction type ("noise") .
        model_fn = model_wrapper(
            model,
            noise_schedule,
            model_type="noise",  # or "x_start" or "v" or "score"
            model_kwargs={},
        )

        # 3. Define dpm-solver and sample by singlestep DPM-Solver.
        # (We recommend singlestep DPM-Solver for unconditional sampling)
        # You can adjust the `steps` to balance the computation
        # costs and the sample quality.
        if aap_file:
            dpm_solver = DPM_Solver(model_fn, noise_schedule, algorithm_type="dpmsolver", skip_type="predefined")
            t_start, t_end = noise_schedule.predefined_aap_cnt, 0
        else:
            dpm_solver = DPM_Solver(model_fn, noise_schedule, algorithm_type="dpmsolver", skip_type=self.skip_type)
            t_start, t_end = None, None

        # Can also try
        # dpm_solver = DPM_Solver(model_fn, noise_schedule, algorithm_type="dpmsolver++")

        self.dpm_solver = dpm_solver
        self.noise_schedule = noise_schedule
        return dpm_solver, t_start, t_end

    def sample_by_dpm_solver(self, x_T, dpm_solver: DPM_Solver, t_start, t_end):
        # You can use steps = 10, 12, 15, 20, 25, 50, 100.
        # Empirically, we find that steps in [10, 20] can generate quite good samples.
        # And steps = 20 can almost converge.
        x_sample = dpm_solver.sample(
            x_T,
            steps=self.steps,
            # skip_type="time_uniform",
            order=self.order,
            method="singlestep",
            t_start=t_start,
            t_end=t_end,
        )
        return x_sample

# class
