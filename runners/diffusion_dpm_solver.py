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
        super().__init__(args, config, device, output_ab=False)
        self.sample_count = 50000    # sample image count
        self.dpm_solver = None
        self.noise_schedule = None
        self.t_start = None
        self.t_end = None
        self.order = 3
        self.steps = 20
        self.skip_type = 'time_uniform'
        self.alpha_bar_all_flag = False

    def save_images(self, config, x, time_start, r_idx, n_rounds, b_sz):
        x = inverse_data_transform(config, x)
        img_cnt = len(x)
        elp, eta = utils.get_time_ttl_and_eta(time_start, r_idx+1, n_rounds)
        img_dir = self.args.sample_output_dir
        img_path = None
        for i in range(img_cnt):
            img_id = r_idx * b_sz + i
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

    def sample_all_scheduled(self):
        args = self.args
        af = args.predefined_aap_file
        logging.info(f"DiffusionDpmSolver::sample_all_scheduled() ***********************************")
        logging.info(f"  args.predefined_aap_file: {af}")
        if not af.startswith('all_scheduled_dir:'):
            raise ValueError(f"Invalid args.predefined_aap_file: {af}")
        sum_dir = args.ab_summary_dir
        if not os.path.exists(sum_dir):
            logging.info(f"  os.makedirs({sum_dir})")
            os.makedirs(sum_dir)
        arr = af.split(':')
        s_dir = arr[1].strip()
        f_list = os.listdir(s_dir)                          # file name list
        f_list = [f for f in f_list if f.endswith('.txt')]  # filter by extension
        f_list = [os.path.join(s_dir, f) for f in f_list]   # file path list
        f_list = [f for f in f_list if os.path.isfile(f)]   # remove folder
        times = args.repeat_times
        logging.info(f"  repeat_times: {times}")
        logging.info(f"  files count : {len(f_list)}")
        fid_dict = {}
        for f_path in sorted(f_list):
            logging.info(f"aap_file: {f_path} ------------------------------------")
            args.predefined_aap_file = f_path
            fid_avg = self.sample_times(times)
            fid_dict[f_path] = fid_avg
            logging.info(f"fid_avg: {f_path}: {fid_avg:8.5f}")
            f_path = os.path.join(sum_dir, 'sample_all_scheduled_fid.txt')
            logging.info(f"Save file: {f_path}")
            with open(f_path, 'w') as f_ptr:
                for key in sorted(fid_dict):
                    f_ptr.write(f"{fid_dict[key]:9.5f}: {key}\n")
            # with
        # for

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
        self.alpha_bar_all_flag = True
        args = self.args
        args.predefined_aap_file = ""
        args.sample_count = 5  # sample is not purpose. So make it small
        logging.info(f"DiffusionDpmSolver::alpha_bar_all()")
        logging.info(f"  args.predefined_aap_file: '{args.predefined_aap_file}'")
        logging.info(f"  args.sample_count       : '{args.sample_count}'")
        order_arr = args.order_arr or [1, 2, 3]
        steps_arr = args.steps_arr or [10, 15, 20, 25, 50, 100]
        skip_arr = args.skip_type_arr or ['logSNR', 'time_quadratic', 'time_uniform']
        ab_dir = args.ab_original_dir or '.'
        logging.info(f"  order_arr: {order_arr}")
        logging.info(f"  steps_arr: {steps_arr}")
        logging.info(f"  skip_arr : {skip_arr}")
        logging.info(f"  o_dir    : {ab_dir}")
        if not os.path.exists(ab_dir):
            logging.info(f"  os.makedirs({ab_dir})")
            os.makedirs(ab_dir)
        for order in order_arr:
            self.order = order
            for steps in steps_arr:
                self.steps = steps
                for skip_type in skip_arr:
                    self.skip_type = skip_type
                    self.sample()
                    f_path = f"dpm_alphaBar_{order}-{steps:03d}-{skip_type}.txt"
                    f_path = os.path.join(ab_dir, f_path)
                    save_ab_file(f_path)
                    logging.info(f"File saved: {f_path}")
                # for
            # for
        # for

    def sample_times(self, times: int):
        args = self.args
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
            order, steps, skip_type = self.order, self.steps, self.skip_type
            ss = args.predefined_aap_file if skip_type == 'predefined' else f"{order}-{steps}-{skip_type}"
            logging.info(f"{ss}-{i} => FID: {fid:.6f}")
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
        logging.info(f"DiffusionDpmSolver::sample(self, {type(model).__name__})...")
        logging.info(f"  sample_output_dir      : {args.sample_output_dir}")
        logging.info(f"  sample_count           : {self.sample_count}")
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
        dpm_solver = self.create_dpm_solver(model, self.device)
        with torch.no_grad():
            for r_idx in range(n_rounds):
                n = b_sz if r_idx + 1 < n_rounds else self.sample_count - r_idx * b_sz
                logging.info(f"DiffusionDpmSolver::round: {r_idx}/{n_rounds}. to generate: {n}")
                x_t = torch.randn(n, d.channels, d.image_size, d.image_size, device=self.device)
                x = self.sample_by_dpm_solver(x_t, dpm_solver)
                self.save_images(config, x, time_start, r_idx, n_rounds, b_sz)
            # for r_idx
        # with

    @staticmethod
    def load_predefined_aap(f_path: str, meta_dict=None):
        if not os.path.exists(f_path):
            raise Exception(f"File not found: {f_path}")
        if not os.path.isfile(f_path):
            raise Exception(f"Not file: {f_path}")
        logging.info(f"  Load file: {f_path}")
        with open(f_path, 'r') as f_ptr:
            lines = f_ptr.readlines()
        cnt_empty = 0
        cnt_comment = 0
        f_arr = []  # float array
        i_arr = []  # int array
        for line in lines:
            line = line.strip()
            if line == '':
                cnt_empty += 1
                continue
            if line.startswith('#'):  # line is like "# order     : 2"
                cnt_comment += 1
                arr = line[1:].strip().split(':')
                key = arr[0].strip()
                if key in meta_dict: meta_dict[key] = arr[1].strip()
                continue
            arr = line.split(':')
            flt, itg = float(arr[0]), int(arr[1])
            f_arr.append(flt)
            i_arr.append(itg)
        f2s = lambda ff: ' '.join([f"{f:8.6f}" for f in ff])
        i2s = lambda ii: ' '.join([f"{i: 8d}" for i in ii])
        logging.info(f"    cnt_empty  : {cnt_empty}")
        logging.info(f"    cnt_comment: {cnt_comment}")
        logging.info(f"    cnt_valid  : {len(f_arr)}")
        logging.info(f"    float[:5]  : [{f2s(f_arr[:5])}]")
        logging.info(f"    float[-5:] : [{f2s(f_arr[-5:])}]")
        logging.info(f"    int[:5]    : [{i2s(i_arr[:5])}]")
        logging.info(f"    int[-5:]   : [{i2s(i_arr[-5:])}]")
        return f_arr, i_arr

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
        if aap_file and aap_file.startswith('geometric_ratio:'):
            ratio = aap_file.split(':')[1]
            ratio = float(ratio)
            series = utils.create_geometric_series(0., 999., ratio, 21)
            series = series[1:]  # ignore the first element
            ts = [int(f) for f in series]
            i_arr_tensor = torch.tensor(ts, device=self.alphas_cumprod.device)
            f_arr_tensor = self.alphas_cumprod.index_select(0, i_arr_tensor)
            aap = f_arr_tensor.tolist()
            self.order = 1
            self.steps = len(aap)
            self.skip_type = 'predefined'  # hard code here.
            noise_schedule = NoiseScheduleVP2(schedule='predefined',
                                              alphas_cumprod=self.alphas_cumprod,
                                              predefined_ts=ts,
                                              predefined_aap=aap)
            noise_schedule.to(device)
        elif aap_file:
            meta_dict = {'order': '', 'steps': ''}
            aap, ts = self.load_predefined_aap(aap_file, meta_dict)
            if meta_dict['steps'] != f"{len(aap)}":
                raise Exception(f"steps not match between comment and real data: {aap_file}."
                                f" {meta_dict['steps']} != {len(aap)}")
            if self.args.dpm_order:
                self.order = self.args.dpm_order
            else:
                if meta_dict['order'] == '': raise Exception(f"Not found order from: {aap_file}")
                self.order = int(meta_dict['order'])
            self.steps = len(aap)
            self.skip_type = 'predefined'  # hard code here.
            noise_schedule = NoiseScheduleVP2(schedule='predefined',
                                              alphas_cumprod=self.alphas_cumprod,
                                              predefined_ts=ts,
                                              predefined_aap=aap)
            noise_schedule.to(device)
        else:
            noise_schedule = NoiseScheduleVP2(schedule='discrete', alphas_cumprod=self.alphas_cumprod)
        if self.alpha_bar_all_flag:
            noise_schedule.alpha_bar_map = {}  # only init it for specific args.

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
            dpm_solver = DPM_Solver(model_fn, noise_schedule, algorithm_type="dpmsolver",
                                    skip_type="predefined",
                                    use_predefined_ts=self.args.use_predefined_ts,
                                    ts_int_flag=self.args.ts_int_flag)
        else:
            dpm_solver = DPM_Solver(model_fn, noise_schedule, algorithm_type="dpmsolver", skip_type=self.skip_type)

        # Can also try
        # dpm_solver = DPM_Solver(model_fn, noise_schedule, algorithm_type="dpmsolver++")

        self.dpm_solver = dpm_solver
        self.noise_schedule = noise_schedule
        return dpm_solver

    def sample_by_dpm_solver(self, x_T, dpm_solver: DPM_Solver):
        # You can use steps = 10, 12, 15, 20, 25, 50, 100.
        # Empirically, we find that steps in [10, 20] can generate quite good samples.
        # And steps = 20 can almost converge.
        x_sample = dpm_solver.sample(
            x_T,
            steps=self.steps,
            # skip_type="time_uniform",
            order=self.order,
            method="singlestep",
            t_start=None,
            t_end=None,
        )
        return x_sample

# class
