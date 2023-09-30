import datetime
import os
import random
import time
import logging

import numpy as np
import torch
import torch_fidelity
import subprocess
import re

import utils
from datasets import inverse_data_transform
from models.diffusion import Model
from runners.diffusion import Diffusion
from models.dpm_solver2_pytorch import model_wrapper, DPM_Solver
from models.noise_schedule2 import NoiseScheduleVP2

import torchvision.utils as tvu


class DiffusionDpmSolver(Diffusion):
    def __init__(self, args, config, order=1, steps=20, skip_type='time_uniform', device=None):
        super().__init__(args, config, device, output_ab=False)
        self.sample_count = 50000    # sample image count
        self.dpm_solver = None
        self.noise_schedule = None
        self.t_start = None
        self.t_end = None
        self.order = order
        self.steps = steps
        self.skip_type = skip_type
        self.fid_subprocess = args.fid_subprocess if hasattr(args, 'fid_subprocess') else False

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

    def save_images_intermediate(self, config, im_arr, time_start, r_idx, n_rounds, b_sz):
        img_cnt = len(im_arr[0])
        int_cnt = len(im_arr)  # intermediate count
        if int_cnt == 10:
            # only choose 5 intermediate images for bedroom
            batch_arr = [inverse_data_transform(config, im) for im in im_arr[1::2]]
        else:
            batch_arr = [inverse_data_transform(config, im) for im in im_arr]
        elp, eta = utils.get_time_ttl_and_eta(time_start, r_idx+1, n_rounds)
        img_dir = self.args.sample_output_dir
        img_path = None
        for i in range(img_cnt):
            img_id = r_idx * b_sz + i
            img_path = os.path.join(img_dir, f"{img_id:05d}.png")
            x = [batch[i] for batch in batch_arr]
            x.reverse()
            tvu.save_image(x, img_path, nrow=10, padding=1)
        logging.info(f"  saved {img_cnt} intermediate images: {img_path}. elp:{elp}, eta:{eta}")

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

    def sample_all(self, order_arr=None, steps_arr=None, skip_arr=None, times=5):
        def save_result(_msg_arr, _fid_arr):
            with open('./sample_all_result.txt', 'w') as f_ptr:
                [f_ptr.write(f"# {m}\n") for m in _msg_arr]
                [f_ptr.write(f"[{dt}] {avg:8.4f} {std:.4f}: {k}\n") for dt, avg, std, k in _fid_arr]
            # with
        # end of inner def
        args = self.args
        args.predefined_aap_file = ""
        logging.info(f"DiffusionDpmSolver::sample_all()")
        logging.info(f"  args.predefined_aap_file: '{args.predefined_aap_file}'")
        order_arr = order_arr or [1, 2, 3]
        steps_arr = steps_arr or [10, 15, 20, 25, 50, 100]
        skip_arr  = skip_arr or ['time_uniform', 'logSNR', 'time_quadratic']
        msg_arr = [f"order_arr : {order_arr}",
                   f"steps_arr : {steps_arr}",
                   f"skip_arr  : {skip_arr}",
                   f"times     : {times}",
                   f"fid_input1: {args.fid_input1}"]
        [logging.info(f"  {m}") for m in msg_arr]
        fid_arr = []
        for steps in steps_arr:
            self.steps = steps
            for order in order_arr:
                self.order = order
                for skip_type in skip_arr:
                    self.skip_type = skip_type
                    fid_avg, fid_std = self.sample_times(times)
                    dtstr = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                    key = f"{order}-{steps}-{skip_type}"
                    fid_arr.append([dtstr, fid_avg, fid_std, key])
                    logging.info(f"fid_avg: {key}: {fid_avg:8.5f}, fid_std:{fid_std:.5f}")
                    save_result(msg_arr, fid_arr)
                # for
            # for
        # for
        logging.info(f"fid_list:")
        [logging.info(f"{fid:8.5f}: {key}") for dtstr, fid, key in fid_arr]

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
            fid_avg, fid_std = self.sample_times(times)
            fid_dict[f_path] = (fid_avg, fid_std)
            logging.info(f"fid_avg: {f_path}: {fid_avg:8.5f}; fid_std:{fid_std:.5f}")
            f_path = os.path.join(sum_dir, 'sample_all_scheduled_fid.txt')
            logging.info(f"Save file: {f_path}")
            with open(f_path, 'w') as f_ptr:
                for key in sorted(fid_dict):
                    avg, std = fid_dict[key]
                    f_ptr.write(f"{avg:9.5f} {std:.5f}: {key}\n")
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
                f_ptr.write(f"# alpha_bar : timestep\n")
                [f_ptr.write(f"{ab_map[k][1]}  : {ab_map[k][0]}\n") for k in ab_list]
            # with
        # def
        args = self.args
        args.predefined_aap_file = ""
        logging.info(f"DiffusionDpmSolver::alpha_bar_all()")
        logging.info(f"  args.predefined_aap_file: '{args.predefined_aap_file}'")
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
                    self.sample(sample_count=1)
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
            order, steps, skip_type = self.order, self.steps, self.skip_type
            ss = args.predefined_aap_file if skip_type == 'predefined' else f"{order}-{steps}-{skip_type}"
            input1, input2 = args.fid_input1 or 'cifar10-train', args.sample_output_dir
            logging.info(f"{ss}-{i} => FID calculating...")
            logging.info(f"  input1: {input1}")
            logging.info(f"  input2: {input2}")
            if args.fid_subprocess:
                gpu = args.gpu_ids[0]
                cmd = f"fidelity --gpu {gpu} --fid --input1 {input1} --input2 {input2} --silent"
                logging.info(f"cmd: {cmd}")
                cmd_arr = cmd.split(' ')
                res = subprocess.run(cmd_arr, stdout=subprocess.PIPE)
                output = str(res.stdout)
                logging.info(f"out: {output}") # frechet_inception_distance: 16.5485\n
                m = re.search(r'frechet_inception_distance: (\d+\.\d+)', output)
                fid = float(m.group(1))
            else:
                metrics_dict = torch_fidelity.calculate_metrics(
                    input1=input1,
                    input2=input2,
                    cuda=True,
                    isc=False,
                    fid=True,
                    kid=False,
                    verbose=False,
                    samples_find_deep=True,
                )
                fid = metrics_dict['frechet_inception_distance']
            logging.info(f"{ss}-{i} => FID: {fid:.6f}")
            fid_arr.append(fid)
        # for
        avg, std = np.mean(fid_arr), np.std(fid_arr)
        return avg, std

    def sample(self, sample_count=None):
        config, args = self.config, self.args
        model = Model(config, ts_type=args.ts_type)
        model = self.model_load_from_local(model)
        model.eval()

        self.sample_count = sample_count or args.sample_count
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
        # set random seed
        seed = args.seed  # if seed is 0. then ignore it.
        if seed:
            # set seed before generating sample. Make sure use same seed to generate.
            logging.info(f"  args.seed: {seed}")
            logging.info(f"  torch.manual_seed({seed})")
            logging.info(f"  np.random.seed({seed})")
            logging.info(f"  random.seed({seed})")
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)
        if seed and torch.cuda.is_available():
            logging.info(f"  torch.cuda.manual_seed({seed})")
            logging.info(f"  torch.cuda.manual_seed_all({seed})")
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        logging.info(f"  final seed: torch.initial_seed(): {torch.initial_seed()}")

        time_start = time.time()
        d = config.data
        dpm_solver = self.create_dpm_solver(model, self.device)
        ri = False  # return_intermediate. hard-code temporarily
        with torch.no_grad():
            for r_idx in range(n_rounds):
                n = b_sz if r_idx + 1 < n_rounds else self.sample_count - r_idx * b_sz
                logging.info(f"DiffusionDpmSolver::round: {r_idx}/{n_rounds}. to generate: {n}")
                x_t = torch.randn(n, d.channels, d.image_size, d.image_size, device=self.device)
                x = self.sample_by_dpm_solver(x_t, dpm_solver, r_idx, return_intermediate=ri)
                if ri:
                    x, im_arr = x
                    self.save_images_intermediate(config, im_arr, time_start, r_idx, n_rounds, b_sz)
                else:
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
        ab_arr = []  # alpha_bar array
        ts_arr = []  # timestep array
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
            ab, ts = float(arr[0]), float(arr[1])
            ab_arr.append(ab)
            ts_arr.append(ts)
        ab2s = lambda ff: ' '.join([f"{f:8.6f}" for f in ff])
        ts2s = lambda ff: ' '.join([f"{f:10.5f}" for f in ff])
        logging.info(f"    cnt_empty  : {cnt_empty}")
        logging.info(f"    cnt_comment: {cnt_comment}")
        logging.info(f"    cnt_valid  : {len(ab_arr)}")
        logging.info(f"    ab[:5]     : [{ab2s(ab_arr[:5])}]")
        logging.info(f"    ab[-5:]    : [{ab2s(ab_arr[-5:])}]")
        logging.info(f"    ts[:5]     : [{ts2s(ts_arr[:5])}]")
        logging.info(f"    ts[-5:]    : [{ts2s(ts_arr[-5:])}]")
        return ab_arr, ts_arr

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
        def create_ns_by_aap_geometric_ratio():
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
            ns = NoiseScheduleVP2(schedule='predefined',
                                  alphas_cumprod=self.alphas_cumprod,
                                  predefined_ts=ts,
                                  predefined_aap=aap)
            return ns

        def create_ns_by_aap_file():
            meta_dict = {'order': '', 'steps': ''}
            aap, ts = self.load_predefined_aap(aap_file, meta_dict)
            if meta_dict['steps'] != f"{len(aap)}":
                raise Exception(f"steps not match between comment and real data: {aap_file}."
                                f" {meta_dict['steps']} != {len(aap)}")
            if args.dpm_order:
                self.order = args.dpm_order
            else:
                if meta_dict['order'] == '': raise Exception(f"Not found order from: {aap_file}")
                self.order = int(meta_dict['order'])
            self.steps = len(aap)
            self.skip_type = 'predefined'  # hard code here.
            logging.info(f"  args.ty_type      : {args.ts_type}")
            logging.info(f"  args.beta_schedule: {args.beta_schedule}")
            if args.ts_type == 'discrete' and args.beta_schedule == 'linear':
                aap.insert(0, 0.9999)
                ts.insert(0, 0)
                logging.info(f"  Insert aap[0]: {aap[0]:.5f}")
                logging.info(f"  Insert ts[0] : {ts[0]}")
            elif args.ts_type == 'continuous' and args.beta_schedule == 'cosine':
                # If cosine schedule and continuous timestep. the init timestep is 0.001.
                # And its corresponding alpha_bar value is 0.99995869
                # see the DPM_Solver::sample() function
                aap.insert(0, 0.99995869)
                ts.insert(0, 0.001)
                logging.info(f"  Insert aap[0]: {aap[0]:.8f}")
                logging.info(f"  Insert ts[0] : {ts[0]:.5f}")
            else:
                raise ValueError(f"Not support: ts_type={args.ts_type}, beta_schedule={args.beta_schedule}")
            ns = NoiseScheduleVP2(schedule='predefined',
                                  alphas_cumprod=self.alphas_cumprod,
                                  predefined_ts=ts,
                                  predefined_aap=aap)
            return ns

        args = self.args
        aap_file = args.predefined_aap_file
        if aap_file and aap_file.startswith('geometric_ratio:'):
            noise_schedule = create_ns_by_aap_geometric_ratio()
            noise_schedule.to(device)
        elif aap_file:
            noise_schedule = create_ns_by_aap_file()
            noise_schedule.to(device)
        else:
            sch = args.noise_schedule
            noise_schedule = NoiseScheduleVP2(schedule=sch, alphas_cumprod=self.alphas_cumprod)

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
                                    use_predefined_ts=args.use_predefined_ts,
                                    ts_type=args.ts_type,
                                    )
        else:
            dpm_solver = DPM_Solver(model_fn, noise_schedule, algorithm_type="dpmsolver", skip_type=self.skip_type)

        # Can also try
        # dpm_solver = DPM_Solver(model_fn, noise_schedule, algorithm_type="dpmsolver++")

        self.dpm_solver = dpm_solver
        self.noise_schedule = noise_schedule
        return dpm_solver

    def sample_by_dpm_solver(self, x_T, dpm_solver: DPM_Solver, batch_idx, return_intermediate=False):
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
            batch_idx=batch_idx,
            return_intermediate=return_intermediate,
        )
        if batch_idx == 0:
            utils.onetime_log_flush()
        return x_sample

# class
