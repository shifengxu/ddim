import os
import random
import time
import datetime
import logging
import numpy as np
import torch
import utils
from datasets import inverse_data_transform
from models.diffusion import Model
from runners.diffusion import Diffusion
from models.dpm_solver2_pytorch import model_wrapper, DPM_Solver
from models.noise_schedule2 import NoiseScheduleVP2

import torchvision.utils as tvu


class DiffusionDpmSolverPredefinedTrajectory(Diffusion):
    def __init__(self, args, config, device=None):
        super().__init__(args, config, device, output_ab=False)
        self.b_sz = args.sample_batch_size or config.sampling.batch_size
        self.dpm_solver = None
        self.noise_schedule = None
        self.t_start = None
        self.t_end = None
        self.order = None
        self.steps = None
        self.time_start = None
        self.batch_total = None
        self.batch_cnt = None

    def save_images(self, config, x, b_idx):
        x = inverse_data_transform(config, x)
        img_cnt = len(x)
        elp, eta = utils.get_time_ttl_and_eta(self.time_start, self.batch_cnt, self.batch_total)
        img_dir = self.args.sample_output_dir
        img_path = None
        for i in range(img_cnt):
            img_id = b_idx * self.b_sz + i
            img_path = os.path.join(img_dir, f"{img_id:05d}.png")
            tvu.save_image(x[i], img_path)
        logging.info(f"  saved {img_cnt} images: {img_path}. elp:{elp}, eta:{eta}")

    @staticmethod
    def load_trajectories(f_path: str, meta_dict: dict):
        """
        The file content is like this:
            # Old comments in file ../exp/dpm_alphaBar/dpm_alphaBar_1-010-logSNR.txt
            # order     : 1
            # steps     : 10
            # skip_type : logSNR
            # data_type : alpha_bar
            #
            # Old alpha_bar and its timestep
            # 0.99930942    3
            # 0.99524850   16
            # 0.96806973   51
            # 0.81441778  138
            # 0.38845780  302
            # 0.08420218  492
            # 0.01313375  652
            # 0.00192265  784
            # 0.00027875  898
            # 0.00004036  998
            #
            # lr           : 1e-06
            # n_epochs     : 10000
            # aa_low       : 0.0001
            # aa_low_lambda: 10000000.0
            # beta_schedule: linear
            # torch.seed() : 8079376260256798063
            # alpha_bar_dir: ../exp/dpm_alphaBar
            # Epoch        : 009999; loss:1290.058856 = 1290.040570 + 0.018285
            # loss_var     : 1428.550366 => 1290.040570
            # model.learning_portion: 0.01
            # model.out_channels    : 10
            # aacum : ts : alpha   ; coef    *weight     =numerator; numerator/aacum   =sub_var
            0.998924:   5: 0.998924; 0.001076*1066.710363= 1.147857;  1.147857/0.998924=  1.149093
            0.995163:  17: 0.996235; 0.001355* 705.650521= 0.955947;  0.955947/0.995163=  0.960593
            0.977908:  42: 0.982662; 0.006350* 430.247043= 2.732271;  2.732271/0.977908=  2.793994
            0.832462: 130: 0.851268; 0.074082* 186.999852=13.853293; 13.853293/0.832462= 16.641353
            0.405384: 295: 0.486970; 0.235691*  75.051656=17.689022; 17.689022/0.405384= 43.635188
            0.091924: 483: 0.226758; 0.343083*  25.851633= 8.869268;  8.869268/0.091924= 96.484813
            0.015257: 641: 0.165977; 0.364956*   7.152960= 2.610514;  2.610514/0.015257=171.099803
            0.002386: 771: 0.156388; 0.367691*   1.577057= 0.579870;  0.579870/0.002386=243.024861
            0.000370: 882: 0.154980; 0.367976*   0.293589= 0.108034;  0.108034/0.000370=292.149930
            0.000057: 981: 0.154787; 0.367981*   0.065657= 0.024160;  0.024160/0.000057=422.100943
        The file content ended.
        :param f_path:
        :param meta_dict:
        :return:
        """
        logging.info(f"  Load file: {f_path}...")
        with open(f_path, 'r') as f_ptr:
            lines = f_ptr.readlines()
        ab_ori_arr = []  # alpha_bar array of original trajectory
        ts_ori_arr = []  # timestep array of original trajectory
        ab_sch_arr = []  # alpha_bar array of scheduled trajectory
        ts_sch_arr = []  # timestep array of scheduled trajectory
        in_ori_ab = False   # in original alpha_bar lines
        for line in lines:
            line = line.strip()
            if line == '':
                continue
            if line == '# Old alpha_bar and its timestep':
                in_ori_ab = True
                continue
            if in_ori_ab and line == '#':
                in_ori_ab = False
                continue
            if in_ori_ab:   # "# 0.99973720    1"
                line = line[1:].strip()
                arr = line.split()
                ab, ts = float(arr[0]), int(arr[1])
                ab_ori_arr.append(ab)
                ts_ori_arr.append(ts)
                continue
            if line.startswith('#'):  # line is like "# order     : 2"
                arr = line[1:].strip().split(':')
                key = arr[0].strip()
                if key in meta_dict: meta_dict[key] = arr[1].strip()
                continue
            arr = line.split(':')
            ab, ts = float(arr[0]), int(arr[1])
            ab_sch_arr.append(ab)
            ts_sch_arr.append(ts)
        ab2s = lambda ff: ' '.join([f"{f:10.8f}" for f in ff])
        ts2s = lambda ff: ' '.join([f"{f:10d}" for f in ff])
        logging.info(f"    ab_ori count : {len(ab_ori_arr)} ---")
        logging.info(f"    ab_ori[:5]   : [{ab2s(ab_ori_arr[:5])}]")
        logging.info(f"    ab_ori[-5:]  : [{ab2s(ab_ori_arr[-5:])}]")
        logging.info(f"    ts_ori[:5]   : [{ts2s(ts_ori_arr[:5])}]")
        logging.info(f"    ts_ori[-5:]  : [{ts2s(ts_ori_arr[-5:])}]")
        logging.info(f"    ab_sch count : {len(ab_sch_arr)} ---")
        logging.info(f"    ab_sch[:5]   : [{ab2s(ab_sch_arr[:5])}]")
        logging.info(f"    ab_sch[-5:]  : [{ab2s(ab_sch_arr[-5:])}]")
        logging.info(f"    ts_sch[:5]   : [{ts2s(ts_sch_arr[:5])}]")
        logging.info(f"    ts_sch[-5:]  : [{ts2s(ts_sch_arr[-5:])}]")
        logging.info(f"  Load file: {f_path}...Done")
        return ab_ori_arr, ts_ori_arr, ab_sch_arr, ts_sch_arr

    def sample_original_and_scheduled(self):
        """
        Sample with original and scheduled trajectories.
        And both trajectories are from text file.
        :return:
        """
        args = self.args
        logging.info(f"DiffusionDpmSolverPredefinedTrajectory::sample_original_and_scheduled()...")
        sch_dir = args.ab_scheduled_dir
        logging.info(f"  sch_dir: {sch_dir}")
        file_list = [f for f in os.listdir(sch_dir) if f.endswith('.txt')]
        file_list.sort()
        file_cnt = len(file_list)
        logging.info(f"  sch_dir: file count {file_cnt}")
        [logging.info(f"           {f}") for f in file_list]
        file_list = [os.path.join(sch_dir, f) for f in file_list]

        self.batch_cnt = 0
        rounds = args.sample_count // self.b_sz
        if rounds * self.b_sz > args.sample_count:
            rounds += 1
        self.batch_total = file_cnt * rounds * 2 # multiply by 2, for both original and scheduled trajectories.
        logging.info(f"  batch_total: {self.batch_total}")
        logging.info(f"  batch_cnt  : {self.batch_cnt}")

        res_file = "./sample_original_and_scheduled.txt"
        res_arr = [
            f"pid : {os.getpid()}",
            f"cwd : {os.getcwd()}",
            f"host: {os.uname().nodename}",
        ]
        def append_res(res_msg=None, flush=True):
            if res_msg is not None:
                dt_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                res_arr.append(f"[{dt_str}] {res_msg}")
            if flush:
                with open(res_file, 'w') as f_ptr:
                    [f_ptr.write(f"{res}\n") for res in res_arr]
        # append_res()

        append_res('')
        gpu = args.gpu_ids[0]
        input1 = args.sample_output_dir
        for f_idx, sch_file_path in enumerate(file_list):
            logging.info(f"{f_idx:2d}/{file_cnt}: {sch_file_path}")
            args.predefined_aap_file = sch_file_path
            meta_dict = {'order': '', 'steps': '', 'skip_type': ''}
            ab_ori, ts_ori, ab_sch, ts_sch = self.load_trajectories(sch_file_path, meta_dict)
            noise_schedule = self.create_ns_by_aap_file(ab_ori, ts_ori, meta_dict)
            self.sample(noise_schedule=noise_schedule)
            logging.info(f"{sch_file_path} Original ==> ISC calculating... {input1}")
            isc_mean, isc_std = utils.calc_isc(gpu, input1)
            logging.info(f"{sch_file_path} Original ==> ISC: isc_mean:{isc_mean:9.6f}, isc_std:{isc_std:9.6f}")
            append_res(f"isc_mean:{isc_mean:9.6f}, isc_std:{isc_std:9.6f}. {sch_file_path} Original")

            noise_schedule = self.create_ns_by_aap_file(ab_sch, ts_sch, meta_dict)
            self.sample(noise_schedule=noise_schedule)
            logging.info(f"{sch_file_path} Scheduled ==> ISC calculating... {input1}")
            isc_mean, isc_std = utils.calc_isc(gpu, input1)
            logging.info(f"{sch_file_path} Scheduled ==> ISC: isc_mean:{isc_mean:9.6f}, isc_std:{isc_std:9.6f}")
            append_res(f"isc_mean:{isc_mean:9.6f}, isc_std:{isc_std:9.6f}. {sch_file_path} Scheduled")
        # for
        logging.info(f"DiffusionDpmSolverPredefinedTrajectory::sample_original_and_scheduled()...Done")

    def create_ns_by_aap_file(self, ab_arr, ts_arr, meta_dict):
        args = self.args
        aap_file = args.predefined_aap_file
        if args.dpm_order:
            self.order = args.dpm_order
        else:
            if meta_dict['order'] == '': raise Exception(f"Not found order from: {aap_file}")
            self.order = int(meta_dict['order'])
        self.steps = len(ab_arr)
        logging.info(f"create_ns_by_aap_file()...")
        logging.info(f"  self.order        : {self.order}")
        logging.info(f"  self.steps        : {self.steps}")
        logging.info(f"  args.ty_type      : {args.ts_type}")
        logging.info(f"  args.beta_schedule: {args.beta_schedule}")
        if args.ts_type == 'discrete' and args.beta_schedule == 'linear':
            ab_arr.insert(0, 0.9999)
            ts_arr.insert(0, 0)
            logging.info(f"  Insert aap[0]: {ab_arr[0]:.5f}")
            logging.info(f"  Insert ts[0] : {ts_arr[0]}")
        else:
            raise ValueError(f"Not support: ts_type={args.ts_type}, beta_schedule={args.beta_schedule}")
        ns = NoiseScheduleVP2(schedule='predefined',
                              alphas_cumprod=self.alphas_cumprod,
                              predefined_ts=ts_arr,
                              predefined_aap=ab_arr)
        ns.to(self.device)
        logging.info(f"create_ns_by_aap_file()...Done")
        return ns

    def sample(self, sample_count=None, noise_schedule=None):
        config, args = self.config, self.args
        model = Model(config, ts_type=args.ts_type)
        model = self.model_load_from_local(model)
        model.eval()

        sample_count = sample_count or args.sample_count
        logging.info(f"DiffusionDpmSolverPredefinedTrajectory::sample(self, {type(model).__name__})...")
        logging.info(f"  sample_output_dir      : {args.sample_output_dir}")
        logging.info(f"  sample_count           : {sample_count}")
        n_rounds = (sample_count - 1) // self.b_sz + 1  # get the ceiling
        logging.info(f"  batch_size             : {self.b_sz}")
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

        self.time_start = time.time()
        d = config.data
        dpm_solver = self.create_dpm_solver(model, noise_schedule)
        with torch.no_grad():
            for r_idx in range(n_rounds):
                self.batch_cnt += 1
                n = self.b_sz if r_idx + 1 < n_rounds else sample_count - r_idx * self.b_sz
                x_t = torch.randn(n, d.channels, d.image_size, d.image_size, device=self.device)
                x = self.sample_by_dpm_solver(x_t, dpm_solver, r_idx, return_intermediate=False)
                self.save_images(config, x, r_idx)
            # for r_idx
        # with

    def create_dpm_solver(self, model, noise_schedule):
        args = self.args

        # 1, create noise schedule, should be in other method.
        if noise_schedule is None:
            logging.info(f"create_dpm_solver() noise_schedule is None. Will create it.")
            meta_dict = {'order': '', 'steps': ''}
            ab_ori, ts_ori, ab_sch, ts_sch = self.load_trajectories(args.ab_scheduled_dir, meta_dict)
            noise_schedule = self.create_ns_by_aap_file(ab_sch, ts_sch, meta_dict)

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
        dpm_solver = DPM_Solver(model_fn, noise_schedule, algorithm_type="dpmsolver",
                                skip_type="predefined",
                                use_predefined_ts=args.use_predefined_ts,
                                ts_type=args.ts_type,
                                )

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
