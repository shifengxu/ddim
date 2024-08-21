"""
Focus on the cumulative prediction error of the sampling process.
2024-08-18: handle predefined trajectory from AutoDiffusion.
    ICCV 2023: AutoDiffusion: Training-free optimization of time steps and architectures
                for automated diffusion model acceleration
    ICCV 2023: And its appendix
    Optimal timesteps: [926, 690, 424, 153]
    Optimal timesteps: [123, 207, 390, 622, 830, 948]
"""

import os
import time

import numpy as np
import torch
import torchvision.utils as tvu
from datasets import inverse_data_transform
from models.diffusion import Model
from runners.diffusion import Diffusion
from schedule.scheduler_vrg.vrg_scheduler import VrgScheduler
from utils import log_info, get_time_ttl_and_eta, calc_fid


class DiffusionCumulativePredictionError(Diffusion):
    def __init__(self, args, config):
        log_info(f"DiffusionCumulativePredictionError::__init__()...")
        super().__init__(args, config, args.device, output_ab=False)
        self.scheduler = VrgScheduler(self.args, self.alphas_cumprod)
        self.model = None
        self.hard_steps = 200  # hard-code ground-truth image steps. Feel free to change it.
        log_info(f"DiffusionCumulativePredictionError::__init__()...Done")

    def sample_and_differ(self):
        self.sample()
        self.differ_image_sets_by_l2()

    def sample(self):
        args, config = self.args, self.config
        model = Model(config)
        model = self.model_load_from_local(model)
        model.eval()
        self.model = model
        sample_count = args.sample_count
        steps_arr = args.steps_arr
        ori_trajectory = args.ts_range  # re-use the arg to input original trajectory
        if ori_trajectory is not None and len(ori_trajectory) > 0 and ori_trajectory[0] > ori_trajectory[-1]:
            # make sure the order is ascent
            ori_trajectory = list(reversed(ori_trajectory))
        ch, h, w = config.data.channels, config.data.image_size, config.data.image_size
        b_sz = args.sample_batch_size      # batch size
        b_cnt = (sample_count - 1) // b_sz + 1  # batch count
        log_info(f"DiffusionCumulativePredictionError::sample()...")
        log_info(f"  config            : {args.config}")
        log_info(f"  model             : {type(model).__name__}")
        log_info(f"  steps_arr         : {steps_arr}")
        log_info(f"  ori_trajectory    : {ori_trajectory}")
        log_info(f"  total_timesteps   : {self.num_timesteps}")
        log_info(f"  sample_count      : {sample_count}")
        log_info(f"  batch_size        : {b_sz}")
        log_info(f"  b_cnt             : {b_cnt}")
        log_info(f"  channel           : {ch}")
        log_info(f"  height            : {h}")
        log_info(f"  width             : {w}")
        if ori_trajectory is None or len(ori_trajectory) == 0:
            new_steps_arr, seq_arr, img_dir_arr = self.get_seq_arr_and_dir_arr(steps_arr)
        else:
            new_steps_arr, seq_arr, img_dir_arr = self.get_seq_arr_and_dir_arr_from_trajectory(ori_trajectory)
        for steps, seq, img_dir in zip(new_steps_arr, seq_arr, img_dir_arr):
            log_info(f"steps:{steps:4d} -> {img_dir}")
            log_info(f"  seq len: {len(seq)}")
            log_info(f"  seq[0] : {seq[0]}")
            log_info(f"  seq[1] : {seq[1]}")
            log_info(f"  seq[-2]: {seq[-2]}")
            log_info(f"  seq[-1]: {seq[-1]}")
        # for
        img_noise_dir = "./img_noise"
        os.makedirs(img_noise_dir, exist_ok=True)
        time_start = time.time()
        for b_idx in range(b_cnt):
            n = b_sz if b_idx + 1 < b_cnt else sample_count - b_idx * b_sz
            noise = torch.randn(n, ch, h, w, device=self.device)
            np.save(f"{img_noise_dir}/batch{b_idx:03d}_size{n}.npy", noise.cpu().numpy())
            for seq, img_dir in zip(seq_arr, img_dir_arr):
                self.sample_and_save_batch(noise, seq, b_idx, b_sz, img_dir)
            # for steps
            elp, eta = get_time_ttl_and_eta(time_start, b_idx + 1, b_cnt)
            log_info(f"Batch:{b_idx:2d}/{b_cnt}, elp:{elp}, eta:{eta}")
        # for b_idx
        log_info(f"DiffusionCumulativePredictionError::sample()...Done")

    def get_seq_arr_and_dir_arr(self, steps_arr):
        step_size = int(self.num_timesteps // self.hard_steps)
        if step_size < 1: step_size = 1
        img_dir = f"./img_steps{self.hard_steps:02d}"
        os.makedirs(img_dir, exist_ok=True)
        seq = list(range(0, self.num_timesteps, step_size))
        new_steps_arr, sequence_arr, img_dir_arr = [self.hard_steps], [seq], [img_dir]

        # track and schedule the 1000-step trajectory. We then can check its cumulative-prediction-error.
        old_tj_file = f"./img_trajectory_steps{self.hard_steps:02d}_original.txt"
        new_tj_file = f"./img_trajectory_steps{self.hard_steps:02d}_schedule.txt"
        self.track_current_trajectory(seq, old_tj_file)
        self.scheduler.schedule(f_path=old_tj_file, output_file=new_tj_file)

        # if num_timesteps=1000 and steps=2, then the step_size will be 500.
        # And the original logic will have seq: 0, 500. This is not good enough.
        # We apply delta = (step_size - 1) // 2, and each seq element plus delta.
        # Then the new seq will be: 249, 749.
        # Similarly, if steps is 10, the new seq will be: 49, 149, 249,..., 949.
        for steps in steps_arr:
            step_size = float(self.num_timesteps) / steps
            delta = (step_size - 1) // 2  # see the above explanation
            seq = [int(s * step_size + delta) for s in range(steps)]
            img_dir = f"./img_steps{steps:02d}_ori"
            os.makedirs(img_dir, exist_ok=True)
            new_steps_arr.append(steps)
            sequence_arr.append(seq)
            img_dir_arr.append(img_dir)
            old_tj_file = f"./img_trajectory_steps{steps:02d}_original.txt"
            new_tj_file = f"./img_trajectory_steps{steps:02d}_schedule.txt"
            self.track_current_trajectory(seq, old_tj_file)
            self.scheduler.schedule(f_path=old_tj_file, output_file=new_tj_file)
            ab_arr, ts_arr, _, _ = self.load_trajectory_file(new_tj_file)
            img_dir = f"./img_steps{steps:02d}_sch"
            os.makedirs(img_dir, exist_ok=True)
            new_steps_arr.append(steps)
            sequence_arr.append(ts_arr)
            img_dir_arr.append(img_dir)
        # for
        return new_steps_arr, sequence_arr, img_dir_arr

    def get_seq_arr_and_dir_arr_from_trajectory(self, trajectory):
        step_size = int(self.num_timesteps // self.hard_steps)
        if step_size < 1: step_size = 1
        img_dir = f"./img_steps{self.hard_steps:02d}"
        os.makedirs(img_dir, exist_ok=True)
        seq = list(range(0, self.num_timesteps, step_size))
        new_steps_arr, seq_arr, img_dir_arr = [self.hard_steps], [seq], [img_dir]

        seq = trajectory
        steps = len(seq)
        img_dir = f"./img_steps{steps:02d}_ori"
        os.makedirs(img_dir, exist_ok=True)
        new_steps_arr.append(steps)
        seq_arr.append(seq)
        img_dir_arr.append(img_dir)
        old_tj_file = f"./img_trajectory_steps{steps:02d}_original.txt"
        new_tj_file = f"./img_trajectory_steps{steps:02d}_schedule.txt"
        self.track_current_trajectory(seq, old_tj_file)
        self.scheduler.schedule(f_path=old_tj_file, output_file=new_tj_file)
        ab_arr, ts_arr, _, _ = self.load_trajectory_file(new_tj_file)
        img_dir = f"./img_steps{steps:02d}_sch"
        os.makedirs(img_dir, exist_ok=True)
        new_steps_arr.append(steps)
        seq_arr.append(ts_arr)
        img_dir_arr.append(img_dir)
        return new_steps_arr, seq_arr, img_dir_arr

    def track_current_trajectory(self, seq, trajectory_path):
        log_info(f"track_current_trajectory()...")
        log_info(f"  seq: {seq}")
        with open(trajectory_path, 'w') as fptr:
            fptr.write(f"# class      : {self.__class__.__name__}\n")
            fptr.write(f"# steps      : {len(seq)}\n")
            fptr.write(f"# steps_total: {self.num_timesteps}\n")
            fptr.write(f"# alpha_bar\t: timestep\n")
            for i, ts in enumerate(seq):
                ab = self.alphas_cumprod[ts]  # alpha_bar_t
                fptr.write(f"{ab:.8f}\t: {ts:3d}\n")
                log_info(f"  {i:02d}:  ab={ab:.8f}, ts={ts:3d}")
        # with
        log_info(f"  save file: {trajectory_path}")
        log_info(f"track_current_trajectory()...Done")

    @staticmethod
    def load_trajectory_file(f_path):
        log_info(f"load_trajectory_file()...")
        log_info(f"  {f_path}")
        with open(f_path, 'r') as f:
            lines = f.readlines()
        cnt_empty = 0
        # alpha_bar array, timestep array, string array, comment array
        ab_arr, ts_arr, str_arr, comment_arr = [], [], [], []
        for line in lines:
            line = line.strip()
            if line == '':
                cnt_empty += 1
            elif line.startswith('#'):
                comment_arr.append(line)
            else:   # 0.88970543  : 200   <<< alpha_bar : timestep
                arr = line.split(':')
                ab, ts = arr[0], arr[1]
                ab_arr.append(float(ab))
                ts_arr.append(int(ts))
                str_arr.append(line)
        # for
        log_info(f"  cnt_empty  : {cnt_empty}")
        log_info(f"  cnt_comment: {len(comment_arr)}")
        log_info(f"  cnt_valid  : {len(ab_arr)}")
        log_info(f"load_trajectory_file()...Done")
        return ab_arr, ts_arr, str_arr, comment_arr

    def sample_and_save_batch(self, noise, seq, b_idx, b_sz, img_dir):
        xt = noise
        ts_size = (len(xt), )
        seq_desc = list(reversed(seq))  # [999, 900, 800, ..., 200, 100]
        seq_des2 = seq_desc[1:] + [0]   # [900, 800, 700, ..., 100, 0]
        with torch.no_grad():
            for i, (t, q) in enumerate(zip(seq_desc, seq_des2)):
                ts = torch.full(ts_size, t, device=self.device)
                et = self.model(xt, ts)               # epsilon_t
                at = self.alphas_cumprod[t]
                aq = self.alphas_cumprod[q]
                mt = at / aq
                if b_idx == 0 and (i < 20 or i < 100 and i % 10 == 0 or i % 100 == 0):
                    # only log first batch, and i = 0, 1, 2, ... 20, 30, 40,..., 100, 200, 300, ...
                    log_info(f"sampling:i={i:3d}, ts:{t:3d}~{q:3d}, ab:{at:.8f}~{aq:.8f}, mt=:{mt:.8f}")
                xt_next = (xt - (1 - at).sqrt() * et) / mt.sqrt() + (1 - aq).sqrt() * et
                xt = xt_next
            # for
        # with
        x0 = xt
        x0 = inverse_data_transform(self.config, x0)
        img_cnt = len(x0)   # save batch
        img_path = None
        for i in range(img_cnt):
            img_id = b_idx * b_sz + i
            img_path = os.path.join(img_dir, f"{img_id:05d}.png")
            tvu.save_image(x0[i], img_path)
        log_info(f"Saved {img_cnt}: {img_path}")

    def differ_image_sets_by_l2(self):
        """
        Calculate L2 norm difference between image sets
        :return:
        """
        import torchvision.transforms as T
        from datasets import ImageDataset
        import torch.utils.data as tu_data
        import datetime

        log_info(f"DiffusionCumulativePredictionError::differ_image_sets_by_l2()...")
        args = self.args
        img_gt_dir = f"./img_steps{self.hard_steps:02d}"     # ground-truth image dir
        img_dir_arr = []
        ori_trajectory = args.ts_range
        if ori_trajectory is None or len(ori_trajectory) == 0:
            for steps in args.steps_arr:
                img_dir_arr.append(f"./img_steps{steps:02d}_ori")
                img_dir_arr.append(f"./img_steps{steps:02d}_sch")
            # for
        else:
            steps = len(ori_trajectory)
            img_dir_arr.append(f"./img_steps{steps:02d}_ori")
            img_dir_arr.append(f"./img_steps{steps:02d}_sch")

        log_info(f"ground-truth img dir: {img_gt_dir}")
        [log_info(f"  {i+1:2d}:  {s}") for i, s in enumerate(img_dir_arr)]

        def get_dataloader(_img_path):
            tf = T.Compose([T.ToTensor()])
            ds = ImageDataset(_img_path, classes=None, transform=tf)
            dl = tu_data.DataLoader(ds, args.sample_batch_size, shuffle=False, num_workers=4)
            return dl

        loader_gt = get_dataloader(img_gt_dir)
        res_arr = [f"# ground truth img dir: {img_gt_dir}", f"# delta : FID : img_dir"]
        res_file_path = "./img_dir_l2_difference.txt"
        dir_cnt = len(img_dir_arr)
        for d_idx, img_dir in enumerate(img_dir_arr):
            log_info(f"{d_idx:2d}/{dir_cnt}: {img_dir}")
            loader = get_dataloader(img_dir)
            delta_sum, delta_cnt = 0., 0
            for b_idx, ((img_gt, _), (img_on, _)) in enumerate(zip(loader_gt, loader)):
                img_gt = img_gt.to(self.device)     # image ground truth
                img_on = img_on.to(self.device)     # image online
                img_gt = (img_gt + 1.) / 2.
                img_on = (img_on + 1.) / 2.
                delta = torch.square(img_gt - img_on)
                delta = delta.mean()
                delta_sum += delta
                delta_cnt += 1
            # for batch
            delta_avg = delta_sum / delta_cnt
            fid = calc_fid(args.gpu_ids[0], args.fid_input1, img_dir)
            msg = f"{delta_avg:.8f}: {fid:8.4f}: {img_dir}"
            log_info(f"{msg}")
            res_arr.append(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}")
            with open(res_file_path, 'w') as fptr:
                [fptr.write(f"{s}\n") for s in res_arr]
            # with
        # for dir
        log_info(f"DiffusionCumulativePredictionError::differ_image_sets_by_l2()...Done")

# class
