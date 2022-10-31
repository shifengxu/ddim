import argparse
import os
import time
import unittest

import numpy as np
import torch
import yaml

from runners.diffusion_latent_sampling import DiffusionLatentSampling
from utils import dict2namespace


class DiffusionLatentSamplingTest(unittest.TestCase):

    def test_sample_fid_vanilla(self):
        parser = argparse.ArgumentParser(description=globals()["__doc__"])
        args = parser.parse_args()
        args.config = './configs/ffhq_latent.yml'
        args.sample_output_dir = './unit_test_output_dir'
        args.beta_schedule = 'linear'
        args.beta_cos_expo = 2
        args.ts_range = [0, 1000]
        args.sample_type = 'generalized'
        args.skip_type = 'uniform'
        args.timesteps = 1000
        print(f"cwd: {os.getcwd()}")
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)
        config = dict2namespace(config)
        runner = DiffusionLatentSampling(args, config, device='cpu')
        b_sz = 2
        x_t = torch.randn(b_sz, 3, 32, 32, device='cpu',)
        model = None
        time_start = time.time()
        runner.num_timesteps = 0  # purely for testing
        runner.sample_fid_vanilla(x_t, model, time_start, 10, 0, b_sz)
        for i in range(b_sz):
            f_path = os.path.join(args.sample_output_dir, '00000', f"{i:05d}.npy")
            print(f"validate: {f_path}")
            tmp = np.load(f_path)
            self.assertEqual((x_t[i].numpy() == tmp).all(), True)
        # for

if __name__ == '__main__':
    unittest.main()