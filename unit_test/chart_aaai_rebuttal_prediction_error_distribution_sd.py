import matplotlib.pyplot as plt
import numpy as np
import os

class ChartAaaiRebuttalPredictionErrorDistributionSd:
    """
    Prediction error is from Stable Diffusion (SD)
    """
    def __init__(self):
        self.tick_size = 30
        self.legend_size = 30
        self.font = {
            # 'family': 'serif',
            'color' : 'black',
            'weight': 'normal',
            'size'  : 30,
            'alpha' : 0.3,
        }


    def run(self):
        """ distribution of delta between predicted noise and ground truth noise """
        xy_label_size = 40
        predefined_bins = 100

        def read_floats_from_file(f):
            x_arr = []
            with open(f, 'r') as fptr:
                lines = fptr.readlines()
                for line in lines:
                    line = line.strip()
                    if line.startswith('#') or line == '': continue
                    f_tmp = float(line)
                    x_arr.append(f_tmp)
                # for
            # with
            return x_arr

        # ts_all = [0, 99, 199, 299, 399, 499, 599, 699, 799, 899, 999]
        ts_list_list = [
            [9, 99, 199],
            [399, 499, 599],
            [799, 899, 999],
        ]
        dim = 1000
        data_dir = f"./ckpt/2024-11-07_prediction_error_SD/ltt00_v2_10K_noises/dim{dim:04d}"
        save_dir = f"./ckpt/2024-11-07_prediction_error_SD"
        x_cnt = 0
        for ts_list in ts_list_list:
            fig = plt.figure(figsize=(12, 8))
            # ax = fig.add_subplot(1, 1, 1)
            bins = predefined_bins
            for ts in ts_list:
                data_path = os.path.join(data_dir, f"ts{ts:03d}.txt")
                print(f"Read: {data_path}")
                x = read_floats_from_file(data_path)
                if x_cnt == 0: x_cnt = len(x)
                if isinstance(bins, int):
                    std = np.std(x)
                    bins = np.linspace(-std * 3, std * 3, num=bins + 1, endpoint=True)
                n, bins, patches = plt.hist(x, bins=bins, histtype='step', label=f"t={ts + 1}", linewidth=2)
            # for
            plt.tick_params('both', labelsize=self.tick_size)
            plt.xlabel(r"$\epsilon_{\theta}^{(t)}[d] - \epsilon^{(t)}[d]$", fontsize=xy_label_size)
            plt.ylabel("Frequency", fontsize=xy_label_size)
            plt.legend(fontsize=self.legend_size, loc='upper right')
            ax = plt.gca()
            plt.text(0.03, 0.9, f"Data cnt: {x_cnt//1000}K", transform=ax.transAxes, fontdict=self.font)
            plt.text(0.03, 0.8, f"Bins: {predefined_bins}", transform=ax.transAxes, fontdict=self.font)
            plt.text(0.03, 0.7, f"Model: SD", transform=ax.transAxes, fontdict=self.font)
            f_path = os.path.join(save_dir, f"pred_distribution_dim{dim:4d}_ts{ts_list[0]:03d}-{ts_list[-1]:03d}.png")
            fig.savefig(f_path, bbox_inches='tight')
            print(f"saved file: {f_path}")
            plt.close()
        # for

# class