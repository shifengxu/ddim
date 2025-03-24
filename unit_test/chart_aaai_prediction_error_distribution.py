import matplotlib.pyplot as plt
import numpy as np


class ChartAaaiPredictionErrorDistribution:
    def __init__(self):
        self.tick_size = 30
        self.legend_size = 30

    def run(self):
        """ distribution of delta between predicted noise and ground truth noise """
        tick_size = self.tick_size
        legend_size = self.legend_size
        xy_label_size = 40
        predefined_bins = 200

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
            [0, 99, 199],
            [399, 499, 599],
        ]
        for ts_list in ts_list_list:
            fig = plt.figure(figsize=(12, 5))
            # ax = fig.add_subplot(1, 1, 1)
            bins = predefined_bins
            for ts in ts_list:
                x = read_floats_from_file(f"./configs/2023-06-17_chart_prediction_error_distribution/ts{ts:03d}.txt")
                if isinstance(bins, int):
                    std = np.std(x)
                    bins = np.linspace(-std * 3, std * 3, num=bins + 1, endpoint=True)
                n, bins, patches = plt.hist(x, bins=bins, histtype='step', label=f"t={ts + 1}", linewidth=3)
            # for
            plt.tick_params('both', labelsize=tick_size)
            plt.xlabel(r"$\epsilon_{\theta}^{(t)}[d] - \epsilon^{(t)}[d]$", fontsize=xy_label_size)
            plt.ylabel("Frequency", fontsize=xy_label_size)
            plt.legend(fontsize=legend_size, loc='upper right')
            f_path = f"./configs/chart_icme2025/fig_delta_distribution_{ts_list[0]:03d}-{ts_list[-1]:03d}.png"
            fig.savefig(f_path, bbox_inches='tight')
            print(f"saved file: {f_path}")
            plt.close()
        # for

        # for ts in ts_all:
        #     fig = plt.figure(figsize=(12, 8))
        #     ax = fig.add_subplot(1, 1, 1)
        #     x = read_floats_from_file(f"./output0_lostats/ts{ts:03d}.txt")
        #     std = np.std(x)
        #     bins = np.linspace(-std * 3, std * 3, num=predefined_bins+1, endpoint=True)
        #     plt.hist(x, bins=bins, histtype='step', label=f"timestep t={ts + 1}")
        #     set_plt_ui()
        #     f_path = f"./output0_lostats/fig_delta_distribution_ts{ts:03d}.png"
        #     # fig.savefig(f_path, bbox_inches='tight')
        #     fig.savefig(f_path)
        #     print(f"saved file: {f_path}")
        #     plt.close()
        # # for

# class