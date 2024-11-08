import matplotlib.pyplot as plt
import numpy as np

class ChartPredictionErrorDistribution:
    def __init__(self):
        self.predefined_bins = 200

    def run(self):
        """ distribution of delta between predicted noise and ground truth noise """
        predefined_bins = self.predefined_bins

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

        def set_plt_ui(d):
            plt.tick_params('both', labelsize=25)
            # plt.title(r"Distribution of prediction error", fontsize=25)
            if d == 1:
                x_str = f"$1$-$st$ dimension of $\\epsilon_\\theta(x_t) - \\epsilon(x_t)$"
            elif d == 2:
                x_str = f"$2$-$nd$ dimension of $\\epsilon_\\theta(x_t) - \\epsilon(x_t)$"
            elif d == 3:
                x_str = f"$3$-$rd$ dimension of $\\epsilon_\\theta(x_t) - \\epsilon(x_t)$"
            else:
                x_str = f"${d}$-$th$ dimension of $\\epsilon_\\theta(x_t) - \\epsilon(x_t)$"
            plt.xlabel(x_str, fontsize=28)
            plt.ylabel("Frequency", fontsize=28)
            plt.legend(fontsize=25, loc='upper left')

        # ts_all = [0, 99, 199, 299, 399, 499, 599, 699, 799, 899, 999]
        ts_list_list = [
            [0, 99, 199],
            [399, 499, 599],
            [799, 899, 999]
        ]
        dim = 0
        for ts_list in ts_list_list:
            fig = plt.figure(figsize=(12, 6))
            # ax = fig.add_subplot(1, 1, 1)
            bins = predefined_bins
            for ts in ts_list:
                x = read_floats_from_file(f"./output0_lostats/dim{dim:04d}_ts{ts:03d}.txt")
                if isinstance(bins, int):
                    std = np.std(x)
                    bins = np.linspace(-std * 3, std * 3, num=bins + 1, endpoint=True)
                n, bins, patches = plt.hist(x, bins=bins, histtype='step', label=f"t={ts + 1}")
            # for
            set_plt_ui(dim + 1)
            f_path = f"./output0_lostats/fig_delta_distribution_dim{dim:04d}_{ts_list[0]:03d}-{ts_list[-1]:03d}.png"
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
