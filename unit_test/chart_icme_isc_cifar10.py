import matplotlib.pyplot as plt


class ChartIcmeIscCifar10:
    def __init__(self):
        self.fig_size = (16, 6)
        self.xy_label_size = 25

    def run(self):
        # data from: D:/Coding2/ddim/ckpt/2024-11-07_Inception_score_on_CIFAR10
        # learning portion: 0.10
        isc_log = [8.1739, 8.7031, 8.9342, 9.0638, 9.3140]  # logSNR
        isc_lo2 = [8.4484, 8.8782, 9.0498, 9.1820, 9.3770]  # logSNR + VRG
        isc_qua = [8.6585, 8.9965, 9.1444, 9.2272, 9.3785]  # quadratic
        isc_qu2 = [8.7678, 9.0571, 9.1528, 9.2269, 9.4279]  # quadratic + VRG
        isc_uni = [8.6956, 9.0422, 9.1700, 9.2207, 9.3546]  # uniform
        isc_un2 = [8.9202, 9.1119, 9.2205, 9.2966, 9.3942]  # uniform + VRG
        steps = ['10', '15', '20', '25', '50']
        fig = plt.figure(figsize=self.fig_size)
        ax1 = fig.add_subplot(1, 3, 1)
        ax2 = fig.add_subplot(1, 3, 2)
        ax3 = fig.add_subplot(1, 3, 3)
        # plt.xlim((0, 1.02))
        # plt.ylim((0, 1800))
        axs = [ax1, ax2, ax3]
        isc_arr = [[isc_log, isc_lo2], [isc_qua, isc_qu2], [isc_uni, isc_un2]]
        lbl_arr = [[s, f"{s}+VRG"] for s in ['logSNR', 'quadratic', 'uniform']]
        ylb_arr = [r"ISC      ", '', '']  # y label on the left
        xlb_arr = ['', r"step count", '']  # x label in the middle
        for ax, fids, lbs, ylb, xlb in zip(axs, isc_arr, lbl_arr, ylb_arr, xlb_arr):
            ax.set_ylabel(ylb, fontsize=self.xy_label_size, rotation=0)  # make it horizontal
            ax.set_xlabel(xlb, fontsize=self.xy_label_size)
            ax.tick_params('both', labelsize=20)
            ax.plot(steps, fids[0], linestyle='-', color='c', marker='o', label=lbs[0])
            ax.plot(steps, fids[1], linestyle='-', color='r', marker='s', label=lbs[1])
            ax.legend(fontsize=18, loc='lower right')
        # for
        fig.suptitle("Comparison with DDIM on CIFAR10 by Inception Score (ISC)", fontsize=30)
        f_path = './configs/chart_icme2025/fig_isc_ddim_cifar10.png'
        fig.savefig(f_path, bbox_inches='tight')
        print(f"file saved: {f_path}")
        plt.close()

# class
