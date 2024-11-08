import matplotlib.pyplot as plt


class ChartAaaiCalcCepVsFid:
    def __init__(self):
        self.fig_size = (10, 8)

    def run(self):
        """
        AAAI2025 paper. Calculate cumulative-prediction-error
        """
        fig_size = self.fig_size
        tick_size = 25
        legend_size = 25
        xy_label_size = 30
        title_size = 35

        steps = ['4', '6', '8', '10', '12', '15', '20', '30', '40', '50', '60']
        ignore_part = 2  # from 4 to 6 to 8 steps, the CPE increases. So ignore the first two elements.
        steps = steps[ignore_part:]

        cpe_old = [0.307016, 0.336888, 0.340091, 0.327074, 0.309525,
                   0.283655, 0.245825, 0.188796, 0.153664, 0.128084, 0.109754]
        cpe_new = [0.246888, 0.218599, 0.199007, 0.191229, 0.179242,
                   0.161496, 0.137091, 0.101494, 0.120442, 0.102084, 0.087668]
        cpe_old = cpe_old[ignore_part:]
        cpe_new = cpe_new[ignore_part:]
        fig = plt.figure(figsize=fig_size)
        ax = fig.add_subplot(1, 1, 1)
        ax.tick_params('both', labelsize=tick_size)
        ax.plot(steps, cpe_old, linestyle='--', color='c', marker='o', label="Original trajectory")
        ax.plot(steps, cpe_new, linestyle='--', color='r', marker='s', label="Optimized trajectory")
        ax.legend(fontsize=legend_size, loc='upper right')
        ax.set_xlabel('number of sampling steps', fontsize=xy_label_size)
        ax.set_ylabel('CPE      ', fontsize=xy_label_size, rotation=0)
        ax.set_title(r"CIFAR10", fontsize=title_size)
        f_path = './configs/chart_aaai2025/fig_sup_cf_cpe_cifar10.png'
        fig.savefig(f_path, bbox_inches='tight')
        print(f"file saved: {f_path}")
        plt.close()

        fid_old = [48.68, 27.53, 18.71, 14.42, 12.11, 10.02, 8.28, 6.63, 5.86, 5.48, 5.25]
        fid_new = [43.21, 23.81, 16.41, 12.79, 10.79, 8.96, 7.55, 6.33, 5.71, 5.37, 5.19]
        fid_old = fid_old[ignore_part:]
        fid_new = fid_new[ignore_part:]
        fig = plt.figure(figsize=fig_size)
        ax = fig.add_subplot(1, 1, 1)
        ax.tick_params('both', labelsize=tick_size)
        ax.plot(steps, fid_old, linestyle='--', color='c', marker='o', label="Original trajectory")
        ax.plot(steps, fid_new, linestyle='--', color='r', marker='s', label="Optimized trajectory")
        ax.legend(fontsize=legend_size, loc='upper right')
        ax.set_xlabel('number of sampling steps', fontsize=xy_label_size)
        ax.set_ylabel('FID      ', fontsize=xy_label_size, rotation=0)
        ax.set_title(r"CIFAR10", fontsize=title_size)
        f_path = './configs/chart_aaai2025/fig_sup_cf_fid_cifar10.png'
        fig.savefig(f_path, bbox_inches='tight')
        print(f"file saved: {f_path}")
        plt.close()

        mse_old = [0.003597, 0.002233, 0.001566, 0.001152, 0.000874,
                   0.000636, 0.000351, 0.000177, 0.000106, 0.000067, 0.000046]
        mse_new = [0.003186, 0.001864, 0.001328, 0.001021, 0.000803,
                   0.000603, 0.000383, 0.000255, 0.000094, 0.000064, 0.000049]
        mse_old = mse_old[ignore_part:]
        mse_new = mse_new[ignore_part:]
        fig = plt.figure(figsize=fig_size)
        ax = fig.add_subplot(1, 1, 1)
        ax.tick_params('both', labelsize=tick_size)
        ax.plot(steps, mse_old, linestyle='--', color='c', marker='o', label="Original trajectory")
        ax.plot(steps, mse_new, linestyle='--', color='r', marker='s', label="Optimized trajectory")
        ax.legend(fontsize=legend_size, loc='upper right')
        ax.set_xlabel('number of sampling steps', fontsize=xy_label_size)
        ax.set_ylabel('MSE      ', fontsize=xy_label_size, rotation=0)
        ax.set_title(r"CIFAR10", fontsize=title_size)
        f_path = './configs/chart_aaai2025/fig_sup_cf_mse_cifar10.png'
        fig.savefig(f_path, bbox_inches='tight')
        print(f"file saved: {f_path}")
        plt.close()

# class
