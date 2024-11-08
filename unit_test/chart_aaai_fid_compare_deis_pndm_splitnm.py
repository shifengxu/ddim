import matplotlib.pyplot as plt


class ChartAaaiFidCompareDeisPndmSplitnm:
    def __init__(self):
        self.fig_size = (6, 8)

    def run(self):
        """AAAI2025 paper"""
        fig_size = self.fig_size
        tick_size = 25
        legend_size = 22
        xy_label_size = 30
        title_size = 35

        fid_ab1_old = [25.07, 9.80, 6.86, 5.67]  # ab_order=1, original trajectory
        fid_ab1_new = [19.92, 7.47, 5.10, 4.28]  # ab_order=1, new trajectory
        fid_ab2_old = [25.13, 8.07, 5.94, 4.95]
        fid_ab2_new = [16.95, 6.26, 4.42, 3.88]
        steps = ['5', '10', '15', '20']
        fig = plt.figure(figsize=fig_size)
        ax = fig.add_subplot(1, 1, 1)
        ax.tick_params('both', labelsize=tick_size)
        ax.plot(steps, fid_ab1_old, linestyle='-', color='c', marker='o', label="ab1")  # ab_order=1
        ax.plot(steps, fid_ab1_new, linestyle='-', color='r', marker='s', label="ab1 with VRG")
        ax.plot(steps, fid_ab2_old, linestyle='--', color='c', marker='o', label="ab2")
        ax.plot(steps, fid_ab2_new, linestyle='--', color='r', marker='s', label="ab2 with VRG")
        ax.legend(fontsize=legend_size, loc='upper right')
        ax.set_xlabel('step count', fontsize=xy_label_size)
        ax.set_ylabel('FID      ', fontsize=xy_label_size, rotation=0)
        ax.set_title(r"CIFAR10", fontsize=title_size)
        f_path = './configs/chart_aaai2025/aaai_fid_deis_cifar10.png'
        fig.savefig(f_path, bbox_inches='tight')
        print(f"file saved: {f_path}")
        plt.close()

        fid_s1 = [22.192, 9.7003, 6.8920, 5.6528]  # S-PNDM           on CIFAR10
        fid_s2 = [17.714, 6.7580, 4.6959, 4.0628]  # S-PNDM + VRG
        fid_f1 = [18.623, 8.3909, 5.5028, 4.6607]  # F-PNDM
        fid_f2 = [12.465, 4.8597, 3.6508, 3.6243]  # F-PNDM + VRG
        steps = ['5', '10', '15', '20']
        fig = plt.figure(figsize=fig_size)
        ax = fig.add_subplot(1, 1, 1)
        ax.tick_params('both', labelsize=tick_size)
        ax.plot(steps, fid_s1, linestyle='-', color='c', marker='o', label="S-PNDM")
        ax.plot(steps, fid_s2, linestyle='-', color='r', marker='s', label="S-PNDM + VRG")
        ax.plot(steps, fid_f1, linestyle='--', color='c', marker='o', label="F-PNDM")
        ax.plot(steps, fid_f2, linestyle='--', color='r', marker='s', label="F-PNDM + VRG")
        ax.legend(fontsize=legend_size, loc='upper right')
        ax.set_xlabel('step count', fontsize=xy_label_size)
        ax.set_ylabel('FID  ', fontsize=xy_label_size, rotation=0)
        ax.set_title("CIFAR10", fontsize=title_size)
        f_path = './configs/chart_aaai2025/aaai_fid_pndm.png'
        fig.savefig(f_path, bbox_inches='tight')
        print(f"file saved: {f_path}")
        plt.close()

        fid_c64_old = [220.4, 151.2, 134.4, 116.7, 46.97]  # conditional 64*64, original trajectory
        fid_c64_new = [28.19, 13.57, 10.64, 9.286, 6.252]  # conditional 64*64, new trajectory
        steps = ['10', '15', '20', '25', '50']
        fig = plt.figure(figsize=fig_size)
        ax = fig.add_subplot(1, 1, 1)
        ax.tick_params('both', labelsize=tick_size)
        ax.plot(steps, fid_c64_old, linestyle='-', color='c', marker='o', label="STSP")
        ax.plot(steps, fid_c64_new, linestyle='-', color='r', marker='s', label="STSP + VRG")
        ax.legend(fontsize=legend_size, loc='upper right')
        ax.set_xlabel('step count', fontsize=xy_label_size)
        ax.set_ylabel('FID  ', fontsize=xy_label_size, rotation=0)
        ax.set_title(r"ImageNet(64$\times$64)", fontsize=title_size)
        f_path = './configs/chart_aaai2025/aaai_fid_splitnm_imagenet.png'
        fig.savefig(f_path, bbox_inches='tight')
        print(f"file saved: {f_path}")
        plt.close()

# class
