import matplotlib.pyplot as plt


class ChartFidCompare:
    def __init__(self):
        self.xy_label_size = 25

    def run_ddim_cifar10(self):
        fid_log = [21.1, 12.7, 9.6, 8.0, 5.5, 4.7]  # logSNR
        fid_lo2 = [15.2, 9.2, 7.4, 7.2, 4.7, 4.1]  # logSNR + VRG
        fid_qua = [14.0, 8.9, 7.1, 6.1, 4.8, 4.3]  # quadratic
        fid_qu2 = [11.9, 7.9, 6.2, 5.7, 4.3, 3.8]  # quadratic + VRG
        fid_uni = [16.8, 10.7, 8.3, 7.1, 4.8, 4.0]  # uniform
        fid_un2 = [11.1, 7.7, 6.1, 5.8, 4.4, 3.8]  # uniform + VRG
        steps = ['10', '15', '20', '25', '50', '100']
        fig = plt.figure(figsize=(16, 6))
        ax1 = fig.add_subplot(1, 3, 1)
        ax2 = fig.add_subplot(1, 3, 2)
        ax3 = fig.add_subplot(1, 3, 3)
        # plt.xlim((0, 1.02))
        # plt.ylim((0, 1800))
        axs = [ax1, ax2, ax3]
        fid_arr = [[fid_log, fid_lo2], [fid_qua, fid_qu2], [fid_uni, fid_un2]]
        lbl_arr = [[s, f"{s}+VRG"] for s in ['logSNR', 'quadratic', 'uniform']]
        ylb_arr = [r"FID      ", '', '']  # y label on the left
        xlb_arr = ['', r"step count", '']  # x label in the middle
        for ax, fids, lbs, ylb, xlb in zip(axs, fid_arr, lbl_arr, ylb_arr, xlb_arr):
            ax.set_ylabel(ylb, fontsize=self.xy_label_size, rotation=0)  # make it horizontal
            ax.set_xlabel(xlb, fontsize=self.xy_label_size)
            ax.tick_params('both', labelsize=20)
            ax.plot(steps, fids[0], linestyle='-', color='c', marker='o', label=lbs[0])
            ax.plot(steps, fids[1], linestyle='-', color='r', marker='s', label=lbs[1])
            ax.legend(fontsize=18, loc='upper right')
        # for
        fig.suptitle("Compare with DDIM on CIFAR10", fontsize=30)
        plt.show()
        f_path = './configs/chart/fid_ddim_cifar10.png'
        fig.savefig(f_path, bbox_inches='tight')
        print(f"file saved: {f_path}")
        plt.close()

    def run_ddim_bedroom(self):
        fid_log = [46.2, 29.5, 23.1]  # logSNR
        fid_lo2 = [43.1, 27.0, 20.9]  # logSNR + VRG
        fid_qua = [35.1, 23.2, 19.0]  # quadratic
        fid_qu2 = [34.1, 22.7, 18.8]  # quadratic + VRG
        fid_uni = [20.1, 13.4, 11.0]  # uniform
        fid_un2 = [18.6, 12.2, 10.7]  # uniform + VRG
        steps = ['10', '15', '20']
        fig = plt.figure(figsize=(16, 6))
        ax1 = fig.add_subplot(1, 3, 1)
        ax2 = fig.add_subplot(1, 3, 2)
        ax3 = fig.add_subplot(1, 3, 3)
        # plt.xlim((0, 1.02))
        # plt.ylim((0, 1800))
        axs = [ax1, ax2, ax3]
        fid_arr = [[fid_log, fid_lo2], [fid_qua, fid_qu2], [fid_uni, fid_un2]]
        lbl_arr = [[s, f"{s}+VRG"] for s in ['logSNR', 'quadratic', 'uniform']]
        ylb_arr = [r"FID      ", '', '']  # y label on the left
        xlb_arr = ['', r"step count", '']  # x label in the middle
        for ax, fids, lbs, ylb, xlb in zip(axs, fid_arr, lbl_arr, ylb_arr, xlb_arr):
            ax.set_ylabel(ylb, fontsize=self.xy_label_size, rotation=0)  # make it horizontal
            ax.set_xlabel(xlb, fontsize=self.xy_label_size)
            ax.tick_params('both', labelsize=20)
            ax.plot(steps, fids[0], linestyle='-', color='c', marker='o', label=lbs[0])
            ax.plot(steps, fids[1], linestyle='-', color='r', marker='s', label=lbs[1])
            ax.legend(fontsize=18, loc='upper right')
        # for
        fig.suptitle("Compare with DDIM on LSUN-bedroom", fontsize=30)
        plt.show()
        f_path = './configs/chart/fid_ddim_bedroom.png'
        fig.savefig(f_path, bbox_inches='tight')
        print(f"file saved: {f_path}")
        plt.close()

    def run_ddim_celeba(self):
        fid_log = [18.1, 12.1, 9.7]  # logSNR
        fid_lo2 = [15.9, 10.8, 8.7]  # logSNR + VRG
        fid_qua = [28.9, 19.4, 14.5]  # quadratic
        fid_qu2 = [27.0, 16.2, 11.9]  # quadratic + VRG
        fid_uni = [15.4, 9.6, 7.1]  # uniform
        fid_un2 = [14.7, 9.2, 7.0]  # uniform + VRG
        steps = ['10', '15', '20']
        fig = plt.figure(figsize=(16, 6))
        ax1 = fig.add_subplot(1, 3, 1)
        ax2 = fig.add_subplot(1, 3, 2)
        ax3 = fig.add_subplot(1, 3, 3)
        # plt.xlim((0, 1.02))
        # plt.ylim((0, 1800))
        axs = [ax1, ax2, ax3]
        fid_arr = [[fid_log, fid_lo2], [fid_qua, fid_qu2], [fid_uni, fid_un2]]
        lbl_arr = [[s, f"{s}+VRG"] for s in ['logSNR', 'quadratic', 'uniform']]
        ylb_arr = [r"FID      ", '', '']  # y label on the left
        xlb_arr = ['', r"step count", '']  # x label in the middle
        for ax, fids, lbs, ylb, xlb in zip(axs, fid_arr, lbl_arr, ylb_arr, xlb_arr):
            ax.set_ylabel(ylb, fontsize=self.xy_label_size, rotation=0)  # make it horizontal
            ax.set_xlabel(xlb, fontsize=self.xy_label_size)
            ax.tick_params('both', labelsize=20)
            ax.plot(steps, fids[0], linestyle='-', color='c', marker='o', label=lbs[0])
            ax.plot(steps, fids[1], linestyle='-', color='r', marker='s', label=lbs[1])
            ax.legend(fontsize=18, loc='upper right')
        # for
        fig.suptitle("Compare with DDIM on CelebA", fontsize=30)
        plt.show()
        f_path = './configs/chart/fid_ddim_celeba.png'
        fig.savefig(f_path, bbox_inches='tight')
        print(f"file saved: {f_path}")
        plt.close()

    def run_pndm_cifar10(self):
        fid_s = [22.192, 9.7003, 6.8920, 5.6528]  # S-PNDM
        fid_s2 = [17.714, 6.7580, 4.6959, 4.0628]  # S-PNDM + VRG
        fid_f = [18.623, 8.3909, 5.5028, 4.6607]  # F-PNDM
        fid_f2 = [12.465, 4.8597, 3.6508, 3.6243]  # F-PNDM + VRG
        steps = ['5', '10', '15', '20']
        fig = plt.figure(figsize=(16, 8))
        ax1 = fig.add_subplot(1, 2, 1)
        ax2 = fig.add_subplot(1, 2, 2)
        ax1.set_ylim((0, 23))
        ax2.set_ylim((0, 23))
        # plt.xlim((0, 1.02))
        # plt.ylim((0, 1800))
        axs = [ax1, ax2]
        fid_arr = [[fid_s, fid_s2], [fid_f, fid_f2]]
        lbl_arr = [[s, f"{s}+VRG"] for s in ['S-PNDM', 'F-PNDM']]
        for ax, fids, lbs in zip(axs, fid_arr, lbl_arr):
            ax.tick_params('both', labelsize=20)
            ax.plot(steps, fids[0], linestyle='-', color='c', marker='o', label=lbs[0])
            ax.plot(steps, fids[1], linestyle='-', color='r', marker='s', label=lbs[1])
            ax.legend(fontsize=20, loc='upper right')
        # for
        fig.supxlabel('step count', fontsize=self.xy_label_size)
        fig.supylabel('   FID', fontsize=self.xy_label_size, rotation=0)
        fig.suptitle("Compare with PNDM on CIFAR10", fontsize=30)
        plt.show()
        f_path = './configs/chart/fid_pndm_cifar10.png'
        fig.savefig(f_path, bbox_inches='tight')
        print(f"file saved: {f_path}")
        plt.close()

    def run_splitnm_imagenet(self):
        fid_c64_old = [220.4, 151.2, 134.4, 116.7, 46.97]  # conditional 64*64, original trajectory
        fid_c64_new = [28.19, 13.57, 10.64, 9.286, 6.252]  # conditional 64*64, new trajectory
        fid_c128_old = [73.06, 59.44, 51.05, 51.33, 53.00]
        fid_c128_new = [71.79, 57.55, 50.53, 50.76, 52.89]
        steps = ['10', '15', '20', '25', '50']
        fig = plt.figure(figsize=(16, 8))
        ax1 = fig.add_subplot(1, 2, 1)
        ax2 = fig.add_subplot(1, 2, 2)
        axs = [ax1, ax2]
        fid_arr = [[fid_c64_old, fid_c64_new], [fid_c128_old, fid_c128_new]]
        ttl_arr = ['image size: 64*64', 'image size: 128*128']
        lbl_arr = ['STSP', 'STSP+VRG']
        for ax, fids, ttl in zip(axs, fid_arr, ttl_arr):
            ax.tick_params('both', labelsize=20)
            ax.plot(steps, fids[0], linestyle='-', color='c', marker='o')
            ax.plot(steps, fids[1], linestyle='-', color='r', marker='s')
            ax.legend(lbl_arr, fontsize=20, loc='upper right')
            ax.set_title(ttl, fontsize=25)
        # for
        fig.supylabel('  FID', fontsize=self.xy_label_size, rotation=0)  # make it horizontal
        fig.supxlabel('step count', fontsize=self.xy_label_size)
        fig.suptitle("Compare with SplitNM on ImageNet", fontsize=30)
        plt.show()
        f_path = './configs/chart/fid_splitnm_imagenet.png'
        fig.savefig(f_path, bbox_inches='tight')
        print(f"file saved: {f_path}")
        plt.close()

# class
