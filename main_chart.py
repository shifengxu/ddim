import math
import os.path

import numpy as np
import torch
import torchvision as tv
import matplotlib.pyplot as plt

def fig_trajectory_ori_vs_new():
    from unit_test.chart_fig_trajectory_ori_vs_new import ChartFigTrajectoryOriVsNew
    ChartFigTrajectoryOriVsNew().run()

def merge_image():
    dir1 = './ckpt/2023-04-07_image_intermediate/bedroom_reverse_2_10_time_quadratic_new'
    dir2 = './ckpt/2023-04-07_image_intermediate/bedroom_reverse_2_10_time_quadratic_ori'
    dir3 = './ckpt/2023-04-07_image_intermediate/bedroom_reverse_2_10_time_quadratic_all'
    if not os.path.exists(dir3):
        print(f"mkdirs: {dir3}")
        os.makedirs(dir3)
    f_list = os.listdir(dir1)
    f_list.sort()  # file name list
    for fn in f_list:
        f1 = os.path.join(dir1, fn)
        f2 = os.path.join(dir2, fn)
        f3 = os.path.join(dir3, fn)
        i1, i2 = tv.io.read_image(f1), tv.io.read_image(f2)
        i1, i2 = i1.float(), i2.float()
        i1 /= 255.0
        i2 /= 255.0
        print(f"saving image: {f3}")
        tv.utils.save_image([i1, i2], f3, nrow=1)

def fid_compare_ddim_cifar10():
    from unit_test.chart_fid_compare import ChartFidCompare
    ChartFidCompare().run_ddim_cifar10()

def fid_compare_ddim_bedroom():
    from unit_test.chart_fid_compare import ChartFidCompare
    ChartFidCompare().run_ddim_bedroom()

def fid_compare_ddim_celeba():
    from unit_test.chart_fid_compare import ChartFidCompare
    ChartFidCompare().run_ddim_celeba()

def fid_compare_pndm_cifar10():
    from unit_test.chart_fid_compare import ChartFidCompare
    ChartFidCompare().run_pndm_cifar10()

def fid_compare_splitnm_imagenet():
    from unit_test.chart_fid_compare import ChartFidCompare
    ChartFidCompare().run_splitnm_imagenet()

def fig_trajectory_compare_all3():
    from unit_test.chart_fig_trajectory_comapre_all3 import ChartFigTrajectoryCompareAll3
    ChartFigTrajectoryCompareAll3().run()

def discretization_error_vs_prediction_error():
    from unit_test.chart_discretization_error_vs_prediction_error import ChartDiscretizationErrorVsPredictionError
    ChartDiscretizationErrorVsPredictionError().run()

def prediction_error_distribution():
    from unit_test.chart_prediction_error_distribution import ChartPredictionErrorDistribution
    ChartPredictionErrorDistribution().run()

def ablation_ddim():
    lp_arr = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]
    legend_arr = ["logSNR", "quadratic", "uniform"]

    def draw_fig(snr_arr, qua_arr, uni_arr, lambda_s):
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(1, 1, 1)
        ax.tick_params('both', labelsize=25)
        ax.plot(lp_arr, snr_arr, marker='o')
        ax.plot(lp_arr, qua_arr, marker='o')
        ax.plot(lp_arr, uni_arr, marker='o')
        ax.legend(legend_arr, fontsize=25, loc=(0.18, 0.73))
        ax.set_title(f"FID by different $lp$ ($\\lambda={lambda_s}$)", fontsize=25)
        ax.set_ylabel('FID      ', fontsize=28, rotation=0)  # make it horizontal
        ax.set_xlabel('$lp$', fontsize=28)
        ax.set_ylim((10, 22))
        f_path = f"./configs/chart/fig_ablation_ddim_lambda{lambda_s}.png"
        fig.savefig(f_path, bbox_inches='tight')  #
        print(f"file saved: {f_path}")
        plt.close()

    fid_snr_arr = [21.087, 18.822, 16.812, 14.250, 13.359, 13.385, 13.126, 13.509, 13.731]
    fid_qua_arr = [14.008, 13.910, 12.684, 12.496, 13.295, 13.333, 13.641, 14.862, 14.459]
    fid_uni_arr = [16.834, 14.241, 12.637, 11.856, 13.157, 14.539, 15.074, 15.073, 15.205]
    lambda_str = '1E07'
    draw_fig(fid_snr_arr, fid_qua_arr, fid_uni_arr, lambda_str)
    fid_snr_arr = [21.087, 18.517, 15.433, 14.332, 14.606, 14.664, 14.861, 14.449, 14.589]
    fid_qua_arr = [14.008, 13.761, 12.163, 11.922, 11.620, 11.724, 11.841, 11.813, 11.799]
    fid_uni_arr = [16.834, 14.224, 12.705, 16.059, 18.232, 19.020, 20.095, 19.801, 18.949]
    lambda_str = '1E08'
    draw_fig(fid_snr_arr, fid_qua_arr, fid_uni_arr, lambda_str)

def dual_loss_reverse_time_ode_gradient():
    from unit_test.chart_dual_loss_reverse_time_ode_gradient import ChartDualLossReverseTimeODEGradient
    ChartDualLossReverseTimeODEGradient().run()

def dual_loss_negative_effect():
    from unit_test.chart_dual_loss_negative_effect import ChartDualLossNegativeEffect
    ChartDualLossNegativeEffect().run()

def dual_loss_trajectory_diffusion_vs_rectified_flow():
    import unit_test.chart_dual_loss_trajectory_diffusion_vs_rectified_flow as c
    c.ChartDualLossTrajectoryDiffusionVsRectifiedFlow().run()

def dual_loss_merge_image_col_row():
    root_dir = './output3_rf_celeba_sampling'
    lambda_arr = ['0.0', '0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9', '1.0']
    # step_arr = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
    step_arr = ['1', '2', '3']
    save_dir = os.path.join(root_dir, f"merged")
    if not os.path.exists(save_dir):
        print(f"mkdirs: {save_dir}")
        os.makedirs(save_dir)
    # first_dir = os.path.join(root_dir, f"gen_lambda{lambda_arr[0]}_step{step_arr[0]}")
    # fname_list = os.listdir(first_dir)
    # fname_list.sort()
    fname_list = ['00026.png']
    for fname in fname_list:
        img_arr = []
        for step in step_arr:
            for lda in lambda_arr:
                img_path = os.path.join(root_dir, f"gen_lambda{lda}_step{step}", fname)
                # img = tv.io.read_image(img_path) # dimension [3, 64, 64]
                # img = img.float()
                # img /= 255.0
                img = plt.imread(img_path) # dimension: [64, 64, 3]
                img = torch.tensor(img)
                img = img.permute(2, 0, 1)
                img_arr.append(img)
                tv.utils.save_image(img, os.path.join(save_dir, f"lambda{lda}_step{step}.png"))
            # for
        # for
        save_img_path = os.path.join(save_dir, fname)
        tv.utils.save_image(img_arr, save_img_path, nrow=len(lambda_arr))
        print(f"save image: {save_img_path}")
    # for fname

def dual_loss_group_image():
    import shutil
    root_dir = './output3_rf_celeba_sampling'
    lambda_arr = ['0.0', '0.7']
    step_arr = ['3']
    save_dir = os.path.join(root_dir, f"aaa_group")
    if not os.path.exists(save_dir):
        print(f"mkdirs: {save_dir}")
        os.makedirs(save_dir)
    fname_list = [
        '00002', '00011', '00019', '00026', '00039',
        '00041', '00042', '00043', '00051', '00059',
        '00062', '00069', '00081', '00086', '00097',
        '00103', '00107', '00109', '00116', '00117',
    ]
    for fname in fname_list:
        for step in step_arr:
            for lda in lambda_arr:
                img_path = os.path.join(root_dir, f"gen_lambda{lda}_step{step}", f"{fname}.png")
                save_img_path = os.path.join(save_dir, f"{fname}_lambda{lda}_step{step}.png")
                shutil.copyfile(img_path, save_img_path)
                print(f"save image: {save_img_path}")
            # for
        # for
    # for fname

def aaai_fid_compare_deis_pndm_splitnm():
    from unit_test.chart_aaai_fid_compare_deis_pndm_splitnm import ChartAaaiFidCompareDeisPndmSplitnm
    ChartAaaiFidCompareDeisPndmSplitnm().run()

def aaai_mse_error_vs_alpha_bar():
    from unit_test.chart_aaai_mse_error_vs_alpha_bar import ChartAaaiMseErrorVsAlphaBar
    ChartAaaiMseErrorVsAlphaBar().run()

def aaai_prediction_error_distribution():
    from unit_test.chart_aaai_prediction_error_distribution import ChartAaaiPredictionErrorDistribution
    ChartAaaiPredictionErrorDistribution().run()

def aaai_ablation_ddim():
    from unit_test.chart_aaai_ablation_ddim import ChartAaaiAblationDdim
    ChartAaaiAblationDdim().run()

def aaai_alpha_and_learning_portion():
    tick_size = 25
    legend_size = 25
    title_size = 35
    alpha_arr = [0.895329, 0.733680, 0.600916, 0.491980, 0.402631,
                 0.329375, 0.269340, 0.220156, 0.179885, 0.146919]
    alph2_arr = [a - 0.1 for a in alpha_arr]
    cnt = len(alpha_arr)
    a_str_arr = [rf"$\alpha_{{ {i} }}$" for i in range(1, 1+cnt)]
    l_ptn_arr = [0.1 for _ in range(1, 1+cnt)]  # learning portion
    fig = plt.figure(figsize=(16, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.tick_params('both', labelsize=tick_size)
    ax.bar(a_str_arr, alpha_arr, width=0.4)
    ax.bar(a_str_arr, l_ptn_arr, width=0.39, bottom=alph2_arr,
           linewidth=2, linestyle='--', fill=False, edgecolor='red')
    ax.bar(a_str_arr, l_ptn_arr, width=0.39, bottom=alpha_arr,
           linewidth=2, linestyle='--', fill=False, edgecolor='red')
    ax.set_title(f"$\\alpha$ values and learning-portion", fontsize=title_size)
    label_arr = [r"$\alpha$ value", r"learning-portion $\gamma$"]
    ax.legend(label_arr, fontsize=legend_size)

    f_path = f"./configs/chart_aaai2025/alpha_and_lp/fig_alpha_values_time_uniform.png"
    fig.savefig(f_path, bbox_inches='tight')  #
    print(f"file saved: {f_path}")
    plt.close()

def aaai_calc_cpe_vs_fid():
    from unit_test.chart_aaai_calc_cpe_vs_fid import ChartAaaiCalcCepVsFid
    ChartAaaiCalcCepVsFid().run()

def aaai_calc_cpe_original_trend():
    """
    AAAI2025 paper. Calculate cumulative-prediction-error
    """
    fig_size = (16, 8)
    tick_size = 25
    legend_size = 22
    xy_label_size = 30
    title_size = 35

    steps = [4, 6, 8, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]

    cpe_old = [0.307016, 0.336888, 0.340091, 0.327074, 0.283655,
               0.245825, 0.188796, 0.153664, 0.128084, 0.109754,
               0.095985, 0.085507, 0.077053, 0.070360, 0.037039,
               0.025406, 0.019536, 0.015222, 0.013713, 0.012195,
               0.010671, 0.009157, 0.007734]
    fig = plt.figure(figsize=fig_size)
    ax = fig.add_subplot(1, 1, 1)
    ax.tick_params('both', labelsize=tick_size)
    ax.plot(steps, cpe_old, linestyle='--', color='r', marker='o', label="Original trajectory")
    ax.legend(fontsize=legend_size, loc='upper right')
    ax.set_xlabel('step count', fontsize=xy_label_size)
    ax.set_ylabel('CPE      ', fontsize=xy_label_size, rotation=0)
    ax.set_title(r"CIFAR10", fontsize=title_size)
    f_path = './configs/chart_aaai2025/fig_sup_cpe_ori_cifar10.png'
    fig.savefig(f_path, bbox_inches='tight')
    print(f"file saved: {f_path}")
    plt.close()

def aaai_rebuttal_prediction_error_distribution_sd():
    import unit_test.chart_aaai_rebuttal_prediction_error_distribution_sd as a
    a.ChartAaaiRebuttalPredictionErrorDistributionSd().run()

def main():
    """ entry point """
    # fig_trajectory_ori_vs_new()
    # merge_image()
    # fid_compare_ddim_cifar10()
    # fid_compare_ddim_bedroom()
    # fid_compare_ddim_celeba()
    # fid_compare_pndm_cifar10()
    # fid_compare_splitnm_imagenet()
    # fig_trajectory_compare_all3()
    # discretization_error_vs_prediction_error()
    # prediction_error_distribution()
    # ablation_ddim()
    # dual_loss_reverse_time_ode_gradient()
    # dual_loss_negative_effect()
    # dual_loss_trajectory_diffusion_vs_rectified_flow()
    # dual_loss_merge_image_col_row()
    # dual_loss_group_image()
    # aaai_fid_compare_deis_pndm_splitnm()
    # aaai_mse_error_vs_alpha_bar()
    # aaai_prediction_error_distribution()
    # aaai_ablation_ddim()
    # aaai_alpha_and_learning_portion()
    # aaai_calc_cpe_vs_fid()
    # aaai_calc_cpe_original_trend()
    aaai_rebuttal_prediction_error_distribution_sd()

if __name__ == '__main__':
    main()
