import matplotlib.pyplot as plt


class ChartAaaiAblationDdim:
    def __init__(self):
        self.tick_size = 25

    def run(self):
        tick_size = self.tick_size
        legend_size = 22
        xy_label_size = 30
        title_size = 35

        lp_arr = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]
        legend_arr = ["logSNR", "quadratic", "uniform"]

        def draw_fig(snr_arr, qua_arr, uni_arr, lambda_s):
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(1, 1, 1)
            ax.tick_params('both', labelsize=tick_size)
            ax.plot(lp_arr, snr_arr, marker='o')
            ax.plot(lp_arr, qua_arr, marker='o')
            ax.plot(lp_arr, uni_arr, marker='o')
            ax.legend(legend_arr, fontsize=legend_size, loc=(0.18, 0.73))
            ax.set_title(f"$\\lambda={lambda_s}$", fontsize=title_size)
            ax.set_ylabel('FID      ', fontsize=xy_label_size, rotation=0)  # make it horizontal
            ax.set_xlabel(r"$\gamma$", fontsize=xy_label_size)
            ax.set_ylim((10, 22))
            f_path = f"./configs/chart_aaai2025/fig_ablation_ddim_lambda{lambda_s}.png"
            fig.savefig(f_path, bbox_inches='tight')  #
            print(f"file saved: {f_path}")
            plt.close()

        fid_snr_arr = [21.087, 18.822, 16.812, 14.250, 13.359, 13.385, 13.126, 13.509, 13.731]
        fid_qua_arr = [14.008, 13.910, 12.684, 12.496, 13.295, 13.333, 13.641, 14.862, 14.459]
        fid_uni_arr = [16.834, 14.241, 12.637, 11.856, 13.157, 14.539, 15.074, 15.073, 15.205]
        lambda_str = '1E04'
        draw_fig(fid_snr_arr, fid_qua_arr, fid_uni_arr, lambda_str)
        fid_snr_arr = [21.087, 18.517, 15.433, 14.332, 14.606, 14.664, 14.861, 14.449, 14.589]
        fid_qua_arr = [14.008, 13.761, 12.163, 11.922, 11.620, 11.724, 11.841, 11.813, 11.799]
        fid_uni_arr = [16.834, 14.224, 12.705, 16.059, 18.232, 19.020, 20.095, 19.801, 18.949]
        lambda_str = '1E05'
        draw_fig(fid_snr_arr, fid_qua_arr, fid_uni_arr, lambda_str)

# class