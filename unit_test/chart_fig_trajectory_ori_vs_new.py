import matplotlib.pyplot as plt


class ChartFigTrajectoryOriVsNew:
    def __init__(self):
        self.fig_size = (12, 8.5)

    def run(self):
        fig = plt.figure(figsize=self.fig_size)
        ax = fig.add_subplot(111)
        plt.xlim((0, 1.02))
        plt.ylim((-0.05, 1.05))
        plt.ylabel(r"$\bar{\alpha}_{t}$    ", fontsize=35, rotation=0)  # make it horizontal
        plt.xlabel(r"timestep $t$", fontsize=30)
        ax.tick_params('y', labelsize=25)
        ax.xaxis.set_ticklabels([])

        # order = 1, steps = 10, skip_type = 'time_uniform'
        # ab: alpha_bar
        # ab_ori = [0.895328, 0.656884, 0.394732, 0.194200, 0.078190,
        #           0.025754, 0.006936, 0.001527, 0.000274, 0.000040, ]
        # ab_new = [0.905291, 0.673222, 0.411267, 0.206440, 0.085181,
        #           0.028907, 0.008075, 0.001858, 0.000353, 0.000055, ]
        # ts_ori = [0.099, 0.199, 0.299, 0.399, 0.499, 0.599, 0.699, 0.799, 0.899, 0.998]
        # ts_new = [0.094, 0.193, 0.292, 0.391, 0.490, 0.589, 0.688, 0.786, 0.885, 0.983]

        # order = 2, steps = 10, skip_type = 'time_quadratic'
        ab_ori = [0.995807, 0.970207, 0.889866, 0.723901, 0.481753,
                  0.236788, 0.075947, 0.013741, 0.001185, 0.000040, ]
        ab_new = [0.991715, 0.956561, 0.868078, 0.697690, 0.457511,
                  0.220373, 0.068510, 0.012959, 0.001219, 0.000054, ]
        ts_ori = [0.015, 0.049, 0.102, 0.174, 0.265, 0.374, 0.502, 0.649, 0.814, 0.998, ]
        ts_new = [0.023, 0.061, 0.113, 0.184, 0.274, 0.383, 0.512, 0.653, 0.813, 0.984, ]

        ax.plot(ts_ori, ab_ori, 'o', ms=10, color='blue')
        ax.plot(ts_new, ab_new, '*', ms=16, color='red')
        ax.plot(ts_ori, ab_ori, '--', ms=10, color='green')

        # add text. annotation for dot and star.
        ax.plot([0.48], [0.815], 'o', ms=10, color='blue')
        ax.text(0.5, 0.8, ': original trajectory', size=25)
        ax.plot([0.48], [0.715], '*', ms=16, color='red')
        ax.text(0.5, 0.7, ': optimized trajectory', size=25)
        ax.text(0.0, -0.06, r"$0$", size=25, transform=ax.transAxes)
        ax.text(0.97, -0.06, r"$T$", size=25, transform=ax.transAxes)
        fig.tight_layout()
        f_path = './configs/chart_aaai2025/fig_abc_o2_s10_tq.png'
        fig.savefig(f_path)
        print(f"Saved: {f_path}")
        plt.close()

# class
