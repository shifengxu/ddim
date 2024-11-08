import matplotlib.pyplot as plt


class ChartDiscretizationErrorVsPredictionError:
    def __init__(self):
        self.fig_size = (16, 8)

    def run(self):
        """
        discretization error, prediction error.
        wde: with discretization error.
        """

        def read_floats_from_file(f):
            x_arr, y_arr = [], []
            with open(f, 'r') as fptr:
                lines = fptr.readlines()
                for line in lines:
                    line = line.strip()
                    if line.startswith('#') or line == '': continue
                    x, y = line.split()
                    x, y = float(x), float(y)
                    x_arr.append(x)
                    y_arr.append(y)
                # for
            # with
            return y_arr, x_arr

        x04, y04 = read_floats_from_file("./configs/2023-06-14_chart_trajectory/fig_trajectory_wde_04.txt")
        x10, y10 = read_floats_from_file("./configs/2023-06-14_chart_trajectory/fig_trajectory_wde_10.txt")
        x1k, y1k = read_floats_from_file("./configs/2023-06-14_chart_trajectory/fig_trajectory_wde_1000.txt")
        f_path = './configs/2023-06-14_chart_trajectory/fig_trajectory_with_diff_errors.png'
        fig = plt.figure(figsize=self.fig_size)
        ax = fig.add_subplot(1, 1, 1)
        ax.tick_params('both', labelsize=25)
        ax.plot(x1k, y1k, linestyle='-', color='r')
        ax.plot(x10, y10, linestyle='-', color='b', marker='o', ms=6)
        ax.plot(x04, y04, linestyle='-', color='c', marker='o', ms=9)

        # arrow
        def add_arrow():
            tmp_fn = lambda xx, yy, idx, r: (xx[idx] * r + xx[idx - 1] * (1 - r), yy[idx] * r + yy[idx - 1] * (1 - r))
            arrow_props = dict(arrowstyle="->", color='c', lw=2.5)
            m, fm = tmp_fn(x04, y04, 2, 0.91)
            n, fn = tmp_fn(x04, y04, 2, 0.99)
            ax.annotate("", xy=(n, fn), xytext=(m, fm), arrowprops=arrow_props, va='center')
            m, fm = tmp_fn(x04, y04, 3, 0.91)
            n, fn = tmp_fn(x04, y04, 3, 0.99)
            ax.annotate("", xy=(n, fn), xytext=(m, fm), arrowprops=arrow_props, va='center')
            m, fm = tmp_fn(x04, y04, 4, 0.91)
            n, fn = tmp_fn(x04, y04, 4, 0.99)
            ax.annotate("", xy=(n, fn), xytext=(m, fm), arrowprops=arrow_props, va='center')
            arrow_props['color'] = 'b'
            m, fm = tmp_fn(x10, y10, 2, 0.91)
            n, fn = tmp_fn(x10, y10, 2, 0.99)
            ax.annotate("", xy=(n, fn), xytext=(m, fm), arrowprops=arrow_props, va='center')
            m, fm = tmp_fn(x10, y10, 3, 0.91)
            n, fn = tmp_fn(x10, y10, 3, 0.99)
            ax.annotate("", xy=(n, fn), xytext=(m, fm), arrowprops=arrow_props, va='center')
            m, fm = tmp_fn(x10, y10, 4, 0.91)
            n, fn = tmp_fn(x10, y10, 4, 0.99)
            ax.annotate("", xy=(n, fn), xytext=(m, fm), arrowprops=arrow_props, va='center')
            tmp_fn = lambda xx, yy, idx, r: (xx[idx] * r + xx[idx - 50] * (1 - r), yy[idx] * r + yy[idx - 50] * (1 - r))
            arrow_props['color'] = 'r'
            m, fm = tmp_fn(x1k, y1k, 400, 0.91)
            n, fn = tmp_fn(x1k, y1k, 400, 0.99)
            ax.annotate("", xy=(n, fn), xytext=(m, fm), arrowprops=arrow_props, va='center')
            m, fm = tmp_fn(x1k, y1k, 666, 0.91)
            n, fn = tmp_fn(x1k, y1k, 666, 0.99)
            ax.annotate("", xy=(n, fn), xytext=(m, fm), arrowprops=arrow_props, va='center')
            m, fm = tmp_fn(x1k, y1k, 977, 0.91)
            n, fn = tmp_fn(x1k, y1k, 977, 0.99)
            ax.annotate("", xy=(n, fn), xytext=(m, fm), arrowprops=arrow_props, va='center')

        # add_arrow()

        ax.plot(x10, y10, linestyle='-', color='b', marker='o', ms=6)  # extra plot. make such lines in front
        ax.plot(x1k, y1k, linestyle='-', color='r')
        legends = ['Ideal continuous trajectory',
                   'Four-step trajectory with discretization error',
                   'Four-step trajectory with discretization and prediction error']
        ax.legend(legends, fontsize=20, loc='upper right')
        ax.set_title(f"Data value Comparison by different trajectories", fontsize=25)
        ax.set_xlabel(r"$x_t[a]$", fontsize=25)
        ax.set_ylabel(r"$x_t[b]$        ", fontsize=25, rotation=0)
        ax.set_ylim((0.1, 1.0))

        # fig.supylabel('  FID', fontsize=25, rotation=0)  # make it horizontal
        # fig.supxlabel('step count', fontsize=25)
        # fig.suptitle("Compare with SplitNM on ImageNet", fontsize=30)
        plt.show()
        fig.savefig(f_path, bbox_inches='tight')
        print(f"file saved: {f_path}")
        plt.close()

# class
