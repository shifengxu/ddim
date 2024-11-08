import matplotlib.pyplot as plt
import math

class ChartDualLossReverseTimeODEGradient:
    def __init__(self):
        self.fig_size = (20, 8)

    def run(self):
        def dm_grad_fn(x_0, epsilon, t):
            grad = -(9.95 * t + 0.05) * math.exp(-4.975 * t * t - 0.05 * t) * x_0
            exp2 = math.exp(-9.95 * t * t - 0.1 * t)
            grad += (9.95 * t + 0.05) * exp2 / math.sqrt(1 - exp2) * epsilon
            return grad

        ts_arr = list(range(1, 1000, 1))  # todo: change step back.
        ts_arr = [step / 1000 for step in ts_arr]  # 0.01 ~ 0.999
        ep_arr = [0.6, 0.0, -0.6]  # epsilon array
        x0_arr = [0.7, 0.1, -0.5]  # x_0 array
        fig = plt.figure(figsize=self.fig_size)
        ax = fig.add_subplot(1, 1, 1)
        ax.tick_params('both', labelsize=20)
        line_arr = []
        # (0, (5, 10)): loosely dash
        # (5, (10, 3)): long dash with offset
        for ep, l_stype in zip(ep_arr, ['-', (0, (5, 10)), ':']):
            for x0, l_color in zip(x0_arr, ['r', 'g', 'b']):
                grad_arr = []
                for ts in ts_arr:
                    gr = dm_grad_fn(x0, ep, ts)
                    grad_arr.append(gr)
                # for
                lbl = f"$\\epsilon$:{ep}, $x_0$:{x0}"
                line, = ax.plot(ts_arr, grad_arr, label=lbl, linestyle=l_stype, color=l_color)
                line_arr.append(line)
            # for
        # for
        plt.gca().invert_xaxis()  # reverse time ODE. So reverse X axis.
        legend1 = ax.legend(handles=line_arr[:3], fontsize=24, loc='upper left')  # multiple legend
        plt.gca().add_artist(legend1)
        legend2 = ax.legend(handles=line_arr[3:6], fontsize=24, loc=(0.307, 0.707))
        plt.gca().add_artist(legend2)
        ax.legend(handles=line_arr[6:], fontsize=24, loc='lower left')

        fig.supylabel(r'    $\frac{dx_t}{dt}$', fontsize=40, rotation=0)  # make it horizontal
        fig.supxlabel(r'reverse-time timestep $t$', fontsize=30)
        fig.suptitle("Diffusion model reverse-time ODE on specific points", fontsize=30)
        # plt.show()
        f_path = './configs/chart_dual_loss/fig_reverse_time_ode_grad_dm.png'
        fig.savefig(f_path, bbox_inches='tight')
        print(f"file saved: {f_path}")
        plt.close()

# class
