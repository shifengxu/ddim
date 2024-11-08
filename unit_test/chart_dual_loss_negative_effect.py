import matplotlib.pyplot as plt
import numpy as np

class ChartDualLossNegativeEffect:
    def __init__(self):
        self.label_size = 30

    def run(self):
        """there will be two 0.125 areas."""
        x = np.linspace(0.0, 1, 1001)
        x1 = np.linspace(0.0, 0.5, 501)
        y11 = 1.0 * x1
        y12 = [0.0] * len(x1)
        x2 = np.linspace(0.5, 1.0, 501)
        y21 = 1.0 * x2
        y22 = [1.0] * len(x2)

        # remove bounding box
        plt.rc('axes.spines', **{'bottom': True, 'left': True, 'right': False, 'top': False})
        fig, ax = plt.subplots(1, 1, figsize=(12, 12))
        ax.tick_params('both', labelsize=self.label_size)

        for val in [0.0, 0.5, 1.0]:
            lst = [val] * len(x)
            ax.plot(x, lst, linestyle='--', color='0.8')  # y = val
            ax.plot(lst, x, linestyle='--', color='0.8')  # x = val
        ax.fill_between(x1, y11, y12)
        ax.fill_between(x2, y21, y22)
        ax.set_title('Negative Effect of Consistency Term', fontsize=45)
        ax.set_xlabel(r"$a$", fontsize=40)
        ax.set_ylabel(r"$b$    ", fontsize=40, rotation=0)
        text_kwargs = dict(ha='center', va='center', fontsize=40, color='w')
        plt.text(0.3, 0.1, r"$b < a < 0.5$", text_kwargs)
        plt.text(0.7, 0.9, r"$b > a > 0.5$", text_kwargs)
        fig.tight_layout()

        f_path = './configs/chart_dual_loss/fig_negative_effect.png'
        fig.savefig(f_path, bbox_inches='tight')
        print(f"file saved: {f_path}")
        plt.close()

# class
