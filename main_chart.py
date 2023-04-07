import matplotlib.pyplot as plt

# order = 1, steps = 10, skip_type = 'time_uniform'
def fig_trajectory_ori_vs_new():
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111)
    plt.xlim((0, 1.02))
    plt.ylim((-0.02, 1))
    plt.ylabel(r"$\bar{\alpha}_{t}$    ", fontsize=35, rotation=0)  # make it horizontal
    plt.xlabel('t', fontsize=30)
    plt.yticks(fontsize=25)
    plt.xticks(fontsize=25)
    # order = 1, steps = 10, skip_type = 'time_uniform'
    # ab: alpha_bar
    ab_ori = [0.895328, 0.656884, 0.394732, 0.194200, 0.078190,
              0.025754, 0.006936, 0.001527, 0.000274, 0.000040, ]
    ab_new = [0.905291, 0.673222, 0.411267, 0.206440, 0.085181,
              0.028907, 0.008075, 0.001858, 0.000353, 0.000055, ]
    ts_ori = [0.099, 0.199, 0.299, 0.399, 0.499, 0.599, 0.699, 0.799, 0.899, 0.998]
    ts_new = [0.094, 0.193, 0.292, 0.391, 0.490, 0.589, 0.688, 0.786, 0.885, 0.983]
    ax.plot(ts_ori, ab_ori, 'o', ms=10, color='blue')
    ax.plot(ts_new, ab_new, '*', ms=16, color='red')
    ax.plot(ts_ori, ab_ori, '--', ms=10, color='green')

    # add text. annotation for dot and star.
    ax.plot([0.48], [0.815], 'o', ms=10, color='blue')
    ax.text(0.5, 0.8, ': original trajectory', size=25)
    ax.plot([0.48], [0.715], '*', ms=16, color='red')
    ax.text(0.5, 0.7, ': improved trajectory', size=25)
    plt.show()
    fig.savefig('./ckpt/chart/abc_o1_s10_tu.png')
    plt.close()

def main():
    fig_trajectory_ori_vs_new()

if __name__ == '__main__':
    main()
