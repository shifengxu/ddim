"""
Testing. Normal sampling has 1000 steps, and we try 2 steps.
"""
import argparse
import sys
import matplotlib.pyplot as plt
from torch.backends import cudnn
from base import *
from var_simulator import VarSimulator
from var_simulator2 import VarSimulator2

log_fn = utils.log_info

torch.set_printoptions(sci_mode=False)
torch.autograd.set_detect_anomaly(True)

def parse_args():
    parser = argparse.ArgumentParser(description=globals()["__doc__"])
    parser.add_argument('--gpu_ids', type=int, nargs='+', default=[7])
    parser.add_argument('--output_dir', type=str, default='./output7_schedule1')
    parser.add_argument("--weight_file", type=str, default='./output7_schedule1/weight_loss_smooth.txt')
    parser.add_argument("--beta_schedule", type=str, default="cosine")

    args = parser.parse_args()

    if not args.weight_file:
        raise Exception(f"Argument weight_file is empty")

    # add device
    gpu_ids = args.gpu_ids
    log_fn(f"gpu_ids : {gpu_ids}")
    args.device = f"cuda:{gpu_ids[0]}" if torch.cuda.is_available() and gpu_ids else "cpu"
    cudnn.benchmark = True

    if not os.path.exists(args.output_dir):
        log_fn(f"os.makedirs({args.output_dir})")
        os.makedirs(args.output_dir)

    return args

def save_vs_plot(args, vs, var_arr):
    ts_cnt = len(var_arr)
    x_arr = ScheduleBase.get_alpha_cumprod(args.beta_schedule, ts_cnt)
    y_arr = vs(x_arr)
    mse = ((var_arr - y_arr) ** 2).mean(axis=0)
    fig, axs = plt.subplots()
    x_axis = np.arange(0, 1000)
    axs.plot(x_axis, var_arr, label="original")
    axs.plot(x_axis, y_arr.numpy(), label=type(vs).__name__)
    axs.set_xlabel(f"timestep. mse={mse:.8f}")
    plt.legend()
    vs_name = type(vs).__name__
    f_path = os.path.join(args.output_dir, f"main_steps2_{vs_name}.png")
    fig.savefig(f_path)
    plt.close()
    log_fn(f"saved: {f_path}")

def main():
    args = parse_args()
    log_fn(f"pid : {os.getpid()}")
    log_fn(f"cwd : {os.getcwd()}")
    log_fn(f"args: {args}")
    var_arr = ScheduleBase.load_floats(args.weight_file)
    # vs = VarSimulator(args.beta_schedule, var_arr, deg=128)
    vs = VarSimulator2(args.beta_schedule, var_arr)
    save_vs_plot(args, vs, var_arr)
    vs.to(args.device)

    k = 0.001  # alpha_1 * alpha_2
    x_arr = np.arange(0.0011, 1.0, 0.0001)
    loss_vivid_a1 = []
    loss_equal_a1 = []
    loss_vivid_a2 = []
    loss_equal_a2 = []
    cnt = len(x_arr)
    for idx, x in enumerate(x_arr):
        a1 = x      # alpha_1
        a2 = k / a1 # alpha_2
        alpha = torch.Tensor([a1, a2]).to(args.device)
        aacum = torch.Tensor([a1, k]).to(args.device)
        new_weight_arr = vs(aacum)
        loss = ScheduleBase.accumulate_variance(alpha, aacum, new_weight_arr)
        loss_vivid_a1.append(loss.item())

        # new_weight_arr = torch.ones_like(aacum, device=args.device)
        # loss = ScheduleBase.accumulate_variance(alpha, aacum, new_weight_arr)
        # loss_equal_a1.append(loss.item())

        a2 = x      # alpha_2
        a1 = k / a2 # alpha_1
        alpha = torch.Tensor([a1, a2]).to(args.device)
        aacum = torch.Tensor([a1, k]).to(args.device)
        new_weight_arr = vs(aacum)
        loss = ScheduleBase.accumulate_variance(alpha, aacum, new_weight_arr)
        loss_vivid_a2.append(loss.item())

        if idx > 0 and idx % 1000 == 0 or idx + 1 == cnt:
            log_fn(f"idx: {idx:04d}/{cnt}. loss:{loss:.6f}, a1:{a1:.6f}, a2:{a2:.6f}")

        # new_weight_arr = torch.ones_like(aacum, device=args.device)
        # loss = ScheduleBase.accumulate_variance(alpha, aacum, new_weight_arr)
        # loss_equal_a2.append(loss.item())

    # for i
    fig, axs = plt.subplots()
    axs.plot(x_arr, loss_vivid_a1, label='final variance depending on a1')
    # axs.plot(x_arr, loss_equal_a1, label='equal_loss_a1')
    axs.plot(x_arr, loss_vivid_a2, label='final variance depending on a2')
    # axs.plot(x_arr, loss_equal_a2, label='equal_loss_a2')
    msg = f"Only 2 steps. k={k:.6f}. x_arr length: {len(x_arr)}; range: {x_arr[0]} ~ {x_arr[-1]}"
    axs.set_xlabel(f"Value of a1 or a2. range: {x_arr[0]:.4f} ~ {x_arr[-1]:.4f}")
    plt.title(f"Sampling by 2 steps: a1 and a2.\nLimitation: a1*a2={k:.4f}")
    fig.legend()
    fig.set_size_inches(18, 8)
    f_path = os.path.join(args.output_dir, f"main_steps2.png")
    fig.savefig(f_path, dpi=100)
    plt.close()
    log_fn(f"Saved: {f_path}")
    f_path = os.path.join(args.output_dir, f"main_steps2_loss_vivid_a1.txt")
    utils.save_list(loss_vivid_a1, 'vivid_a1', f_path, msg=msg, fmt="{:6.1f}")
    log_fn(f"Saved: {f_path}")
    f_path = os.path.join(args.output_dir, f"main_steps2_loss_equal_a1.txt")
    utils.save_list(loss_equal_a1, 'equal_a1', f_path, msg=msg, fmt="{:6.1f}")
    log_fn(f"Saved: {f_path}")
    f_path = os.path.join(args.output_dir, f"main_steps2_loss_vivid_a2.txt")
    utils.save_list(loss_vivid_a2, 'vivid_a2', f_path, msg=msg, fmt="{:6.1f}")
    log_fn(f"Saved: {f_path}")
    f_path = os.path.join(args.output_dir, f"main_steps2_loss_equal_a2.txt")
    utils.save_list(loss_equal_a2, 'equal_a2', f_path, msg=msg, fmt="{:6.1f}")
    log_fn(f"Saved: {f_path}")

    f_path = os.path.join(args.output_dir, f"main_steps2_vs_x.txt")
    utils.save_list(vs.x_arr, 'x', f_path, fmt="{:.6f}")
    log_fn(f"Saved: {f_path}")
    f_path = os.path.join(args.output_dir, f"main_steps2_vs_y.txt")
    utils.save_list(vs.y_arr, 'y', f_path, fmt="{:7.2f}")
    log_fn(f"Saved: {f_path}")

    x_arr = vs.x_arr
    output = vs(x_arr)
    f_path = os.path.join(args.output_dir, f"main_steps2_vs_in.txt")
    utils.save_list(x_arr, 'in', f_path, fmt="{:7.2f}")
    log_fn(f"Saved: {f_path}")
    f_path = os.path.join(args.output_dir, f"main_steps2_vs_out.txt")
    utils.save_list(output, 'out', f_path, fmt="{:7.2f}")
    log_fn(f"Saved: {f_path}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
