import time
import torch
from torch import optim
from torch import nn

import utils
log_fn = utils.log_info

###################################################################################################
class RectifiedFlowSchedulingModel(nn.Module):
    """
    Rectified Flow Scheduling Model
    """
    def __init__(self, out_channels, max_ts=1.0):
        super().__init__()
        self.out_channels = out_channels
        self.max_ts = max_ts
        log_fn(f"RectifiedFlowSchedulingModel()")
        log_fn(f"  out_channels: {self.out_channels}")
        log_fn(f"  max_ts      : {self.max_ts}")
        # the seed. we choose value 0.5. And it is better than value 1.0
        ones_1k = torch.mul(torch.ones((1000,), dtype=torch.float64), 0.5)
        self.seed_1k = torch.nn.Parameter(ones_1k, requires_grad=False)
        self.linear1 = torch.nn.Linear(1000,  2000, dtype=torch.float64)
        self.linear2 = torch.nn.Linear(2000,  2000, dtype=torch.float64)
        self.linear3 = torch.nn.Linear(2000,  self.out_channels, dtype=torch.float64)

    def gradient_clip(self):
        if self.linear1.weight.grad is not None:
            self.linear1.weight.grad = torch.tanh(self.linear1.weight.grad)
            self.linear1.weight.grad = torch.tanh(self.linear1.weight.grad)
        if self.linear2.weight.grad is not None:
            self.linear2.weight.grad = torch.tanh(self.linear2.weight.grad)
            self.linear2.weight.grad = torch.tanh(self.linear2.weight.grad)
        if self.linear3.weight.grad is not None:
            self.linear3.weight.grad = torch.tanh(self.linear3.weight.grad)
            self.linear3.weight.grad = torch.tanh(self.linear3.weight.grad)

    def forward(self):
        output = self.linear1(self.seed_1k)
        output = self.linear2(output)
        output = self.linear3(output)
        # assume delta_arr will be [0.1, 0.2, 0.3, 0.4] with summation 1.0
        # assume self.max_ts is 0.9
        delta_arr = torch.softmax(output, dim=0)        # [0.1,  0.2,  0.3,  0.4 ]
        sum_arr   = torch.cumsum(delta_arr, dim=0)      # [0.1,  0.3,  0.6,  1.0 ]
        delta_arr = torch.flip(delta_arr, dims=[0, ])   # [0.4,  0.3,  0.2,  0.1 ]
        sum_arr   = torch.flip(sum_arr, dims=[0, ])     # [1.0,  0.6,  0.3,  0.1 ]
        delta_arr *= self.max_ts                        # [0.36, 0.27, 0.18, 0.09]
        sum_arr   *= self.max_ts                        # [0.9,  0.54, 0.27, 0.09]
        return sum_arr, delta_arr

###################################################################################################
class ScheduleRectifiedFlow:
    def __init__(self, weight_simulator, steps=10, max_ts=0.955, lr=0.000001, n_epochs=1000,
                 log_interval=50, loss_decay=0.4, delta_expo=0.5, output_file=None, device=None):
        self.steps          = steps
        self.max_ts         = max_ts
        self.loss_decay     = loss_decay
        self.delta_expo     = delta_expo
        self.lr             = lr
        self.n_epochs       = n_epochs
        self.log_interval   = log_interval
        self.output_file    = output_file
        self.device         = device
        log_fn(f"ScheduleRectifiedFlow() =======================================")
        log_fn(f"  self.steps       : {self.steps}")
        log_fn(f"  self.max_ts      : {self.max_ts}")
        log_fn(f"  self.loss_decay  : {self.loss_decay}")
        log_fn(f"  self.delta_expo  : {self.delta_expo}")
        log_fn(f"  self.lr          : {self.lr}")
        log_fn(f"  self.n_epochs    : {self.n_epochs}")
        log_fn(f"  self.log_interval: {self.log_interval}")
        log_fn(f"  self.output_file : {self.output_file}")
        log_fn(f"  self.device      : {self.device}")
        self.weight_simulator = weight_simulator # LinearInterpreter
        self.weight_simulator.to(self.device)

    def train(self):
        model = RectifiedFlowSchedulingModel(out_channels=self.steps, max_ts=self.max_ts)
        model.to(self.device)
        optimizer = optim.SGD(model.parameters(), lr=self.lr, momentum=0.9)
        model.train()
        start_time = time.time()
        loss_ori = None
        loss_min = None
        e_cnt = self.n_epochs
        log_fn(f"ScheduleRectifiedFlow::train()")
        ts_arr = None
        for e_idx in range(0, e_cnt):
            optimizer.zero_grad()
            ts_arr, delta_arr = model()
            weight_arr = self.weight_simulator(ts_arr, include_index=False)
            loss, sub_loss_arr = self.calc_loss(delta_arr, weight_arr)
            if loss_ori is None: loss_ori = loss.item() # track original loss value
            loss.backward()
            model.gradient_clip()
            optimizer.step()
            if e_idx % self.log_interval == 0 or e_idx == e_cnt - 1:
                elp, eta = utils.get_time_ttl_and_eta(start_time, e_idx, e_cnt)
                log_fn(f"E{e_idx:05d}/{e_cnt} loss: {loss:10.5f}, ts[-1]:{ts_arr[-1]:.4f}. elp:{elp}, eta:{eta}")
                if loss_min is None or loss_min > loss.item(): loss_min = loss.item()
            # if
            if e_idx == e_cnt - 1:
                self.detail_save(ts_arr, delta_arr, sub_loss_arr, loss, loss_ori, loss_min)
        # for e_idx
        return ts_arr

    def calc_loss(self, delta_arr, weight_arr):
        sub_loss_arr = delta_arr * weight_arr
        loss = delta_arr[0] * delta_arr[0] * weight_arr[0]
        for i in range(1, len(delta_arr)):
            delta = delta_arr[i]
            to_minus = self.loss_decay * delta.pow(self.delta_expo)
            to_add = delta.pow(0.9)
            loss *= 1 + to_add - to_minus           # original loss adjust
            loss += delta * delta * weight_arr[i]   # new loss introduced
        return loss, sub_loss_arr

    def detail_save(self, ts_arr, delta_arr, sub_loss_arr, loss, loss_ori, loss_min):
        log_fn(f"ScheduleRectifiedFlow::detail_save()")
        comments = [
            f"# steps     : {self.steps}",
            f"# max_ts    : {self.max_ts}",
            f"# loss_decay: {self.loss_decay}",
            f"# delta_expo: {self.delta_expo}",
            f"# lr        : {self.lr}",
            f"# e_porch   : {self.n_epochs}",
            f"# loss_ori  : {loss_ori:8.4f}",
            f"# loss_min  : {loss_min:8.4f}",
            f"# loss      : {loss:8.4f}",
            f"# timestep  : delta    : sub_loss",
        ]
        new_ts = ts_arr * 1000      # change range from [0, 1] to [0, 1000]
        new_dt = delta_arr * 1000   # change range from [0, 1] to [0, 1000]
        contents = []
        for i in range(len(ts_arr)):
            msg = f"{new_ts[i]:8.4f}  : {new_dt[i]:8.4f} : {sub_loss_arr[i]:8.4f}"
            contents.append(msg)
        [log_fn(c) for c in comments]
        [log_fn(c) for c in contents]
        f_path = self.output_file
        if not f_path:
            log_fn(f"self.output_file is invalid: {self.output_file}.")
            return
        with open(f_path, 'w') as fptr:
            [fptr.write(f"{c}\n") for c in comments]
            [fptr.write(f"{c}\n") for c in contents]
        log_fn(f"Saved details to {self.output_file}")
# class
