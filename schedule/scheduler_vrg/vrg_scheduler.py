import os
import torch
from torch import nn, optim, Tensor

from utils import log_info
from .variance_simulator import VarianceSimulator

def load_floats(f_path):
    if not os.path.exists(f_path):
        raise Exception(f"File not found: {f_path}")
    if not os.path.isfile(f_path):
        raise Exception(f"Not file: {f_path}")
    log_info(f"load_floats(): {f_path}...")
    with open(f_path, 'r') as f:
        lines = f.readlines()
    cnt_empty = 0
    cnt_comment = 0
    f_arr = []
    for line in lines:
        line = line.strip()
        if line == '':
            cnt_empty += 1
            continue
        if line.startswith('#'):
            cnt_comment += 1
            continue
        flt = float(line)
        f_arr.append(flt)
    log_info(f"  cnt_empty  : {cnt_empty}")
    log_info(f"  cnt_comment: {cnt_comment}")
    log_info(f"  cnt_valid  : {len(f_arr)}")
    weights = torch.tensor(f_arr, dtype=torch.float64)
    log_info(f"  weights first 5: {weights[:5].numpy()}")
    log_info(f"  weights last 5 : {weights[-5:].numpy()}")
    # because the weights is sum of an CIFAR10 image, which has 3x32x32 dimension
    # and we need the weight of each single element
    log_info(f"  weights = weights / 3072")
    weights = weights / 3072
    log_info(f"  weights first 5: {weights[:5].numpy()}")
    log_info(f"  weights last 5 : {weights[-5:].numpy()}")
    log_info(f"load_floats(): {f_path}...Done")
    return weights

def accumulate_variance(alpha: Tensor, aacum: Tensor, weight_arr: Tensor, details=False):
    """
    accumulate variance from x_1000 to x_1.
    """
    # delta is to avoid torch error:
    #   RuntimeError: Function 'MulBackward0' returned nan values in its 0th output.
    # Or:
    #   the 2nd epoch will have output: tensor([nan, nan, ,,,])
    # Of that error, a possible reason is: torch tensor 0.sqrt()
    # So here, we make sure alpha > aacum.
    delta = torch.zeros_like(aacum)
    delta[0] = 1e-16
    coef = ((1-aacum).sqrt() - (alpha+delta-aacum).sqrt())**2
    numerator = coef * weight_arr
    sub_var = numerator / aacum
    # sub_var *= alpha
    final_var = torch.sum(sub_var)
    if details:
        return final_var, coef, numerator, sub_var
    return final_var

class ScheduleParamAlphaModel(nn.Module):
    """
    Predict alpha, but with predefined alpha base
        Predict alpha, but with predefined alpha base.
    This is for "order-1" DPM solver. The paper detail:
        DPM-Solver: A Fast ODE Solver for Diffusion Probabilistic Model Sampling in Around 10 Steps
        NIPS 2022 Cheng Lu
    """
    def __init__(self, alpha=None, alpha_bar=None, learning_portion=0.01, log_fn=log_info):
        super().__init__()
        if alpha is not None:
            a_base = alpha_bar
            a_base = torch.tensor(a_base)
        elif alpha_bar is not None:
            a_bar = torch.tensor(alpha_bar)
            a_tmp = a_bar[1:] / a_bar[:-1]
            a_base = torch.cat([a_bar[0:1], a_tmp], dim=0)
        else:
            raise ValueError(f"Both alpha and alpha_bar are None")
        a_min = torch.min(a_base)
        a_max = torch.max(a_base)
        assert a_min > 0., f"all alpha must be > 0.: a_min: {a_min}"
        assert a_max < 1., f"all alpha must be < 1.: a_max: {a_max}"
        self.out_channels = len(a_base)
        self.learning_portion = learning_portion
        # make sure learning-portion is small enough. Then new alpha won't exceed range of [0, 1]
        _lp = torch.mul(torch.ones_like(a_base, dtype=torch.float64), learning_portion)
        _lp = torch.minimum(1-a_base, _lp)
        _lp = torch.minimum(a_base, _lp)
        _lp = torch.nn.Parameter(_lp, requires_grad=False)
        self._lp = _lp
        self.log_fn = log_fn
        # hard code the alpha base, which is from DPM-Solver
        # a_base = [0.370370, 0.392727, 0.414157, 0.434840, 0.457460,   # by original TS: 49, 99, 149,,,
        #           0.481188, 0.506092, 0.532228, 0.559663, 0.588520,
        #           0.618815, 0.650649, 0.684075, 0.719189, 0.756066,
        #           0.794792, 0.835464, 0.878171, 0.923015, 0.970102, ]
        # ab.reverse()
        #
        # by geometric with ratio 1.07
        # a_base = [0.991657, 0.978209, 0.961940, 0.942770, 0.920657,
        #           0.895610, 0.867686, 0.828529, 0.797675, 0.750600,
        #           0.704142, 0.654832, 0.597398, 0.537781, 0.477242,
        #           0.417018, 0.353107, 0.292615, 0.236593, 0.177778, ]
        self.alpha_base = torch.nn.Parameter(a_base, requires_grad=False)
        self.linear1 = torch.nn.Linear(1000,  2000, dtype=torch.float64)
        self.linear2 = torch.nn.Linear(2000,  2000, dtype=torch.float64)
        self.linear3 = torch.nn.Linear(2000,  self.out_channels, dtype=torch.float64)

        # the seed. we choose value 0.5. And it is better than value 1.0
        ones_k = torch.mul(torch.ones((1000,), dtype=torch.float64), 0.5)
        self.seed_k = torch.nn.Parameter(ones_k, requires_grad=False)
        f2s = lambda arr: ' '.join([f"{f:.6f}" for f in arr])
        log_fn(f"ScheduleParamAlphaModel::__init__()...")
        log_fn(f"  out_channels     : {self.out_channels}")
        log_fn(f"  learning_portion : {self.learning_portion}")
        log_fn(f"  _lp length       : {len(self._lp)}")
        log_fn(f"  _lp[:5]          : [{f2s(self._lp[:5])}]")
        log_fn(f"  _lp[-5:]         : [{f2s(self._lp[-5:])}]")
        log_fn(f"  alpha_base       : {len(self.alpha_base)}")
        log_fn(f"  alpha_base[:5]   : [{f2s(self.alpha_base[:5])}]")
        log_fn(f"  alpha_base[-5:]  : [{f2s(self.alpha_base[-5:])}]")
        log_fn(f"ScheduleParamAlphaModel::__init__()...Done")

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
        output = self.linear1(self.seed_k)
        output = self.linear2(output)
        output = self.linear3(output)
        output = torch.tanh(output)
        alpha = torch.add(self.alpha_base, output * self._lp)
        aacum = torch.cumprod(alpha, dim=0)

        return alpha, aacum

class VrgScheduler:
    def __init__(self, args, alpha_bar_list):
        log_info(f"VrgScheduler::__init__()...")
        cur_dir = os.path.dirname(os.path.abspath(__file__))
        weight_file = os.path.join(cur_dir, 'mse_weight_file.txt')
        self.wt_arr = load_floats(weight_file)
        self.vs = VarianceSimulator(alpha_bar_list, self.wt_arr)
        self.vs.to(args.device)
        self.lr             = args.lr   # learning-rate
        self.lp             = args.lp   # learning_portion
        self.n_epochs       = args.n_epochs
        self.aa_low         = args.aa_low
        self.aa_low_lambda  = args.aa_low_lambda
        self.device         = args.device
        self.log_interval   = 100
        log_info(f"  device       : {self.device}")
        log_info(f"  lr           : {self.lr}")
        log_info(f"  lp           : {self.lp}")
        log_info(f"  n_epochs     : {self.n_epochs}")
        log_info(f"  aa_low       : {self.aa_low}")
        log_info(f"  aa_low_lambda: {self.aa_low_lambda}")
        log_info(f"  log_interval : {self.log_interval}")
        log_info(f"VrgScheduler::__init__()...Done")

    def schedule(self, f_path, output_file):
        def load_floats_from_file(_f_path):
            log_info(f"  load_floats_from_file()...")
            log_info(f"    {_f_path}")
            with open(_f_path, 'r') as f:
                lines = f.readlines()
            cnt_empty = 0
            float_arr, str_arr, comment_arr = [], [], []
            for line in lines:
                line = line.strip()
                if line == '':
                    cnt_empty += 1
                elif line.startswith('#'):
                    comment_arr.append(line)
                else:
                    # old version: 0.99970543
                    # new version: 0.99970543  : 0000.00561   <<< 2nd column is timestep
                    flt = float(line.split(':')[0].strip()) if ':' in line else float(line)
                    float_arr.append(flt)
                    str_arr.append(line)
            # for
            log_info(f"    cnt_empty  : {cnt_empty}")
            log_info(f"    cnt_comment: {len(comment_arr)}")
            log_info(f"    cnt_valid  : {len(float_arr)}")
            log_info(f"  load_floats_from_file()...Done")
            return float_arr, str_arr, comment_arr

        log_info(f"VrgScheduler::schedule()...")
        log_info(f"  f_path     : {f_path}")
        log_info(f"  output_file: {output_file}")
        alpha_bar, line_arr, c_arr = load_floats_from_file(f_path)
        c_arr = [c[1:] for c in c_arr]  # remove prefix '#'
        c_arr.insert(0, f" Old comments in file {f_path}")
        _, idx_arr = self.vs(torch.tensor(alpha_bar, device=self.device), include_index=True)
        s_arr = [f"{line_arr[i]} : {idx_arr[i]:4d}" for i in range(len(alpha_bar))]
        s_arr.insert(0, "Old alpha_bar and its timestep, and estimated timestep in vs")
        c_arr = c_arr + [''] + s_arr

        new_msg_arr = [f"lr           : {self.lr}",
                       f"lp           : {self.lp}",
                       f"aa_low       : {self.aa_low}",
                       f"aa_low_lambda: {self.aa_low_lambda}",
                       f"n_epochs     : {self.n_epochs}",
                       f"torch.seed() : {torch.seed()}"]  # message array
        c_arr = c_arr + [''] + new_msg_arr

        res = self.train(alpha_bar, c_arr, output_file)
        log_info(f"VrgScheduler::schedule()...Done")
        return res

    def train(self, alpha_bar, msg_arr, output_file):
        log_info(f"VrgScheduler::train()...")

        def model_generate():
            m = ScheduleParamAlphaModel(alpha_bar=alpha_bar, learning_portion=self.lp)
            log_info(f"  model: {type(m).__name__}")
            log_info(f"  out_channels: {m.out_channels}")
            log_info(f"  model.to({self.device})")
            m.to(self.device)
            return m

        # cpe: cumulative-prediction-error
        # reg: regularizer
        # calculate cpe for original trajectory
        aacum = torch.tensor(alpha_bar, device=self.device)
        a_tmp = aacum[1:] / aacum[:-1]
        alpha = torch.cat([aacum[0:1], a_tmp], dim=0)
        weight_arr, idx_arr = self.vs(aacum, include_index=True)
        cpe_ori, _, _, _ = accumulate_variance(alpha, aacum, weight_arr, True)

        e_cnt = self.n_epochs
        model = model_generate()
        optimizer = optim.SGD(model.parameters(), lr=self.lr, momentum=0.9)
        model.train()
        loss_low = None
        for e_idx in range(0, e_cnt):
            optimizer.zero_grad()
            alpha, aacum = model()
            weight_arr, idx_arr = self.vs(aacum, include_index=True)
            cpe, coef, numerator, sub_var = accumulate_variance(alpha, aacum, weight_arr, True)
            aa_min = aacum[-1]
            reg = torch.square(aa_min - self.aa_low) * self.aa_low_lambda
            loss = torch.add(cpe, reg)
            loss.backward()
            model.gradient_clip()
            optimizer.step()
            if e_idx % self.log_interval == 0 or e_idx == e_cnt - 1:
                log_info(f"  E{e_idx:03d}/{e_cnt} loss: {cpe:.5f} {reg:.5f}."
                         f" a:{alpha[0]:.8f}~{alpha[-1]:.8f}; aa:{aacum[0]:.8f}~{aacum[-1]:.5f}")
                if loss_low is None or loss_low > loss.item():
                    loss_low = loss.item()
                    mm = list(msg_arr)
                    mm.append(f"model.lp     : {model.learning_portion}")
                    mm.append(f"model.out_ch : {model.out_channels}")
                    mm.append(f"loss : loss = cumulative_prediction_error + regularizer")
                    mm.append(f"loss : {loss:05.6f} = {cpe:05.6f} + {reg:05.6f}  <<< epoch:{e_idx}")
                    mm.append(f"cpe  : {cpe_ori:10.6f} => {cpe:10.6f}")
                    self.detail_save(output_file, alpha, aacum, idx_arr, weight_arr, coef, numerator, sub_var, mm)
                    log_info(f"  Save file: {output_file}. new loss: {loss.item():.8f}")
                # if
            # if
        # for e_idx
        log_info(f"VrgScheduler::train()...Done")
        return output_file

    @staticmethod
    def detail_save(f_path, alpha, aacum, idx_arr, weight_arr, coef, numerator, sub_var, m_arr):
        combo = []
        for i in range(len(aacum)):
            s = f"{aacum[i]:8.6f}: {idx_arr[i]:3d}: {alpha[i]:8.6f};" \
                f" {coef[i]:8.6f}*{weight_arr[i]:11.6f}={numerator[i]:9.6f};" \
                f" {numerator[i]:9.6f}/{aacum[i]:8.6f}={sub_var[i]:10.6f}"
            s = s.replace('0.000000', '0.0     ')
            combo.append(s)
        m_arr.append('aacum : ts : alpha   ; coef    *weight     =numerator; numerator/aacum   =sub_var')
        with open(f_path, 'w') as f_ptr:
            [f_ptr.write(f"# {m}\n") for m in m_arr]
            [f_ptr.write(f"{s}\n") for s in combo]
        # with

# class
