import os

import torch
import math
import utils

log_fn = utils.log_info

class NoiseScheduleVP2:
    def __init__(
            self,
            schedule='discrete',
            betas=None,
            alphas_cumprod=None,
            continuous_beta_0=0.1,
            continuous_beta_1=20.,
            dtype=torch.float32,
            predefined_aap_file=None,
    ):
        """Create a wrapper class for the forward SDE (VP type).

        ***
        Update: We support discrete-time diffusion models by implementing a picewise linear interpolation for log_alpha_t.
                We recommend to use schedule='discrete' for the discrete-time diffusion models, especially for high-resolution images.
        ***

        The forward SDE ensures that the condition distribution q_{t|0}(x_t | x_0) = N ( alpha_t * x_0, sigma_t^2 * I ).
        We further define lambda_t = log(alpha_t) - log(sigma_t), which is the half-logSNR (described in the DPM-Solver paper).
        Therefore, we implement the functions for computing alpha_t, sigma_t and lambda_t. For t in [0, T], we have:

            log_alpha_t = self.marginal_log_mean_coeff(t)
            sigma_t = self.marginal_std(t)
            lambda_t = self.marginal_lambda(t)

        Moreover, as lambda(t) is an invertible function, we also support its inverse function:

            t = self.inverse_lambda(lambda_t)

        ===============================================================

        We support both discrete-time DPMs (trained on n = 0, 1, ..., N-1) and continuous-time DPMs (trained on t in [t_0, T]).

        1. For discrete-time DPMs:

            For discrete-time DPMs trained on n = 0, 1, ..., N-1, we convert the discrete steps to continuous time steps by:
                t_i = (i + 1) / N
            e.g. for N = 1000, we have t_0 = 1e-3 and T = t_{N-1} = 1.
            We solve the corresponding diffusion ODE from time T = 1 to time t_0 = 1e-3.

            Args:
                betas: A `torch.Tensor`. The beta array for the discrete-time DPM. (See the original DDPM paper for details)
                alphas_cumprod: A `torch.Tensor`. The cumprod alphas for the discrete-time DPM. (See the original DDPM paper for details)

            Note that we always have alphas_cumprod = cumprod(1 - betas). Therefore, we only need to set one of `betas` and `alphas_cumprod`.

            **Important**:  Please pay special attention for the args for `alphas_cumprod`:
                The `alphas_cumprod` is the \hat{alpha_n} arrays in the notations of DDPM. Specifically, DDPMs assume that
                    q_{t_n | 0}(x_{t_n} | x_0) = N ( \sqrt{\hat{alpha_n}} * x_0, (1 - \hat{alpha_n}) * I ).
                Therefore, the notation \hat{alpha_n} is different from the notation alpha_t in DPM-Solver. In fact, we have
                    alpha_{t_n} = \sqrt{\hat{alpha_n}},
                and
                    log(alpha_{t_n}) = 0.5 * log(\hat{alpha_n}).


        2. For continuous-time DPMs:

            We support two types of VPSDEs: linear (DDPM) and cosine (improved-DDPM). The hyperparameters for the noise
            schedule are the default settings in DDPM and improved-DDPM:

        ===============================================================

        Args:
            schedule: A `str`. The noise schedule of the forward SDE. 'discrete' for discrete-time DPMs,
                    'linear' or 'cosine' for continuous-time DPMs.
        Returns:
            A wrapper object of the forward SDE (VP type).

        ===============================================================

        Example:

        # For discrete-time DPMs, given betas (the beta array for n = 0, 1, ..., N - 1):
        >>> ns = NoiseScheduleVP2('discrete', betas=betas)

        # For discrete-time DPMs, given alphas_cumprod (the \hat{alpha_n} array for n = 0, 1, ..., N - 1):
        >>> ns = NoiseScheduleVP2('discrete', alphas_cumprod=alphas_cumprod)

        # For continuous-time DPMs (VPSDE), linear schedule:
        >>> ns = NoiseScheduleVP2('linear', continuous_beta_0=0.1, continuous_beta_1=20.)

        """

        if schedule not in ['discrete', 'linear', 'cosine', 'predefined']:
            msg = f"Unsupported noise schedule {schedule}. The schedule" \
                  f" needs to be 'discrete' or 'linear' or 'cosine'"
            raise ValueError(msg)

        self.alpha_bar_map = None  # alpha_bar map: ts_index -> alpha_bar.
        self.schedule = schedule
        log_fn(f"NoiseScheduleVP2()")
        log_fn(f"  schedule: {self.schedule}")
        if schedule == 'discrete':
            if betas is not None:
                log_alphas = 0.5 * torch.log(1 - betas).cumsum(dim=0)
            else:
                assert alphas_cumprod is not None
                log_alphas = 0.5 * torch.log(alphas_cumprod)
            self.total_N = len(log_alphas)
            self.T = 1.
            self.t_array = torch.linspace(0., 1., self.total_N + 1)[1:].reshape((1, -1)).to(dtype=dtype)
            self.log_alpha_array = log_alphas.reshape((1, -1,)).to(dtype=dtype)
        elif schedule == 'predefined':
            self.predefined_aap_file = predefined_aap_file
            self.alphas_cumprod = alphas_cumprod
            log_fn(f"  load_predefined_aap() from: {self.predefined_aap_file}")
            aap, ts = self.load_predefined_aap(self.predefined_aap_file)
            self.predefined_aap_cnt = len(aap)
            aap.insert(0, 0.9999)
            ts.insert(0, 0)
            self.predefined_aap = torch.tensor(aap)  # predefined alpha accumulated product
            self.predefined_ts = torch.tensor(ts)    # predefined timesteps
            self.T = 1.
            f2s = lambda arr: ' '.join([f"{f:8.6f}" for f in arr])
            i2s = lambda arr: ' '.join([f"{i: 8d}" for i in arr])
            log_fn(f"  predefined_aap     : {len(self.predefined_aap)}")
            log_fn(f"  predefined_aap[:5] : [{f2s(self.predefined_aap[:5])}]")
            log_fn(f"  predefined_aap[-5:]: [{f2s(self.predefined_aap[-5:])}]")
            log_fn(f"  predefined_ts      : {len(self.predefined_ts)}")
            log_fn(f"  predefined_ts[:5]  : [{i2s(self.predefined_ts[:5])}]")
            log_fn(f"  predefined_ts[-5:] : [{i2s(self.predefined_ts[-5:])}]")
            log_fn(f"  T                  : {self.T}")
        else:
            self.total_N = 1000
            self.beta_0 = continuous_beta_0
            self.beta_1 = continuous_beta_1
            self.cosine_s = 0.008
            self.cosine_beta_max = 999.
            self.cosine_t_max = math.atan(self.cosine_beta_max * (1. + self.cosine_s) / math.pi) * 2. * (
                        1. + self.cosine_s) / math.pi - self.cosine_s
            self.cosine_log_alpha_0 = math.log(math.cos(self.cosine_s / (1. + self.cosine_s) * math.pi / 2.))
            if schedule == 'cosine':
                # For the cosine schedule, T = 1 will have numerical issues. So we manually set the ending time T.
                # Note that T = 0.9946 may be not the optimal setting. However, we find it works well.
                self.T = 0.9946
            else:
                self.T = 1.

    def load_predefined_aap(self, f_path: str):
        if f_path.startswith('geometric_ratio:'):
            ratio = f_path.split(':')[1]
            ratio = float(ratio)
            series = utils.create_geometric_series(0., 999., ratio, 21)
            series = series[1:]  # ignore the first element
            i_arr = [int(f) for f in series]
            i_arr_tensor = torch.tensor(i_arr, device=self.alphas_cumprod.device)
            f_arr_tensor = self.alphas_cumprod.index_select(0, i_arr_tensor)
            f_arr = f_arr_tensor.tolist()
            return f_arr, i_arr

        if not os.path.exists(f_path):
            raise Exception(f"File not found: {f_path}")
        if not os.path.isfile(f_path):
            raise Exception(f"Not file: {f_path}")
        with open(f_path, 'r') as f:
            lines = f.readlines()
        cnt_empty = 0
        cnt_comment = 0
        f_arr = []  # float array
        i_arr = []  # int array
        for line in lines:
            line = line.strip()
            if line == '':
                cnt_empty += 1
                continue
            if line.startswith('#'):
                cnt_comment += 1
                continue
            arr = line.split(':')
            flt, itg = float(arr[0]), int(arr[1])
            f_arr.append(flt)
            i_arr.append(itg)
        f2s = lambda ff: ' '.join([f"{f:8.6f}" for f in ff])
        i2s = lambda ii: ' '.join([f"{i: 8d}" for i in ii])
        log_fn(f"    cnt_empty  : {cnt_empty}")
        log_fn(f"    cnt_comment: {cnt_comment}")
        log_fn(f"    cnt_valid  : {len(f_arr)}")
        log_fn(f"    float[:5]  : [{f2s(f_arr[:5])}]")
        log_fn(f"    float[-5:] : [{f2s(f_arr[-5:])}]")
        log_fn(f"    int[:5]    : [{i2s(i_arr[:5])}]")
        log_fn(f"    int[-5:]   : [{i2s(i_arr[-5:])}]")
        return f_arr, i_arr

    def to(self, device):
        self.predefined_aap = self.predefined_aap.to(device)
        self.predefined_ts = self.predefined_ts.to(device)

    def _marginal_log_mean_coeff(self, t):
        """
        Compute log(alpha_t) of a given continuous-time label t in [0, T].
        """
        if self.schedule == 'discrete':
            tmp = interpolate_fn(t.reshape((-1, 1)), self.t_array.to(t.device), self.log_alpha_array.to(t.device))
            return tmp.reshape((-1))
        elif self.schedule == 'linear':
            return -0.25 * t ** 2 * (self.beta_1 - self.beta_0) - 0.5 * t * self.beta_0
        elif self.schedule == 'cosine':
            log_alpha_fn = lambda s: torch.log(torch.cos((s + self.cosine_s) / (1. + self.cosine_s) * math.pi / 2.))
            log_alpha_t = log_alpha_fn(t) - self.cosine_log_alpha_0
            return log_alpha_t
        elif self.schedule == 'predefined':
            aap = self.predefined_aap.index_select(0, t)
            log_alpha_t = torch.log(aap) * 0.5
            return log_alpha_t

    def marginal_log_mean_coeff(self, t, t_idx=None):
        """"""
        log_alpha_t = self._marginal_log_mean_coeff(t)
        if self.alpha_bar_map is not None:
            if t_idx is None: raise ValueError(f"t_idx is None")
            t, t_idx = t.reshape((-1)), t_idx.reshape((-1))
            if len(t) != len(t_idx): raise ValueError(f"t.len != t_idx.len. {len(t)} != {len(t_idx)}")
            alpha_bar_arr = torch.exp(log_alpha_t*2)
            for i in range(len(alpha_bar_arr)):
                ti_str = f"{t_idx[i]:03d}"
                ab_str = f"{alpha_bar_arr[i]:.8f}"
                if ti_str not in self.alpha_bar_map:
                    self.alpha_bar_map[ti_str] = ab_str
                    log_fn(f"{type(self).__name__}::marginal_log_mean_coeff() {ti_str}: {ab_str}")
            # for
        # if
        return log_alpha_t

    def marginal_alpha(self, t):
        """
        Compute alpha_t of a given continuous-time label t in [0, T].
        """
        return torch.exp(self.marginal_log_mean_coeff(t))

    def marginal_std(self, t):
        """
        Compute sigma_t of a given continuous-time label t in [0, T].
        """
        return torch.sqrt(1. - torch.exp(2. * self.marginal_log_mean_coeff(t)))

    def marginal_lambda(self, t, t_idx=None):
        """
        Compute lambda_t = log(alpha_t) - log(sigma_t) of a given continuous-time label t in [0, T].
        """
        log_mean_coeff = self.marginal_log_mean_coeff(t, t_idx)
        log_std = 0.5 * torch.log(1. - torch.exp(2. * log_mean_coeff))
        return log_mean_coeff - log_std

    def inverse_lambda(self, lamb):
        """
        Compute the continuous-time label t in [0, T] of a given half-logSNR lambda_t.
        """
        if self.schedule == 'linear':
            tmp = 2. * (self.beta_1 - self.beta_0) * torch.logaddexp(-2. * lamb, torch.zeros((1,)).to(lamb))
            Delta = self.beta_0 ** 2 + tmp
            return tmp / (torch.sqrt(Delta) + self.beta_0) / (self.beta_1 - self.beta_0)
        elif self.schedule == 'discrete':
            log_alpha = -0.5 * torch.logaddexp(torch.zeros((1,)).to(lamb.device), -2. * lamb)
            tmp1 = torch.flip(self.log_alpha_array.to(lamb.device), [1])
            tmp2 = torch.flip(self.t_array.to(lamb.device), [1])
            t = interpolate_fn(log_alpha.reshape((-1, 1)), tmp1, tmp2)
            return t.reshape((-1,))
        else:
            log_alpha = -0.5 * torch.logaddexp(-2. * lamb, torch.zeros((1,)).to(lamb))
            t_fn = lambda log_alpha_t: torch.arccos(torch.exp(log_alpha_t + self.cosine_log_alpha_0)) * 2. * (
                        1. + self.cosine_s) / math.pi - self.cosine_s
            t = t_fn(log_alpha)
            return t

def interpolate_fn(x, xp, yp):
    """
    A piecewise linear function y = f(x), using xp and yp as key points.
    We implement f(x) in a differentiable way (i.e. applicable for autograd).
    The function f(x) is well-defined for all x-axis. (For x beyond the bounds of xp,
     we use the outmost points of xp to define the linear function.)

    Args:
        x: PyTorch tensor with shape [N, C], where N is the batch size,
            C is the number of channels (we use C = 1 for DPM-Solver).
        xp: PyTorch tensor with shape [C, K], where K is the number of keypoints.
        yp: PyTorch tensor with shape [C, K].
    Returns:
        The function values f(x), with shape [N, C].
    """
    N, K = x.shape[0], xp.shape[1]
    all_x = torch.cat([x.unsqueeze(2), xp.unsqueeze(0).repeat((N, 1, 1))], dim=2)
    sorted_all_x, x_indices = torch.sort(all_x, dim=2)
    x_idx = torch.argmin(x_indices, dim=2)
    cand_start_idx = x_idx - 1
    start_idx = torch.where(
        torch.eq(x_idx, 0),
        torch.tensor(1, device=x.device),
        torch.where(torch.eq(x_idx, K), torch.tensor(K - 2, device=x.device), cand_start_idx),
    )
    end_idx = torch.where(torch.eq(start_idx, cand_start_idx), start_idx + 2, start_idx + 1)
    start_x = torch.gather(sorted_all_x, dim=2, index=start_idx.unsqueeze(2)).squeeze(2)
    end_x = torch.gather(sorted_all_x, dim=2, index=end_idx.unsqueeze(2)).squeeze(2)
    start_idx2 = torch.where(
        torch.eq(x_idx, 0),
        torch.tensor(0, device=x.device),
        torch.where(torch.eq(x_idx, K), torch.tensor(K - 2, device=x.device), cand_start_idx),
    )
    y_positions_expanded = yp.unsqueeze(0).expand(N, -1, -1)
    start_y = torch.gather(y_positions_expanded, dim=2, index=start_idx2.unsqueeze(2)).squeeze(2)
    end_y = torch.gather(y_positions_expanded, dim=2, index=(start_idx2 + 1).unsqueeze(2)).squeeze(2)
    cand = start_y + (x - start_x) * (end_y - start_y) / (end_x - start_x)
    return cand
