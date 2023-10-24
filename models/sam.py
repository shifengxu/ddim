"""
Sharpness-Aware Minimization
https://github.com/davda54/sam
"""
import logging

import torch
from torch.nn.modules.batchnorm import _BatchNorm

class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer_type, rho=0.05, adaptive=False, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(SAM, self).__init__(params, defaults)

        self.rho = rho
        self.adaptive = adaptive
        self.base_optimizer = base_optimizer_type(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)

            for p in group["params"]:
                if p.grad is None: continue
                self.state[p]["old_p"] = p.data.clone()
                tmp = torch.pow(p, 2) if group["adaptive"] else 1.0
                e_w = tmp * p.grad * scale.to(p)
                p.add_(e_w)  # climb to the local maximum "w + e(w)"
                # self.state[p]["e_w"] = e_w   # by yan_zhu

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                p.data = self.state[p]["old_p"]  # get back to "w" from "w + e(w)"
                # p.add_(self.state[p]["e_w"], alpha=-1.0)   # by yan_zhu

        self.base_optimizer.step()  # do the actual "sharpness-aware" update

        if zero_grad: self.zero_grad()

    def _grad_norm(self):
        # put everything on the same device, in case of model parallelism
        shared_device = self.param_groups[0]["params"][0].device
        # -- lyz "adaptive" for ASAM, else for SAM
        tmp_list = []
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    tmp_elem = ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device)
                    tmp_list.append(tmp_elem)
                # if
            # for
        # for
        # tmp_list = [((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device)
        #             for group in self.param_groups for p in group["params"] if p.grad is not None]
        norm = torch.norm(torch.stack(tmp_list), p=2)
        return norm

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups

def disable_running_stats(model):
    def _disable(module):
        if isinstance(module, _BatchNorm):
            module.backup_momentum = module.momentum
            module.momentum = 0

    model.apply(_disable)

def enable_running_stats(model):
    def _enable(module):
        if isinstance(module, _BatchNorm) and hasattr(module, "backup_momentum"):
            module.momentum = module.backup_momentum

    model.apply(_enable)
