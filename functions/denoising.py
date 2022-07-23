import torch


def compute_alpha(beta, t):
    beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
    a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1, 1)
    return a


def generalized_steps(x_T, seq, model, b, **kwargs):
    """
    Original paper: Denoising Diffusion Implicit Models. ICLR. 2021
    :param x_T: x_T in formula; it has initial Gaussian Noise
    :param seq:
    :param model:
    :param b:
    :param kwargs:
    :return:
    """
    eta = kwargs.get("eta", 0)
    with torch.no_grad():
        b_sz = x_T.size(0)
        seq_next = [-1] + list(seq[:-1])
        x0_preds = []
        xs = [x_T]
        for i, j in zip(reversed(seq), reversed(seq_next)):
            t = (torch.ones(b_sz) * i).to(x_T.device)   # [999., 999.]
            q = (torch.ones(b_sz) * j).to(x_T.device)   # [998., 998.]. next t; can assume as t-1
            at = compute_alpha(b, t.long()) # alpha_t
            aq = compute_alpha(b, q.long()) # alpha_{t-1}
            xt = xs[-1].to(x_T.device)      # x_t
            et = model(xt, t)               # epsilon_t
            x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()  # formula (9)
            x0_preds.append(x0_t.to('cpu'))
            sigma_t = eta * ((1 - at / aq) * (1 - aq) / (1 - at)).sqrt()  # formula (16)
            c2 = (1 - aq - sigma_t ** 2).sqrt()
            xt_next = aq.sqrt() * x0_t + c2 * et + sigma_t * torch.randn_like(x_T)  # formula (12)
            xs.append(xt_next.to('cpu'))

    return xs, x0_preds


def ddpm_steps(x, seq, model, b, **kwargs):
    with torch.no_grad():
        n = x.size(0)
        seq_next = [-1] + list(seq[:-1])
        xs = [x]
        x0_preds = []
        betas = b
        for i, j in zip(reversed(seq), reversed(seq_next)):
            t = (torch.ones(n) * i).to(x.device)
            next_t = (torch.ones(n) * j).to(x.device)
            at = compute_alpha(betas, t.long())
            atm1 = compute_alpha(betas, next_t.long())
            beta_t = 1 - at / atm1
            x = xs[-1].to(x.device)

            output = model(x, t.float())
            e = output

            x0_from_e = (1.0 / at).sqrt() * x - (1.0 / at - 1).sqrt() * e
            x0_from_e = torch.clamp(x0_from_e, -1, 1)
            x0_preds.append(x0_from_e.to('cpu'))
            mean_eps = (
                (atm1.sqrt() * beta_t) * x0_from_e + ((1 - beta_t).sqrt() * (1 - atm1)) * x
            ) / (1.0 - at)

            mean = mean_eps
            noise = torch.randn_like(x)
            mask = 1 - (t == 0).float()
            mask = mask.view(-1, 1, 1, 1)
            logvar = beta_t.log()
            sample = mean + mask * torch.exp(0.5 * logvar) * noise
            xs.append(sample.to('cpu'))
    return xs, x0_preds
