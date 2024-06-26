import torch


def noise_estimation_loss(model,
                          x0: torch.Tensor,
                          t: torch.LongTensor,
                          e: torch.Tensor,
                          b: torch.Tensor, keepdim=False):
    a = (1-b).cumprod(dim=0).index_select(0, t).view(-1, 1, 1, 1)
    x = x0 * a.sqrt() + e * (1.0 - a).sqrt()
    output = model(x, t.float())
    if keepdim:
        return (e - output).square().sum(dim=(1, 2, 3))
    else:
        return (e - output).square().sum(dim=(1, 2, 3)).mean(dim=0)


loss_registry = {
    'simple': noise_estimation_loss,
}


def noise_estimation_loss2(model,
                           x0: torch.Tensor,
                           t: torch.LongTensor,
                           e: torch.Tensor,
                           a: torch.Tensor,
                           keepdim=False):
    """
    Assume batch size is 250; and image size is 32*32.
    :param model:
    :param x0:  shape [250, 3, 32, 32]  x0
    :param t:   shape [250]             timestep
    :param e:   shape [250, 3, 32, 32]  epsilon
    :param a:   shape [1000].           alpha: (1-b).cumprod(dim=0)
    :param keepdim:
    :return:
    """
    at = a.index_select(0, t).view(-1, 1, 1, 1)  # alpha_t
    x = x0 * at.sqrt() + e * (1.0 - at).sqrt()
    output = model(x, t.float())
    if keepdim:
        return (e - output).square().sum(dim=(1, 2, 3)), x
    else:
        return (e - output).square().sum(dim=(1, 2, 3)).mean(dim=0), x


def x0_estimation_loss2(model_x0,
                        x0: torch.Tensor,
                        t: torch.LongTensor,
                        e: torch.Tensor,
                        a: torch.Tensor,
                        keepdim=False):
    """
    Assume batch size is 250; and image size is 32*32.
    :param model_x0: model to predict x0
    :param x0:  shape [250, 3, 32, 32]  x0
    :param t:   shape [250]             timestep
    :param e:   shape [250, 3, 32, 32]  epsilon
    :param a:   shape [1000].           alpha: (1-b).cumprod(dim=0)
    :param keepdim:
    :return:
    """
    at = a.index_select(0, t).view(-1, 1, 1, 1)  # alpha_t
    xt = x0 * at.sqrt() + e * (1.0 - at).sqrt()
    output = model_x0(xt, t.float())
    if keepdim:
        return (x0 - output).square().sum(dim=(1, 2, 3)), xt
    else:
        return (x0 - output).square().sum(dim=(1, 2, 3)).mean(dim=0), xt


def xp_estimation_loss2(model_xp,
                        x0: torch.Tensor,
                        t: torch.LongTensor,
                        e: torch.Tensor,
                        a: torch.Tensor,
                        p: torch.Tensor,
                        keepdim=False):
    """
    Assume batch size is 250; and image size is 32*32.
    :param model_xp: model to predict x0
    :param x0:  shape [250, 3, 32, 32]  x0
    :param t:   shape [250]             timestep
    :param e:   shape [250, 3, 32, 32]  epsilon
    :param a:   shape [1000].           alpha: (1-b).cumprod(dim=0)
    :param p:   shape [1000].           alpha_t-1
    :param keepdim:
    :return:
    """
    at = a.index_select(0, t).view(-1, 1, 1, 1)  # alpha_t
    xt = x0 * at.sqrt() + e * (1.0 - at).sqrt()
    ap = p.index_select(0, t).view(-1, 1, 1, 1)  # alpha_t-1
    xp = x0 * ap.sqrt() + e * (1.0 - ap).sqrt()  # x_t-1. "p": previous
    output = model_xp(xt, t.float())
    if keepdim:
        return (xp - output).square().sum(dim=(1, 2, 3)), xt
    else:
        return (xp - output).square().sum(dim=(1, 2, 3)).mean(dim=0), xt
