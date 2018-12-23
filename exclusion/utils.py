import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np


# Computing K-L distribution between q and p
def kl_div_with_logit(q_logit, p_logit):
    q = F.softmax(q_logit, dim=1)
    logq = F.log_softmax(q_logit, dim=1)
    logp = F.log_softmax(p_logit, dim=1)

    qlogq = (q * logq).sum(dim=1).mean(dim=0)
    qlogp = (q * logp).sum(dim=1).mean(dim=0)
    return qlogq - qlogp


# Computing l2_normalization for 3_d
def _l2__normalize_3d(d):
    d = d.numpy()
    d /= (np.sqrt(np.sum(d ** 2, axis=(1, 2, 3, 4))).reshape((-1, 1, 1, 1, 1)) + 1e-16)
    return torch.from_numpy(d)


# Computing vat_loss
def vat_loss(model, ul_x, ul_y, xi=1e-6, eps=2.5, num_iters=1):

    # find r_adv
    d = torch.Tensor(ul_x.size()).normal_()
    for i in range(num_iters):
        d = xi * _l2__normalize_3d(d)
        d = Variable(d.cuda(), requires_grad=True)
        y_hat = model(ul_x + d)
        delta_kl = kl_div_with_logit(ul_y.detach(), y_hat)
        delta_kl.backward()

        d = d.grad.data.clone().cpu()
        model.zero_grad()

    d = _l2__normalize_3d(d)
    d = Variable(d.cuda())
    r_adv = eps * d

    # compute LDS
    y_hat = model(ul_x + r_adv.detach())
    delta_kl = kl_div_with_logit(ul_y.detach(), y_hat)
    return delta_kl


# Computing entropy loss
def entropy_loss(ul_y):
    p = F.softmax(ul_y, dim=1)
    return -(p * F.log_softmax(ul_y, dim=1)).sum(dim=1).mean(dim=0)
