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


# Computing l2_normalization , modified for 3d networks
def _l2__normalize_3d(d):
    d = d.numpy()
    d /= (np.sqrt(np.sum(d ** 2, axis=(1, 2, 3, 4))).reshape((-1, 1, 1, 1, 1)) + 1e-16)
    return torch.from_numpy(d)


# Computing vat_loss, modified for dual inputs
def vat_loss(model, ul_x_s, ul_x_l, ul_y, xi=1e-6, eps=2.5, num_iters=1):

    # find r_adv for two inputs
    d_s = torch.Tensor(ul_x_s.size()).normal_()
    d_l = torch.Tensor(ul_x_l.size()).normal_()
    for i in range(num_iters):
        d_s = xi * _l2__normalize_3d(d_s)
        d_l = xi * _l2__normalize_3d(d_l)

        d_s = Variable(d_s.cuda(), requires_grad=True)
        d_l = Variable(d_l.cuda(), requires_grad=True)

        y_hat = model(ul_x_s + d_s, ul_x_l + d_l)
        delta_kl = kl_div_with_logit(ul_y.detach(), y_hat)
        delta_kl.backward()

        d_s = d_s.grad.data.clone().cpu()
        d_l = d_l.grad.data.clone().cpu()
        model.zero_grad()

    d_s = _l2__normalize_3d(d_s)
    d_l = _l2__normalize_3d(d_l)

    d_s = Variable(d_s.cuda())
    d_l = Variable(d_l.cuda())

    r_adv_s = eps * d_s
    r_adv_l = eps * d_l

    # compute LDS
    y_hat = model(ul_x_s + r_adv_s.detach(), ul_x_l + r_adv_l.detach())
    delta_kl = kl_div_with_logit(ul_y.detach(), y_hat)
    return delta_kl


# Computing entropy loss
def entropy_loss(ul_y):
    p = F.softmax(ul_y, dim=1)
    return -(p * F.log_softmax(ul_y, dim=1)).sum(dim=1).mean(dim=0)
