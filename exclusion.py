import torch
from torch.optim import Adam
from torch.nn import DataParallel
from exclusion.net import *
from exclusion.utils import *
import os
import argparse
import matplotlib.pyplot as plt
import numpy as np
import sklearn.metrics as metrics
from sklearn.model_selection import StratifiedKFold
from scipy import interp
from tqdm import tqdm

parser = argparse.ArgumentParser(description='False Positive Reduction in semi method')
parser.add_argument('-b', '--batch-size', default=16, type=int, metavar='N',
                    help='mini-batch size (default: 16)')
parser.add_argument('--epochs', default=120, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--epsilon', type=float, default=2.5, required=True,
                    help='epsilon for VAT')
parser.add_argument('--test', default=1, type=int, metavar='SPLIT', choices=[0, 1],
                    help='1 do test evaluation, 0 not')
parser.add_argument('--cuda', default=True, type=bool,
                    help='Whether use cuda, defualt: True')
parser.add_argument('--fold', required=True, type=int, choices=[0, 1, 2], metavar='N',
                    help='Training data in fold...')
parser.add_argument('--semi-spv', type=int, default=0, choices=[0, 1],
                    help='1 do semi supervised classification, 0 not')
parser.add_argument('--save-dir', default='', type=str, metavar='SAVE',
                    help='directory to save checkpoint and results (default: none)')

cuda_device = "0, 1, 2, 3"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = cuda_device


# Some parameters for training
top_bn = True
args = parser.parse_args()


# Some functions for traning
def tocuda(x, use_cuda=args.cuda):
    if use_cuda:
        return x.cuda()
    return x


def extract_half(x_large):
    assert len(x_large.shape) == 5
    assert x_large.shape[1] == 1
    centre = np.asarray((x_large.shape[2], x_large.shape[3], x_large.shape[4])) / 2
    x_small = x_large[:, :,
                      (centre[0] - centre[0] / 2): (centre[0] + centre[0] / 2),
                      (centre[1] - centre[1] / 2): (centre[1] + centre[1] / 2),
                      (centre[2] - centre[2] / 2): (centre[2] + centre[2] / 2)]
    return x_small


def train_semi(model, x_s, x_l, y, ul_x_s, ul_x_l, optimizer, criterion, epsilon):

    y_pred = model(x_s, x_l)
    ce_loss = criterion(y_pred, y)

    ul_y = model(ul_x_s, ul_x_l)
    v_loss = vat_loss(model, ul_x_s, ul_x_l, ul_y, eps=epsilon)
    loss = v_loss + ce_loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return v_loss, ce_loss


def train_supervise(model, x_s, x_l, y, optimizer, criterion):

    y_pred = model(x_s, x_l)
    loss = criterion(y_pred, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss


def eval(model, x_s, x_l, y):

    y_pred = model(x_s, x_l)
    score_pos = y_pred.cpu().detach().numpy()[:, 1]
    return score_pos


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        m.bias.data.fill_(0)


def main():

    torch.manual_seed(114514)
    torch.cuda.manual_seed_all(114514)

    model = tocuda(NetBasic(top_bn))
    model = DataParallel(model, device_ids=[0, 1, 2, 3])
    model.apply(weights_init)
    criterion = nn.CrossEntropyLoss()  # ce_loss
    optimizer = Adam(model.parameters(), lr=args.lr)

    # TODO: Finish loading data
    X_train = torch.randn(400, 1, 64, 64, 64)
    _, y_train = torch.max(torch.randn(400, 2), dim=1)
    X_ul = torch.randn(600, 1, 32, 32, 32)

    # parameters for training
    batch_size = args.batch_size
    num_iter_per_epoch = (max(X_train.size()[0], X_ul.size()[0]) / batch_size) + 20

    print "executing fold %d" % args.fold
    if args.semi_spv == 0:
        print "supervised mission"
    else:
        print "semi-supervised mission"
    print
    print

    for epoch in range(args.epochs):
        print "epoch: %d" % (epoch + 1)

        # epoch decay settings
        if epoch <= args.epochs * 0.5:
            decayed_lr = args.lr
        elif epoch <= args.epochs * 0.8:
            decayed_lr = 0.1 * args.lr
        else:
            decayed_lr = ((args.epochs - epoch) * (0.1 * args.lr)) / (args.epochs - (0.8 * args.epochs))
        optimizer.lr = decayed_lr
        optimizer.betas = (0.5, 0.999)

        print "contains %d iterations." % num_iter_per_epoch
        for i in tqdm(range(num_iter_per_epoch)):
            # training in batches
            batch_indices = torch.LongTensor(np.random.choice(X_train.size()[0], batch_size, replace=False))
            x_64 = X_train[batch_indices]
            x_32 = extract_half(x_64)
            y = y_train[batch_indices]

            # semi-supervised, we used same batch-size for both labeled and unlabeled
            if args.semi_spv == 1:
                batch_indices_unlabeled = torch.LongTensor(np.random.choice(X_ul.size()[0],
                                                                            batch_size, replace=False))
                ul_x_64 = X_ul[batch_indices_unlabeled]
                ul_x_32 = extract_half(ul_x_64)
                v_loss, ce_loss = train_semi(model.train(), x_32, x_64, y, ul_x_32, ul_x_64,
                                             optimizer, criterion, epsilon=args.epsilon)

            # supervised with cross-entropy loss
            else:
                sv_loss = train_supervise(model.train(), x_32, x_64, y, optimizer, criterion)

    # saving model
    print "saving model..."
    state_dict = model.module.state_dict()
    for key in state_dict.keys():
        state_dict[key] = state_dict[key].cpu()

    save_dir = os.path.join(args.save_dir, 'fold%d' % args.fold)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    torch.save({'save_dir': args.save_dir,
                'state_dict': state_dict,
                'args': args},
               os.path.join(save_dir, 'fold%d' % args.fold, 'model.ckpt')
               )

    # TODO: Finish the test set loading
    x_test_64 = torch.randn(350, 1, 64, 64, 64)
    _, y_test = torch.max(torch.randn(350, 2), dim=1)

    x_test_32 = extract_half(x_test_64)
    # evaluation: output the positive probability
    print "generating evaluation results..."
    pos_prob = eval(model.eval(), Variable(x_test_32), Variable(x_test_64), y_test)
    np.save(os.path.join(save_dir, 'pos_prob.npy'), pos_prob)

    print "Complete!"
