import torch
from torch.nn import DataParallel
from torch.optim import Adam
from reducer.net import *
from reducer.utils import *
from reducer.data import ExclusionDataset, load_data
import os
import argparse
import numpy as np
from tqdm import tqdm
from pandas import DataFrame

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
luna_dir = '/home/user/wuyunheng/work/DataBowl3/data/luna/preprocessed_luna_data'
data_index_dir = 'reducer/detect_post'
nodule_dir = 'nodule-data'
top_bn = True
args = parser.parse_args()


# Some functions for traning
def to_cuda(x, use_cuda=args.cuda):
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


def evaluate(model, x_s, x_l):

    y_pred = model(x_s, x_l)
    if len(y_pred.size()) == 2:
        score_pos = y_pred.cpu().detach().numpy()[:, 1]
        return score_pos
    elif len(y_pred.size()) == 1:
        score_pos = y_pred.cpu().detach().numpy()[1]
        return score_pos
    else:
        raise Exception('Wrong Shape of score_pos in evaluation')


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

    model = to_cuda(NetBasic(top_bn))
    model = DataParallel(model, device_ids=[0, 1, 2, 3])
    model.apply(weights_init)
    criterion = nn.CrossEntropyLoss()  # ce_loss
    optimizer = Adam(model.parameters(), lr=args.lr)

    # TODO: Verify it for import the entire dataset
    # import dataset
    train_dataset = ExclusionDataset(luna_dir, data_index_dir, fold=args.fold, phase='train')
    X_train, y_train = load_data(train_dataset, nodule_dir)
    unlabeled_dataset = ExclusionDataset(luna_dir, data_index_dir, fold=args.fold, phase='unlabeled')
    X_ul = load_data(unlabeled_dataset, nodule_dir)
    # parameters for training
    batch_size = args.batch_size
    num_iter_per_epoch = (max(len(train_dataset), len(unlabeled_dataset)) / batch_size) + 30

    print "executing fold %d" % args.fold
    print "Labeled training samples: %d" % len(train_dataset)
    if args.semi_spv == 0:
        print "supervised mission"
    else:
        print "semi-supervised mission"
        print "Unlabeled training samples: %d" % len(unlabeled_dataset)
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
            # TODO: Verify it for import the entire dataset
            batch_indices = torch.LongTensor(np.random.choice(len(train_dataset), batch_size, replace=False))
            x_64 = X_train[batch_indices]
            y = y_train[batch_indices]
            x_32 = extract_half(x_64)

            # semi-supervised, we used same batch-size for both labeled and unlabeled
            if args.semi_spv == 1:
                batch_indices_unlabeled = torch.LongTensor(np.random.choice(len(unlabeled_dataset), batch_size, replace=False))
                # TODO: Verify it for import the entire dataset
                ul_x_64 = X_ul[batch_indices_unlabeled]
                ul_x_32 = extract_half(ul_x_64)
                v_loss, ce_loss = train_semi(model.train(),
                                             Variable(to_cuda(x_32)), Variable(to_cuda(x_64)),
                                             Variable(to_cuda(y)),
                                             Variable(to_cuda(ul_x_32)), Variable(to_cuda(ul_x_64)),
                                             optimizer, criterion, epsilon=args.epsilon)

            # supervised with cross-entropy loss
            else:
                sv_loss = train_supervise(model.train(),
                                          Variable(to_cuda(x_32)), Variable(to_cuda(x_64)),
                                          Variable(to_cuda(y)),
                                          optimizer, criterion)

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
               os.path.join(save_dir, 'model.ckpt')
               )

    # Generating test results one by one
    print "Testing step..."
    # TODO: Verify it for import the entire dataset
    test_dataset = ExclusionDataset(luna_dir, data_index_dir, fold=args.fold, phase='test')
    print "Testing samples: %d" % len(test_dataset)
    X_test, y_test, uids, center = load_data(test_dataset, nodule_dir)
    prob_pos = evaluate(model.eval(), X_test, extract_half(X_test))
    series_uid_list = []
    coord_x_list = []
    coord_y_list = []
    coord_z_list = []
    probability_list = []
    for i in tqdm(range(len(test_dataset))):
        series_uid_list.append(uids[i])
        coord_x_list.append(center[i][0])
        coord_y_list.append(center[i][1])
        coord_z_list.append(center[i][2])
        probability_list.append(prob_pos[i])
    print "Finished evaluation step, generating evaluation files.."
    # Saving results
    data_frame = DataFrame({
        'seriesuid': series_uid_list,
        'coordX': coord_x_list,
        'coordY': coord_y_list,
        'coordZ': coord_z_list,
        'probability': probability_list
    })
    data_frame.to_csv(os.path.join(save_dir, 'results.csv'), index=True, sep=',')


if __name__ == '__main__':
    main()
