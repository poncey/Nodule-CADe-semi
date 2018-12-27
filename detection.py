# coding=utf-8
import argparse
import os
import time
import numpy as np
import detector.data as data
from importlib import import_module
import shutil
from detector.utils import *
from detector.split_combine import SplitComb
import torch
from torch.nn import DataParallel
from torch.backends import cudnn
from torch.utils.data import DataLoader
from torch.autograd import Variable
from detector.config_detector import config as config_detector
from detector.layers import acc
import sys
sys.path.append('../')

parser = argparse.ArgumentParser(description='PyTorch DataBowl3 Detector')
parser.add_argument('--model', '-m', metavar='MODEL', default='base',
                    help='model')
parser.add_argument('-j', '--workers', default=2, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=16, type=int,
                    metavar='N', help='mini-batch size (default: 16)')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--save-freq', default='10', type=int, metavar='S',
                    help='save frequency')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--save-dir', default='', type=str, metavar='SAVE',
                    help='directory to save checkpoint (default: none)')
parser.add_argument('--test', default=0, type=int, metavar='TEST',
                    help='1 do test evaluation, 0 not')
parser.add_argument('--split', default=8, type=int, metavar='SPLIT',
                    help='In the test phase, split the image to 8 parts')
parser.add_argument('--gpu', default='all', type=str, metavar='N',
                    help='use gpu')
parser.add_argument('--n_test', default=1, type=int, metavar='N',
                    help='number of gpu for test')

cuda_device = "0, 1, 2, 3"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = cuda_device

# eps=100
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8 python main.py --model res18 -b 32 --epochs $eps --save-dir res18
# 训练detector时，使用的模型是res18，batch size是32，epoch=100，训练完毕后的模型参数保存在res18文件夹内。


def main():
    global args
    args = parser.parse_args()

    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    model = import_module(args.model)
    config, net, loss, get_pbb = model.get_model()
    start_epoch = args.start_epoch
    save_dir = args.save_dir
    
    if args.resume:
        checkpoint = torch.load(args.resume)
        if start_epoch == 0:
            start_epoch = checkpoint['epoch'] + 1
        if not save_dir:
            save_dir = checkpoint['save_dir']
        else:
            save_dir = os.path.join('results', save_dir)
        net.load_state_dict(checkpoint['state_dict'])
    else:
        if start_epoch == 0:
            start_epoch = 1
        if not save_dir:
            exp_id = time.strftime('%Y%m%d-%H%M%S', time.localtime())
            save_dir = os.path.join('results', args.model + '-' + exp_id)
        else:
            save_dir = os.path.join('results', save_dir)
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    logfile = os.path.join(save_dir, 'log')
    if args.test != 1:
        sys.stdout = Logger(logfile)
        pyfiles = [f for f in os.listdir('./') if f.endswith('.py')]
        for f in pyfiles:
            shutil.copy(f, os.path.join(save_dir,f))
    n_gpu = setgpu(args.gpu)
    args.n_gpu = n_gpu
    net = net.cuda()
    loss = loss.cuda()
    cudnn.benchmark = True
    net = DataParallel(net)
    datadir = config_detector['preprocess_result_path']
    print 'datadir = ',datadir

    net = DataParallel(net, device_ids=[0])
    
    # dataset = data.DataBowl3Detector(
    #     datadir,
    #     'detector/luna_file_id/file_id_train.npy',
    #     config,
    #     phase='train')
    # train_loader = DataLoader(
    #     dataset,
    #     batch_size=args.batch_size,
    #     shuffle=True,
    #     num_workers=args.workers,
    #     pin_memory=True)
    #
    # dataset = data.DataBowl3Detector(
    #     datadir,
    #     'detector/luna_file_id/file_id_val.npy',
    #     config,
    #     phase='val')
    # val_loader = DataLoader(
    #     dataset,
    #     batch_size=args.batch_size,
    #     shuffle=False,
    #     num_workers=args.workers,
    #     pin_memory=True)

    # optimizer = torch.optim.SGD(
    #     net.parameters(),
    #     args.lr,
    #     momentum=0.9,
    #     weight_decay=args.weight_decay)
    
    def get_lr(epoch):
        if epoch <= args.epochs * 0.5:
            lr = args.lr
        elif epoch <= args.epochs * 0.8:
            lr = 0.1 * args.lr
        else:
            lr = 0.01 * args.lr
        return lr

    # # Training and Validation approach
    # train_loss_l, validate_loss_l = [], []
    # train_tpr_l, validate_tpr_l = [], []
    # for epoch in range(start_epoch, args.epochs + 1):
    #     train_loss, train_tpr = train(train_loader, net, loss, epoch, optimizer, get_lr, args.save_freq, save_dir)
    #     validate_loss, validate_tpr = validate(val_loader, net, loss)
    #     train_loss_l.append(train_loss)
    #     validate_loss_l.append(validate_loss)
    #     train_tpr_l.append(train_tpr)
    #     validate_tpr_l.append(validate_tpr)
    # # Save Train-loss and Validate-Loss
    # if not os.path.exists('./train-vali-results'):
    #     os.mkdir('./train-vali-results')
    # np.save('./train-vali-results/train-loss.npy', np.asarray(train_loss_l).astype(np.float64))
    # np.save('./train-vali-results/validation-loss.npy', np.asarray(validate_loss_l).astype(np.float64))
    # np.save('./train-vali-results/train-tpr.npy', np.asarray(train_tpr_l).astype(np.float64))
    # np.save('./train-vali-results/validation-tpr.npy', np.asarray(validate_tpr_l).astype(np.float64))
    # print "Finished saving training results"
    #
    # # test process
    # if args.test == 1:
    #     margin = 32
    #     sidelen = 144
    #
    #     split_comber = SplitComb(sidelen, config['max_stride'], config['stride'], margin, config['pad_value'])
    #     dataset = data.DataBowl3Detector(
    #         datadir,
    #         'detector/luna_file_id/file_id_test.npy',
    #         config,
    #         phase='test',
    #         split_comber=split_comber)
    #     test_loader = DataLoader(
    #         dataset,
    #         batch_size=1,  # 在测试阶段，batch size 固定为1
    #         shuffle=False,
    #         num_workers=args.workers,
    #         collate_fn=data.collate,
    #         pin_memory=False)
    #
    #     test(test_loader, net, get_pbb, save_dir, config)
    #     return
    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            m.weight.data.normal_(0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)
        elif classname.find('Linear') != -1:
            m.bias.data.fill_(0)

    # TODO: Cross-Validation with file_ids
    for k_fold in range(10):
        print "Authorizing fold: {:d}".format(k_fold)

        # Loading training set
        dataset = data.DataBowl3Detector(
            datadir,
            'detector/luna_file_id/subset_{:d}'.format(k_fold) + '/file_id_train.npy',
            config,
            phase='train'
        )
        train_loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.workers,
            pin_memory=True)
        # Loading validation set
        dataset = data.DataBowl3Detector(
            datadir,
            'detector/luna_file_id/subset_{:d}'.format(k_fold) + '/file_id_val.npy',
            config,
            phase='val')
        val_loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=True)

        optimizer = torch.optim.SGD(
            net.parameters(),
            args.lr,
            momentum=0.9,
            weight_decay=args.weight_decay)

        # Training process
        train_loss_l, validate_loss_l = [], []
        train_tpr_l, validate_tpr_l = [], []

        # weights initialize
        net.apply(weights_init)

        for epoch in range(start_epoch, args.epochs + 1):
            if not os.path.exists(os.path.join(save_dir, 'fold{:d}'.format(k_fold))):
                os.makedirs(os.path.join(save_dir, 'fold{:d}'.format(k_fold)))
            train_loss, train_tpr = train(train_loader, net, loss, epoch, optimizer, get_lr, args.save_freq,
                                          os.path.join(save_dir, 'fold{:d}'.format(k_fold)))
            validate_loss, validate_tpr = validate(val_loader, net, loss)

            # Append loss results
            train_loss_l.append(train_loss)
            validate_loss_l.append(validate_loss)
            train_tpr_l.append(train_tpr)
            validate_tpr_l.append(validate_tpr)

        # Save Train-Validation results
        if not os.path.exists('./train-vali-results/fold{:d}'.format(k_fold)):
            os.makedirs('./train-vali-results/fold{:d}'.format(k_fold))
        np.save('./train-vali-results/fold{:d}'.format(k_fold) + '/train-loss.npy',
                np.asarray(train_loss_l).astype(np.float64))
        np.save('./train-vali-results/fold{:d}'.format(k_fold) + '/validation-loss.npy',
                np.asarray(validate_loss_l).astype(np.float64))
        np.save('./train-vali-results/fold{:d}'.format(k_fold) + '/train-tpr.npy',
                np.asarray(train_tpr_l).astype(np.float64))
        np.save('./train-vali-results/fold{:d}'.format(k_fold) + '/validation-tpr.npy',
                np.asarray(validate_tpr_l).astype(np.float64))

        # Testing process
        if args.test == 1:
            margin = 32
            sidelen = 144

            split_comber = SplitComb(sidelen, config['max_stride'], config['stride'], margin, config['pad_value'])
            dataset = data.DataBowl3Detector(
                datadir,
                'detector/luna_file_id/subset_{:d}'.format(k_fold) + '/file_id_test.npy',
                config,
                phase='test',
                split_comber=split_comber)
            test_loader = DataLoader(
                dataset,
                batch_size=1,  # 在测试阶段，batch size 固定为1
                shuffle=False,
                num_workers=args.workers,
                collate_fn=data.collate,
                pin_memory=False)

            if not os.path.exists(os.path.join(save_dir, 'fold{:d}'.format(k_fold))):
                os.makedirs(os.path.join(save_dir, 'fold{:d}'.format(k_fold)))
            test(test_loader, net, get_pbb, os.path.join(save_dir, 'fold{:d}'.format(k_fold)), config)
            return


def train(data_loader, net, loss, epoch, optimizer, get_lr, save_freq, save_dir):
    start_time = time.time()
    
    net.train()  # Setting to
    lr = get_lr(epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    metrics = []
    for i, (data, target, coord) in enumerate(data_loader):
        data = Variable(data.cuda(async=True))
        target = Variable(target.cuda(async=True))
        coord = Variable(coord.cuda(async=True))

        output = net(data, coord)
        loss_output = loss(output, target)  # 损失函数返回的loss_output格式是[总的损失，分类损失，回归损失1,回归损失2,回归损失3,回归损失4，tp,tp+fn,tn,tn+fp]
        optimizer.zero_grad()  # 此处每个图块为RPN的一个batch，所以计算针对每个图块的loss时，都要先清空梯度，然后再反向传播、更新参数
        loss_output[0].backward()
        optimizer.step()

        loss_output[0] = loss_output[0].data[0]
        metrics.append(loss_output)  # metrics, 一个list，每个元素是一个图块的loss。总共图块数=所有结节数/0.7
        # print('epoch = ',epoch,'    i = ',i,'    loss_output = ',loss_output)

    if epoch % args.save_freq == 0:            
        state_dict = net.module.state_dict()
        for key in state_dict.keys():
            state_dict[key] = state_dict[key].cpu()
            
        torch.save({
            'epoch': epoch,
            'save_dir': save_dir,
            'state_dict': state_dict,
            'args': args},
            os.path.join(save_dir, '%03d.ckpt' % epoch))
        print('save path = ', os.path.join(save_dir, '%03d.ckpt' % epoch))

    end_time = time.time()

    metrics = np.asarray(metrics, np.float32)
    tpr = np.sum(metrics[:, 6]) / np.sum(metrics[:, 7])

    print 'metrics[0] = ',metrics[0]
    print('Epoch %03d (lr %.5f)' % (epoch, lr))
    print('Train:      tpr %3.2f, tnr %3.2f, total pos %d, total neg %d, time %3.2f' % (
        100.0 * tpr,
        100.0 * np.sum(metrics[:, 8]) / np.sum(metrics[:, 9]),
        np.sum(metrics[:, 7]),
        np.sum(metrics[:, 9]),
        end_time - start_time))
    print('loss %2.4f, classify loss %2.4f, regress loss %2.4f, %2.4f, %2.4f, %2.4f' % (
        np.mean(metrics[:, 0]),
        np.mean(metrics[:, 1]),
        np.mean(metrics[:, 2]),
        np.mean(metrics[:, 3]),
        np.mean(metrics[:, 4]),
        np.mean(metrics[:, 5])))
    print

    return np.mean(metrics[:, 0]), tpr  # Return total loss of single epoch


def validate(data_loader, net, loss):
    start_time = time.time()
    
    net.eval()

    metrics = []
    for i, (data, target, coord) in enumerate(data_loader):
        with torch.no_grad():
            data = Variable(data.cuda(async=True))  # data = Variable(data.cuda(async = True), volatile = True)
            target = Variable(target.cuda(async=True))  # target = Variable(target.cuda(async = True), volatile = True)
            coord = Variable(coord.cuda(async=True))  # coord = Variable(coord.cuda(async = True), volatile = True)

            output = net(data, coord)
            loss_output = loss(output, target, train=False)

            loss_output[0] = loss_output[0].data[0]
            metrics.append(loss_output)    
    end_time = time.time()

    metrics = np.asarray(metrics, np.float32)
    tpr = np.sum(metrics[:, 6]) / np.sum(metrics[:, 7])
    print('Validation: tpr %3.2f, tnr %3.8f, total pos %d, total neg %d, time %3.2f' % (
        100.0 * tpr,
        100.0 * np.sum(metrics[:, 8]) / np.sum(metrics[:, 9]),
        np.sum(metrics[:, 7]),
        np.sum(metrics[:, 9]),
        end_time - start_time))
    print('loss %2.4f, classify loss %2.4f, regress loss %2.4f, %2.4f, %2.4f, %2.4f' % (
        np.mean(metrics[:, 0]),
        np.mean(metrics[:, 1]),
        np.mean(metrics[:, 2]),
        np.mean(metrics[:, 3]),
        np.mean(metrics[:, 4]),
        np.mean(metrics[:, 5])))
    print
    print

    return np.mean(metrics[:, 0]), tpr  # Return total loss of single epoch


def test(data_loader, net, get_pbb, save_dir, config):
    start_time = time.time()
    save_dir = os.path.join(save_dir, 'bbox')  # detector/results/res18/bbox
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    print('saving path:', save_dir)
    net.eval()  # 把模型设置为评估模式，只对dropout、batch norm有影响
    namelist = []
    split_comber = data_loader.dataset.split_comber
    for i_name, (data, target, coord, nzhw) in enumerate(data_loader):
        # 从data_loader获取出来一个batch的数据。每个torch.tensor被封装为长度为batch size的List，其它类型的数据（如numpy.array）
        # 被封装成长度为batch size的tuple。本模型在test阶段的batch size 为 抠出的肺部块分割成了208*208*208之后总共分割的块数。
        s = time.time()
        target = [np.asarray(t, np.float32) for t in target]
        lbb = target[0]
        nzhw = nzhw[0]
        
        name = data_loader.dataset.filenames[i_name].split('/')[-1].split('_clean')[0]
        data = data[0][0]
        coord = coord[0][0]
        isfeat = False
        if 'output_feature' in config:
            if config['output_feature']:
                isfeat = True
        # print 'isfeat = ',isfeat
        n_per_run = args.n_test  # 测试时的gpu数量
        print('data.size = ', data.size())
        splitlist = range(0, len(data)+1,n_per_run)
        if splitlist[-1] != len(data):
            splitlist.append(len(data))
        outputlist = []
        featurelist = []
        print('splitlist = ',splitlist)

        for i in range(len(splitlist)-1):
            with torch.no_grad():  # 注意，只要是tensor参与计算的，如果不需要反向传播，就一定要将tensor放在torch.no_grad():区域内，以节省显存
                input = Variable(data[splitlist[i]:splitlist[i+1]]).cuda()  # input = Variable(data[splitlist[i]:splitlist[i+1]], volatile = True).cuda()
                inputcoord = Variable(coord[splitlist[i]:splitlist[i+1]]).cuda()  # inputcoord = Variable(coord[splitlist[i]:splitlist[i+1]], volatile = True).cuda()
                print('input.shape = ', input.shape, ' inputcoord.shape = ', inputcoord.shape)
                if isfeat:
                    output, feature = net(input,inputcoord)
                    featurelist.append(feature.data.cpu().numpy())
                else:
                    output = net(input,inputcoord)
                outputlist.append(output.data.cpu().numpy())
        output = np.concatenate(outputlist,0)
        print('output.shape = ', output.shape)
        # print('output = ',output)
        output = split_comber.combine(output,nzhw=nzhw) # 这里的返回值就是整个肺部扫描（抠出了肺部）所对应的完整bbox预测
        # if isfeat:
            # 下面的语句都执行不到，应该是作者遗留下来的无用的语句。所以直接注释掉
            # feature = np.concatenate(featurelist,0).transpose([0,2,3,4,1])[:,:,:,:,:,np.newaxis]
            # feature = split_comber.combine(feature,sidelen)[...,0]
            # print('sidelen = ',sidelen)

        thresh = -3
        pbb, mask = get_pbb(output,thresh, ismask=True)  # pbb的形状为[-1,5]，因为使用了sigmoid(thresh)作为阈值进行了过滤，因此这里可以直接使用作者写好的nms函数进行去重，这里没有调用nms，应该是在分类的时候才进行了调用
        print('pbb.shape = ', pbb.shape)
        if isfeat:
            feature_selected = feature[mask[0], mask[1],mask[2]]
            np.save(os.path.join(save_dir, name+'_feature.npy'), feature_selected)
        # tp,fp,fn,_ = acc(pbb,lbb,0,0.1,0.1)
        # print([len(tp),len(fp),len(fn)])
        print('iname,name = ', [i_name,name])
        e = time.time()
        np.save(os.path.join(save_dir, name+'_pbb.npy'), pbb.astype(np.float64))  # pbb的形状为[-1,5]，已经使用sigmoid(thresh)过滤了
        np.save(os.path.join(save_dir, name+'_lbb.npy'), lbb.astype(np.float64))  # lbb只是把人工标记给保存了起来
    np.save(os.path.join(save_dir, 'namelist.npy'), namelist)
    end_time = time.time()

    print('elapsed time is %3.2f seconds' % (end_time - start_time))
    print
    print


def singletest(data, net, config, splitfun, combinefun, n_per_run, margin=64, isfeat=False):
    z, h, w = data.size(2), data.size(3), data.size(4)
    print(data.size())
    data = splitfun(data,config['max_stride'],margin)
    data = Variable(data.cuda(async=True), volatile=True, requires_grad=False)
    splitlist = range(0,args.split+1, n_per_run)
    outputlist = []
    featurelist = []
    for i in range(len(splitlist)-1):
        if isfeat:
            output, feature = net(data[splitlist[i]: splitlist[i+1]])
            featurelist.append(feature)
        else:
            output = net(data[splitlist[i]: splitlist[i+1]])
        output = output.data.cpu().numpy()
        outputlist.append(output)
        
    output = np.concatenate(outputlist, 0)
    output = combinefun(output, z / config['stride'], h / config['stride'], w / config['stride'])
    if isfeat:
        feature = np.concatenate(featurelist, 0).transpose([0, 2, 3, 4, 1])
        feature = combinefun(feature, z / config['stride'], h / config['stride'], w / config['stride'])
        return output, feature
    else:
        return output


if __name__ == '__main__':
    main()
