# encoding=utf-8

import numpy as np
import torch
from torch.utils.data import Dataset
import os
import time
import collections
import random
from layers import iou
from scipy.ndimage import zoom
import warnings
from scipy.ndimage.interpolation import rotate


class DataBowl3Detector(Dataset):
    def __init__(self, data_dir, split_path, config, phase='train', split_comber=None):
        assert(phase == 'train' or phase == 'val' or phase == 'test')
        self.phase = phase
        self.max_stride = config['max_stride']       
        self.stride = config['stride']  # =4
        sizelim = config['sizelim']/config['reso']
        sizelim2 = config['sizelim2']/config['reso']
        sizelim3 = config['sizelim3']/config['reso']
        self.blacklist = config['blacklist']
        self.isScale = config['aug_scale']
        self.r_rand = config['r_rand_crop']  # =0.3
        self.augtype = config['augtype']
        self.pad_value = config['pad_value']
        self.split_comber = split_comber
        idcs = np.load(split_path)  # 获取当前phase对应的所有数据集的index
        if phase != 'test':
            idcs = [f for f in idcs if (f not in self.blacklist)]

        self.filenames = [os.path.join(data_dir, '%s_clean.npy' % idx) for idx in idcs]
        #self.kagglenames = [f for f in self.filenames if len(f.split('/')[-1].split('_')[0])>20]
        self.lunanames = [f for f in self.filenames if len(f.split('/')[-1].split('_')[0])<20]
        
        #print('self.lunanames[0] = ',self.lunanames[0])
        #print('len(self.lunanames) = ',len(self.lunanames))
        labels = []
        
        for idx in idcs:
            l = np.load(os.path.join(data_dir, '%s_label.npy' % idx))
            if np.all(l == 0):
                l = np.array([])
            labels.append(l)

        self.sample_bboxes = labels  # labels是长度为文件数的list，每个list元素为一个numpy数组，存储了对应文件的所有结节信息
        if self.phase != 'test':
            self.bboxes = []
            for i, l in enumerate(labels):
                if len(l) > 0:  # 有的文件没有结节，所以去掉
                    # print '第{%d}个文件有{%d}个结节' % (i,len(l))
                    for t in l:
                        if t[3] > sizelim:  # 也就是小于6mm的结节直接丢弃，不使用
                            self.bboxes.append([np.concatenate([[i],t])])
                        if t[3] > sizelim2:
                            self.bboxes += [[np.concatenate([[i],t])]]*2  # 大于30毫米的多录入2次
                        if t[3] > sizelim3:
                            self.bboxes += [[np.concatenate([[i],t])]]*4  # 大于40毫米的多录入6次
            
            self.bboxes = np.concatenate(self.bboxes, axis=0)
            # print('self.bboxes.shape = ',self.bboxes.shape)
            # print('self.bboxes = ',self.bboxes)

        self.crop = Crop(config)
        self.label_mapping = LabelMapping(config, self.phase)

    def __getitem__(self, idx, split=None):  # 获取第idx个样本，所有样本分为70%正例和30%反例。每个样本包括图块和对应的标签。
        t = time.time()
        np.random.seed(int(str(t % 1)[2:7]))  # seed according to time

        isRandomImg = False
        if self.phase != 'test':
            if idx >= len(self.bboxes):  # 如果idx大于结节总数，就是要进行随机生成30%的反例了
                isRandom = True  # 这个变量控制着当前是生成正例还是反例，true代表反例，false代表正例
                idx = idx % len(self.bboxes)
                # isRandomImg = np.random.randint(2) # 这个变量控制着是从所有train数据集还是从kaggle数据集生成反例（作者采用的train数据集包括luna和kaggle两个数据集）
            else:
                isRandom = False
        else:
            isRandom = False
        
        if self.phase != 'test':
            if not isRandomImg:  #当需要生成70%正例、或使用所有train数据集生成30%反例时，在所有的train数据集中（作者采用的是luna+kaggle数据集）生成正例、反例
                bbox = self.bboxes[idx]  # 获取第idx个结节
                filename = self.filenames[int(bbox[0])]  # 获取当前第idx个结节对应的肺部图片文件名的索引
                #print('bbox = ',bbox)
                imgs = np.load(filename)  # imgs.shape=[1,depth,height,width]
                
                bboxes = self.sample_bboxes[int(bbox[0])]  # 获取肺部图片对应的所有结节
                isScale = self.augtype['scale'] and (self.phase=='train')  # 当生成训练数据时，包含结节的图块需要在一定范围内随机缩放
                sample, target, bboxes, coord = self.crop(imgs, bbox[1:], bboxes,isScale,isRandom)
                if self.phase=='train' and not isRandom:  # 当处于训练阶段，且生成的是正例时，对正例做数据增强
                     sample, target, bboxes, coord = augment(sample, target, bboxes, coord,
                        ifflip=self.augtype['flip'], ifrotate=self.augtype['rotate'], ifswap = self.augtype['swap'])
            else:  # 使用kaggle数据。因为我们没有kaggle数据，所以isRandomImg一直为False
                randimid = np.random.randint(len(self.kagglenames))
                filename = self.kagglenames[randimid]
                imgs = np.load(filename)
                bboxes = self.sample_bboxes[randimid]
                isScale = self.augtype['scale'] and (self.phase=='train')
                sample, target, bboxes, coord = self.crop(imgs, [], bboxes,isScale=False,isRand=True)
                
            label = self.label_mapping(sample.shape[1:], target, bboxes)
            sample = (sample.astype(np.float32)-128)/128
            #if filename in self.kagglenames and self.phase=='train':
            #    label[label==-1]=0
            #print('label.shape = ',label.shape)
            #print('label[0] = ',label[0])
            return torch.from_numpy(sample), torch.from_numpy(label), coord
        else:
            imgs = np.load(self.filenames[idx])
            bboxes = self.sample_bboxes[idx]
            nz, nh, nw = imgs.shape[1:]
            pz = int(np.ceil(float(nz) / self.stride)) * self.stride
            ph = int(np.ceil(float(nh) / self.stride)) * self.stride  
            pw = int(np.ceil(float(nw) / self.stride)) * self.stride
            imgs = np.pad(imgs, [[0,0],[0, pz - nz], [0, ph - nh], [0, pw - nw]], 'constant',constant_values = self.pad_value)
            
            xx,yy,zz = np.meshgrid(np.linspace(-0.5,0.5,imgs.shape[1]/self.stride),
                                   np.linspace(-0.5,0.5,imgs.shape[2]/self.stride),
                                   np.linspace(-0.5,0.5,imgs.shape[3]/self.stride),indexing ='ij')
            coord = np.concatenate([xx[np.newaxis,...], yy[np.newaxis,...],zz[np.newaxis,:]],0).astype('float32') # 所有anchor的中心点坐标（归一化后的），注意，所有的坐标范围是[-0.5,0.5]
            imgs, nzhw = self.split_comber.split(imgs)  # 当前肺部扫描分割出的所有1*208*208*208图块，nzhw是一个数组，包含三个元素nz,nh,nw，即每个维度分割出了多少个144（也可以理解为分割出了多少个208）
            coord2, nzhw2 = self.split_comber.split(coord,
                                                   side_len=self.split_comber.side_len/self.stride,
                                                   max_stride=self.split_comber.max_stride/self.stride,
                                                   margin=self.split_comber.margin/self.stride)  # 对anchor的中心点坐标做同样的分割，使得每个208*208*208图块有与之对应的anchor，且anchor的坐标为原肺部图坐标
            assert np.all(nzhw==nzhw2)
            imgs = (imgs.astype(np.float32)-128)/128 # 对图片归一化
            return torch.from_numpy(imgs), bboxes, torch.from_numpy(coord2), np.array(nzhw)  # 类型属于torch.tensor的会被封装成长度为batch size的List，其它的类型会被封装成长度为batch size的tuple

    def __len__(self):
        if self.phase == 'train':
            return len(self.bboxes)/(1-self.r_rand) # train阶段的样本总量为：结节总数/0.7
        elif self.phase == 'val':
            return len(self.bboxes)  # val阶段的样本总量为：结节总数
        else:
            return len(self.sample_bboxes)  # test阶段的样本总量：肺部扫描的文件数
        
        
def augment(sample, target, bboxes, coord, ifflip=True, ifrotate=True, ifswap=True):
    #                     angle1 = np.random.rand()*180
    if ifrotate:
        validrot = False
        counter = 0
        while not validrot:
            newtarget = np.copy(target)
            angle1 = np.random.rand()*180
            size = np.array(sample.shape[2:4]).astype('float')
            rotmat = np.array([[np.cos(angle1/180*np.pi),-np.sin(angle1/180*np.pi)],[np.sin(angle1/180*np.pi),np.cos(angle1/180*np.pi)]])
            newtarget[1:3] = np.dot(rotmat,target[1:3]-size/2)+size/2
            if np.all(newtarget[:3]>target[3]) and np.all(newtarget[:3]< np.array(sample.shape[1:4])-newtarget[3]):
                validrot = True
                target = newtarget
                sample = rotate(sample,angle1,axes=(2,3),reshape=False)
                coord = rotate(coord,angle1,axes=(2,3),reshape=False)
                for box in bboxes:
                    box[1:3] = np.dot(rotmat,box[1:3]-size/2)+size/2
            else:
                counter += 1
                if counter ==3:
                    break
    if ifswap:
        if sample.shape[1]==sample.shape[2] and sample.shape[1]==sample.shape[3]:
            axisorder = np.random.permutation(3)
            sample = np.transpose(sample,np.concatenate([[0],axisorder+1]))
            coord = np.transpose(coord,np.concatenate([[0],axisorder+1]))
            target[:3] = target[:3][axisorder]
            bboxes[:,:3] = bboxes[:,:3][:,axisorder]
            
    if ifflip:
        # flipid = np.array([np.random.randint(2),np.random.randint(2),np.random.randint(2)])*2-1
        flipid = np.array([1,np.random.randint(2),np.random.randint(2)])*2-1
        sample = np.ascontiguousarray(sample[:,::flipid[0],::flipid[1],::flipid[2]])
        coord = np.ascontiguousarray(coord[:,::flipid[0],::flipid[1],::flipid[2]])
        for ax in range(3):
            if flipid[ax]==-1:
                target[ax] = np.array(sample.shape[ax+1])-target[ax]
                bboxes[:,ax]= np.array(sample.shape[ax+1])-bboxes[:,ax]
    return sample, target, bboxes, coord 


class Crop(object):
    def __init__(self, config):
        self.crop_size = config['crop_size'] # =[128, 128, 128]
        self.bound_size = config['bound_size'] # =12
        self.stride = config['stride'] # =4
        self.pad_value = config['pad_value'] # =170
    def __call__(self, imgs, target, bboxes,isScale=False,isRand=False):
        if isScale:
            radiusLim = [8.,120.]
            scaleLim = [0.75,1.25]
            scaleRange = [np.min([np.max([(radiusLim[0]/target[3]),scaleLim[0]]), 1])
                         ,np.max([np.min([(radiusLim[1]/target[3]),scaleLim[1]]), 1])]
            scale = np.random.rand()*(scaleRange[1]-scaleRange[0])+scaleRange[0]  # 待研究，初步判断为缩放比例scale的范围是[0.75,1.25]。注意，这里的比例的用法是 缩放后的待裁剪尺寸=[128,128,128]/scale
            crop_size = (np.array(self.crop_size).astype('float')/scale).astype('int')  # 对裁剪尺寸进行缩放
        else:
            crop_size=self.crop_size
        bound_size = self.bound_size
        target = np.copy(target)  # target是第idx个结节的坐标和直径信息
        bboxes = np.copy(bboxes)  # bboxes是当前图片的所有肺结节的坐标和直径信息
        
        start = []
        for i in range(3):
            if not isRand: # 当idx小于等于bboxes数量的时候，即需要生成70%的正例时，不随机裁剪，裁剪的块中至少包含一个肺结节
                r = target[3] / 2
                s = np.floor(target[i] - r)+ 1 - bound_size
                e = np.ceil (target[i] + r)+ 1 + bound_size - crop_size[i]
                #print('target = ',target)
                #print('s = ',s,'    e = ',e) 
            else: # 当idx大于bboxes数量时，idx=idx%len(bboxes)，需要通过随机裁剪生成30%的反例
                #print('-'*70)
                #print('isRand = true')
                s = np.max([imgs.shape[i+1]-crop_size[i]/2,imgs.shape[i+1]/2+bound_size])
                e = np.min([crop_size[i]/2,              imgs.shape[i+1]/2-bound_size])
                target = np.array([np.nan,np.nan,np.nan,np.nan])
                #print('s = ',s,'    e = ',e,'    imgs.shape[i+1] = ',imgs.shape[i+1],'    crop_size[i] = ',crop_size[i])
                #print('-'*70)
            if s>e: # 当生成70%的正例时，几乎所有的都是s>e
                start.append(np.random.randint(e,s))#!
            else:
                start.append(int(target[i])-crop_size[i]/2+np.random.randint(-bound_size/2,bound_size/2))
            # 以上操作是找到开始切出128*128*128块的点start，然后从start开始切出一个128*128*128的块
                
                
        normstart = np.array(start).astype('float32')/np.array(imgs.shape[1:])-0.5 # 将start点归一化
        normsize = np.array(crop_size).astype('float32')/np.array(imgs.shape[1:])
        xx,yy,zz = np.meshgrid(np.linspace(normstart[0],normstart[0]+normsize[0],self.crop_size[0]/self.stride),
                           np.linspace(normstart[1],normstart[1]+normsize[1],self.crop_size[1]/self.stride),
                           np.linspace(normstart[2],normstart[2]+normsize[2],self.crop_size[2]/self.stride),indexing ='ij')
        coord = np.concatenate([xx[np.newaxis,...], yy[np.newaxis,...],zz[np.newaxis,:]],0).astype('float32') # 32*32*32个anchor在CT原图中的坐标。其中coord[0]中的z,h,w三个值，又是裁剪图块的起始点。

        pad = []
        pad.append([0,0])
        for i in range(3):
            leftpad = max(0,-start[i])
            rightpad = max(0,start[i]+crop_size[i]-imgs.shape[i+1])
            pad.append([leftpad,rightpad])
        crop = imgs[:,
            max(start[0],0):min(start[0] + crop_size[0],imgs.shape[1]),
            max(start[1],0):min(start[1] + crop_size[1],imgs.shape[2]),
            max(start[2],0):min(start[2] + crop_size[2],imgs.shape[3])] # 注意，这里有可能裁剪的尺寸不够crop_size。
        # （这里我发现了一个点比较左右为难：肺结节有大有小，如果想对结节大小比较有鲁棒性，则需要裁剪出包含结节的图块时，需要做缩放操作。
        #   但做缩放操作的同时，又破坏了图像采样的固定性——即：将所有图片重采样到单个像素长宽高为1*1*1毫米）
        crop = np.pad(crop,pad,'constant',constant_values =self.pad_value) # 剪切出形状为[1，128/scale,128/scale,128/scale]的原始肺部图像块
        #print '裁剪的原图大小为：',crop.shape
        for i in range(3): # 将结节中心点相对于当前整个肺部图块的坐标，改成相对于当前裁剪块的坐标
            target[i] = target[i] - start[i] 
        for i in range(len(bboxes)): # 将同文件的其它肺结节的相对于当前整个肺部图块的坐标，改成相对于当前裁剪块的坐标
            for j in range(3):
                bboxes[i][j] = bboxes[i][j] - start[j] 
                
        if isScale:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                crop = zoom(crop,[1,scale,scale,scale],order=1) # 将剪切的图块按之前的scale缩放回来。注意，这里有时并不能精确地缩放回128*128*128
                #print '裁剪的图缩放到128*128*128：',crop.shape
            newpad = self.crop_size[0]-crop.shape[1:][0] # 看缩放后的图的尺寸和128*128*128每个维度相差几个元素。
            #print('newpad = ',newpad)
            if newpad<0:
                crop = crop[:,:-newpad,:-newpad,:-newpad]
            elif newpad>0: # 将crop填充回128*128*128
                pad2 = [[0,0],[0,newpad],[0,newpad],[0,newpad]]
                crop = np.pad(crop,pad2,'constant',constant_values =self.pad_value)
            for i in range(4):
                target[i] = target[i]*scale  # 将第idx个结节的相对裁剪块的中心点坐标和直径也像裁剪的图块一样缩放
            for i in range(len(bboxes)):
                for j in range(4):
                    bboxes[i][j] = bboxes[i][j]*scale # 将同文件的其它结节相对于裁剪块的中心点坐标和直径也像裁剪的块一样缩放
        return crop, target, bboxes, coord  # 返回1*128*128*128的图块、第idx个结节坐标（相对于图块）和直径、同文件内其它结节的坐标（相对于图块）和直径、对于当前图块需要生成的所有anchor的中心坐标（相对于原图）


class LabelMapping(object):
    def __init__(self, config, phase):
        self.stride = np.array(config['stride']) # =4
        self.num_neg = int(config['num_neg']) # =800
        self.th_neg = config['th_neg'] # =0.02
        self.anchors = np.asarray(config['anchors']) # =[ 10.0, 30.0, 60.]
        self.phase = phase
        if phase == 'train':
            self.th_pos = config['th_pos_train'] # =0.5
        elif phase == 'val':
            self.th_pos = config['th_pos_val'] # =1
            
    def __call__(self, input_size, target, bboxes):  # 该函数返回当前图块的标签，标签形状为[32,32,32,3,5]。注意，在这32*32*32*3个anchor中，只有一个被记为正例，反例最多为800个，剩下的anchor不参与损失函数的计算
        stride = self.stride
        num_neg = self.num_neg
        th_neg = self.th_neg
        anchors = self.anchors
        th_pos = self.th_pos
        
        output_size = []
        for i in range(3):
            assert(input_size[i] % stride == 0)
            output_size.append(input_size[i] / stride)
        
        label = -1 * np.ones(output_size + [len(anchors), 5], np.float32)
        offset = ((stride.astype('float')) - 1) / 2 # offset = 1.5
        #print('offset = ',offset)
        oz = np.arange(offset, offset + stride * (output_size[0] - 1) + 1, stride) # =[1.5,5.5,9.5,......,125.5]
        oh = np.arange(offset, offset + stride * (output_size[1] - 1) + 1, stride)
        ow = np.arange(offset, offset + stride * (output_size[2] - 1) + 1, stride)
        #print('oz.shape = ',oz.shape)
        #print('oz = ',oz)
        #print('oh.shape = ',oh.shape)
        #print('oh = ',oh)
        #print('ow.shape = ',ow.shape)
        #print('ow = ',ow)
        

        for bbox in bboxes:
            for i, anchor in enumerate(anchors):
                iz, ih, iw = select_samples(bbox, anchor, th_neg, oz, oh, ow) # 返回bbox和anchor交并比满足阈值的anchor的滑动窗口位置（不是中心点坐标）。这一句的目的其实只是为了将属于各个bbox的所有反例找出来
                label[iz, ih, iw, i, 0] = 0

        if self.phase == 'train' and self.num_neg > 0:
            neg_z, neg_h, neg_w, neg_a = np.where(label[:, :, :, :, 0] == -1) # 属于反例的anchor的滑动位置与anchor索引（RPN在feature map上滑动窗口的位置）数组
            neg_idcs = random.sample(range(len(neg_z)), min(num_neg, len(neg_z))) # 随机选取最多800个属于反例的anchor
            neg_z, neg_h, neg_w, neg_a = neg_z[neg_idcs], neg_h[neg_idcs], neg_w[neg_idcs], neg_a[neg_idcs]
            label[:, :, :, :, 0] = 0
            label[neg_z, neg_h, neg_w, neg_a, 0] = -1 # -1代表反例

        if np.isnan(target[0]):
            return label
        iz, ih, iw, ia = [], [], [], []
        for i, anchor in enumerate(anchors):
            iiz, iih, iiw = select_samples(target, anchor, th_pos, oz, oh, ow) # 返回target和anchor交并比满足阈值的anchor的位置，这里是正例的阈值，所以这些位置对应的类别应该设置为正例
            iz.append(iiz)
            ih.append(iih)
            iw.append(iiw)
            ia.append(i * np.ones((len(iiz),), np.int64)) # 属于正例的这些anchor的边长索引
        iz = np.concatenate(iz, 0) # 将所有属于当前target的正例的anchor的位置坐标连接起来
        ih = np.concatenate(ih, 0)
        iw = np.concatenate(iw, 0)
        ia = np.concatenate(ia, 0)
        flag = True 
        if len(iz) == 0: # 如果没有一个anchor是属于当前target的正例（比如target直径相对于anchor直径来说太小或者太大），则要保证至少有一个正例
            #没有一个anchor是属于当前target的正例，则保留与target最接近的一个anchor，将其设置为target的正例
            pos = []
            for i in range(3):
                pos.append(max(0, int(np.round((target[i] - offset) / stride))))
            idx = np.argmin(np.abs(np.log(target[3] / anchors)))
            pos.append(idx)
            flag = False
        else:
            idx = random.sample(range(len(iz)), 1)[0] # 在众多属于target的正例的anchor中，只随机选取一个作为正例？奇怪，跟faster rcnn 不一样啊，这样会不会造成正例正例数量太少了导致假阳太多？又或者是有意为之？
            pos = [iz[idx], ih[idx], iw[idx], ia[idx]]
        dz = (target[0] - oz[pos[0]]) / anchors[pos[3]]
        dh = (target[1] - oh[pos[1]]) / anchors[pos[3]]
        dw = (target[2] - ow[pos[2]]) / anchors[pos[3]]
        dd = np.log(target[3] / anchors[pos[3]])
        label[pos[0], pos[1], pos[2], pos[3], :] = [1, dz, dh, dw, dd]
        
        
        positive_negative_ignore = label[:,:,:,:,0]
        num_positive = len(np.where(positive_negative_ignore==1)[0])
        num_negative = len(np.where(positive_negative_ignore==-1)[0])
        num_ignore = len(np.where(positive_negative_ignore==0)[0])
        assert num_positive+num_negative+num_ignore == 32*32*32*3
        #print '正例有{%d}个'%num_positive,'反例有{%d}个'%num_negative,'忽略的有{%d}'%num_ignore,'    num_positive+num_negative+num_ignore == 32*32*32*3'
        return label # 此处返回的label形状为[32,32,32,3,5]，3为anchor数，5为每个anchor的标签，格式为（class,dz,dh,dw,dd）。其中反例class为-1，正例class为1，当class=0时忽略

def select_samples(bbox, anchor, th, oz, oh, ow): # 该函数返回bbox和anchor交并比满足阈值的anchor的滑动窗口位置（不是中心点坐标）
    #这个方法之所以没有用穷举法产生所有anchor和标签，可能是基于效率和处理时长的考虑（穷举法太耗时？）
    z, h, w, d = bbox # z,h,w是bbox相对于图块的中心点坐标
    max_overlap = min(d, anchor) # bbox和anchor中心点重合时，重叠区域的最大边长
    min_overlap = np.power(max(d, anchor), 3) * th / max_overlap / max_overlap # bbox和anchor中心点重合时，重叠区域的最小边长（满足阈值的）
    #print('d = ',d,'    anchor = ',anchor)
    #print('max_overlap = ',max_overlap)
    #print('min_overlap = ',min_overlap)
    if min_overlap > max_overlap: # 说明当前bbox和anchor不匹配（要么是结节过小，要么是anchor过小）
        return np.zeros((0,), np.int64), np.zeros((0,), np.int64), np.zeros((0,), np.int64)
    else:
        s = z - 0.5 * np.abs(d - anchor) - (max_overlap - min_overlap)
        e = z + 0.5 * np.abs(d - anchor) + (max_overlap - min_overlap)
        mz = np.logical_and(oz >= s, oz <= e)
        iz = np.where(mz)[0]
        
        s = h - 0.5 * np.abs(d - anchor) - (max_overlap - min_overlap)
        e = h + 0.5 * np.abs(d - anchor) + (max_overlap - min_overlap)
        mh = np.logical_and(oh >= s, oh <= e)
        ih = np.where(mh)[0]
            
        s = w - 0.5 * np.abs(d - anchor) - (max_overlap - min_overlap) # 与当前bbox对应的anchor中心点坐标的最小值
        e = w + 0.5 * np.abs(d - anchor) + (max_overlap - min_overlap) # 与当前bbox对应的anchor中心点坐标的最大值
        mw = np.logical_and(ow >= s, ow <= e)
        iw = np.where(mw)[0]

        if len(iz) == 0 or len(ih) == 0 or len(iw) == 0: # 与当前bbox对应的anchor中心点范围超出当前的图块区域
            return np.zeros((0,), np.int64), np.zeros((0,), np.int64), np.zeros((0,), np.int64)
        
        lz, lh, lw = len(iz), len(ih), len(iw)
        #print('lz, lh, lw = ',lz, lh, lw)
        #print('oz[iz] = ',oz[iz])
        #print('oh[ih] = ',oh[ih])
        #print('ow[iw] = ',ow[iw])
        iz = iz.reshape((-1, 1, 1))
        ih = ih.reshape((1, -1, 1))
        iw = iw.reshape((1, 1, -1))
        # 思考：这里可不可以用np.meshgrid构造所有centers？
        iz = np.tile(iz, (1, lh, lw)).reshape((-1)) # z轴上的每一个切片（二维矩阵）上的所有值都是一样的。然后再reshape成一维数组
        ih = np.tile(ih, (lz, 1, lw)).reshape((-1))
        iw = np.tile(iw, (lz, lh, 1)).reshape((-1))
        centers = np.concatenate([
            oz[iz].reshape((-1, 1)),
            oh[ih].reshape((-1, 1)),
            ow[iw].reshape((-1, 1))], axis = 1)
        #centers.shape = [lz*lh*lw,3]，即每行为一个anchor的中心点坐标
        
        r0 = anchor / 2
        s0 = centers - r0 # anchor的对角坐标起始点(z1,h1,w1)，类似于二维bbox中的(x1,y1)
        e0 = centers + r0 # anchor的对角坐标终止点(z2,h2,w2)，类似于二维bbox中的(x2,y2)
        
        r1 = d / 2
        s1 = bbox[:3] - r1
        s1 = s1.reshape((1, -1)) # bbox的对角坐标起始点(z1,h1,w1)
        e1 = bbox[:3] + r1
        e1 = e1.reshape((1, -1)) # bbox的对角坐标终止点(z2,h2,w2)
        
        overlap = np.maximum(0, np.minimum(e0, e1) - np.maximum(s0, s1)) # 重叠区域的边长[dz,dh,dw]
        
        intersection = overlap[:, 0] * overlap[:, 1] * overlap[:, 2] # bbox和anchors交集的体积
        union = anchor * anchor * anchor + d * d * d - intersection # 并集的体积

        iou = intersection / union # 交并比

        mask = iou >= th # 交并比大于等于th的才是可以属于正例的anchor
        #if th > 0.4:
         #   if np.sum(mask) == 0:
          #      print(['iou not large', iou.max()])
           # else:
            #    print(['iou large', iou[mask]])
        iz = iz[mask]
        ih = ih[mask]
        iw = iw[mask]
        # 将属于正例的anchor的坐标点索引返回
        return iz, ih, iw

def collate(batch):
    if torch.is_tensor(batch[0]):
        return [b.unsqueeze(0) for b in batch]
    elif isinstance(batch[0], np.ndarray):
        return batch
    elif isinstance(batch[0], int):
        return torch.LongTensor(batch)
    elif isinstance(batch[0], collections.Iterable):
        transposed = zip(*batch)
        return [collate(samples) for samples in transposed]

