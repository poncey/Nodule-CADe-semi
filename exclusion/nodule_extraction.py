# -*- coding:UTF-8 -*-
import numpy as np
import os
import cv2
import csv
import torch
from torch.utils.data import Dataset


class ExculsionDataset(Dataset):
    def __init__(self, data_dir, index_dir, fold, phase='train'):
        assert(phase == 'train' or phase == 'unlabeled' or phase == 'test')
        self.data_dir = data_dir
        self.phase = phase
        self.fold = fold
        self.index_dir = index_dir

        if phase == 'test':
            load_dir = os.path.join(index_dir, 'fold%d' % fold, 'test.csv')
        elif phase == 'unlabeled':
            load_dir = os.path.join(index_dir, 'fold%d' % fold, 'unlabel.csv')
        elif phase == 'train':
            load_dir = os.path.join(index_dir, 'fold%d' % fold, 'total_train.csv')
        else:
            raise ValueError('Please check your phase.')


def extract_nodule(lung_image, centre, size):
    if len(size) != 3:
        raise Exception("Wrong shape of size, it should be 3-d vector")
    size = np.asarray(size, dtype=np.int16)
    size_up = size // 2  # Cut as the length start from the centre
    size_down = size - size_up
    centre = np.asarray(centre, dtype=np.int16)
    
    if len(lung_image.shape) == 4:
        assert lung_image.shape[0] == 1
        lung_image = lung_image.reshape(lung_image.shape[1], lung_image.shape[2], lung_image.shape[3])

    # padding width (0-th dimension)
    pad_up = np.zeros((size_up[0], lung_image.shape[1], lung_image.shape[2]))
    pad_down = np.zeros((size_down[0], lung_image.shape[1], lung_image.shape[2]))
    lung_image = np.concatenate((pad_up, lung_image), axis=0)
    lung_image = np.concatenate((lung_image, pad_down), axis=0)

    # padding length (1-th dimension)
    pad_up = np.zeros((lung_image.shape[0], size_up[1], lung_image.shape[2]))
    pad_down = np.zeros((lung_image.shape[0], size_down[1], lung_image.shape[2]))
    lung_image = np.concatenate((pad_up, lung_image), axis=1)
    lung_image = np.concatenate((lung_image, pad_down), axis=1)

    # padding height (2-th dimension)
    pad_up = np.zeros((lung_image.shape[0], lung_image.shape[1], size_up[2]))
    pad_down = np.zeros((lung_image.shape[0], lung_image.shape[1], size_down[2]))
    lung_image = np.concatenate((pad_up, lung_image), axis=2)
    lung_image = np.concatenate((lung_image, pad_down), axis=2)

    centre += size_up
    print "shape after padding: ", lung_image.shape
    print "centre of nodule after padding: ", centre

    # Extraction
    nodule_image = lung_image[
        centre[0] - size_up[0]: centre[0] + size_down[0],
        centre[1] - size_up[1]: centre[1] + size_down[1],
        centre[2] - size_up[2]: centre[2] + size_down[2]
    ]

    # Verification of result
    if True in (np.asarray(nodule_image.shape) != size):
        raise Exception('Nodule shape is not right.')

    return nodule_image.reshape(1, nodule_image.shape[0], nodule_image.shape[1], nodule_image.shape[2])


def save_3d_image(image, name):
    direction = 'saving-imgs'
    if not os.path.exists(direction):
        os.makedirs(direction)
    name = name + '.jpg'
    cv2.imwrite(os.path.join(direction, name), image)
    
'''

direction1 = '/home/user/work/DataBowl3/DSB2017-master/training/'
filepath = os.path.join(direction1,'fold',fold,'/')


def data_loader(fold,val):
    if val=='test':
        name1 = val + '.csv'
        data = os.path.join(direction1,name1)
        with open(data) as csvfile:
            reader = pd.read_csv(csvfile)
            sample_number =0;
            for row in reader:
                index=reader.loc([[row],['index']])
                (x,y,z) = reader.loc([[row],['x','y','z']])
                label = reader.loc([[row],['label']])
                lung=np.load(index+'_clean.npy')
                nodule_64=extract_nodule(lung,(x,y,z),(64,64,64))
                channel = nodule[:, :, :, 32].reshape(64,64)
                save_3d_image(nodule[:, :, :, 32].reshape(64,64),index+row)
                sample_number+=1
    else if val == 'total_train':
        name = val + '.csv'
        with open(name) as csvfile:
            reader = pd.read_csv(csvfile)
            sample_number =0;
            for row in reader:
                index=reader.loc([[row],['index']])
                (x,y,z) = reader.loc([[row],['x','y','z']])
                lung=np.load(index+'_clean.npy')
                nodule_64=extract_nodule(lung,(x,y,z),(64,64,64))
                save_3d_image(nodule[:, :, :, 32].reshape(64,64),'index'+row)
                sample_number+=1
    else if val == 'unlabel':
        name = val + '.csv'
        with open(name) as csvfile:
            reader = pd.read_csv(csvfile)
            sample_number =0;
            for row in reader:
                index=reader.loc([[row],['index']])
                (x,y,z) = reader.loc([[row],['x','y','z']])
                lung=np.load(index+'_clean.npy')
                nodule_64=extract_nodule(lung,(x,y,z),(64,64,64))
                save_3d_image(nodule[:, :, :, 32].reshape(64,64),'index'+row)
                sample_number +=1
    else raise Exception('Wrong val type,please check!')
    
    return tensor_32(sample_number,channel,label)
'''