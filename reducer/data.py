# -*- coding:UTF-8 -*-
import numpy as np
import os
import cv2
from pandas import read_csv
import torch
from torch.utils.data import Dataset


class ExclusionDataset(Dataset):
    def __init__(self, lung_dir, index_dir, fold, phase='train'):
        assert(phase == 'train' or phase == 'unlabeled' or phase == 'test')
        self.lung_dir = lung_dir
        self.phase = phase
        self.fold = fold
        self.shorter_list = np.asarray(read_csv(os.path.join(index_dir, 'shorter.csv'), header=None))

        if phase == 'test':
            self.index_list = read_csv(os.path.join(index_dir, 'fold%d' % fold, 'test.csv'))
        elif phase == 'unlabeled':
            self.index_list = read_csv(os.path.join(index_dir, 'fold%d' % fold, 'unlabel.csv'))
        elif phase == 'train':
            self.index_list = read_csv(os.path.join(index_dir, 'fold%d' % fold, 'total_train.csv'))
        else:
            raise ValueError('Please check your phase.')

    def __getitem__(self, idx):

        # Loading lung image
        lung_file = '%03d_clean.npy' % self.index_list['index'][idx]
        lung_img = np.load(os.path.join(self.lung_dir, lung_file))

        # extract the of nodule
        centre = (self.index_list['x'][idx],
                  self.index_list['y'][idx],
                  self.index_list['z'][idx])
        nodule_image = extract_nodule(lung_img, centre, size=[64, 64, 64])

        # Obtain the label
        if self.phase == 'train':
            # get label
            if not self.index_list['label'][idx]:
                label = 0
            else:
                label = 1
            return nodule_image, label
        elif self.phase == 'test':
            # get label
            if not self.index_list['label'][idx]:
                label = 0
            else:
                label = 1
            # name list for shorter File-ids
            xx, yy = np.where(self.shorter_list == self.index_list['index'][idx])
            series_uid = self.shorter_list[xx[0]][-1]
            return nodule_image, label, series_uid, centre
        else:
            return nodule_image

    def __len__(self):
        return len(self.index_list)


def extract_nodule(lung_image, centre, size):
    if len(size) != 3:
        raise Exception("Wrong shape of size, it should be single value or 3-d vector")
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
    # print "shape after padding: ", lung_image.shape
    # print "centre of nodule after padding: ", centre

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


def load_data(dataset, batch_indices):

    if dataset.phase == 'train':
        assert type(batch_indices) == np.ndarray
        images = []
        labels = []
        for i in batch_indices:
            image = dataset[i][0]
            image = np.expand_dims(image, axis=0)
            images.append(image)

            labels.append(dataset[i][1])
        images = np.concatenate(images, axis=0)
        labels = np.asarray(labels, dtype=np.int)

        return torch.from_numpy(images), torch.from_numpy(labels)

    if dataset.phase == 'unlabeled':
        assert type(batch_indices) == np.ndarray
        images = []
        for i in batch_indices:
            image = dataset[i]
            image = np.expand_dims(image, axis=0)
            images.append(image)

        images = np.concatenate(images, axis=0)
        return torch.from_numpy(images)

    if dataset.phase == 'test':
        assert type(batch_indices) == int
        image, label, file_id, centre = dataset[batch_indices]

        return torch.from_numpy(image), torch.from_numpy(np.asarray(label)), file_id, centre
