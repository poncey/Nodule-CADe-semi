# -*- coding:UTF-8 -*-
import numpy as np
import os
import cv2
from pandas import read_csv
import torch
from torch.utils.data import Dataset
from tqdm import tqdm


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
        nodule_image = extract_nodule(lung_img, centre)

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


def extract_nodule(lung_image, centre):
    # Extract 64 * 64 * 64 nodules
    centre = np.asarray(centre, dtype=np.int16)
    
    if len(lung_image.shape) == 4:
        assert lung_image.shape[0] == 1
        lung_image = lung_image.reshape(lung_image.shape[1], lung_image.shape[2], lung_image.shape[3])

    # padding width (0-th dimension)
    pad_up = np.zeros((128, lung_image.shape[1], lung_image.shape[2]))
    pad_down = np.zeros((128, lung_image.shape[1], lung_image.shape[2]))
    lung_image = np.concatenate((pad_up, lung_image), axis=0)
    lung_image = np.concatenate((lung_image, pad_down), axis=0)

    # padding length (1-th dimension)
    pad_up = np.zeros((lung_image.shape[0], 128, lung_image.shape[2]))
    pad_down = np.zeros((lung_image.shape[0], 128, lung_image.shape[2]))
    lung_image = np.concatenate((pad_up, lung_image), axis=1)
    lung_image = np.concatenate((lung_image, pad_down), axis=1)

    # padding height (2-th dimension)
    pad_up = np.zeros((lung_image.shape[0], lung_image.shape[1], 128))
    pad_down = np.zeros((lung_image.shape[0], lung_image.shape[1], 128))
    lung_image = np.concatenate((pad_up, lung_image), axis=2)
    lung_image = np.concatenate((lung_image, pad_down), axis=2)

    centre = centre + 128
    # print "shape after padding: ", lung_image.shape
    # print "centre of nodule after padding: ", centre

    # Extraction
    nodule_image = lung_image[
        centre[0] - 32: centre[0] + 32,
        centre[1] - 32: centre[1] + 32,
        centre[2] - 32: centre[2] + 32
    ]

    # Verification of result
    if True in (np.asarray(nodule_image.shape) != np.asarray((64, 64, 64))):
        raise Exception('Nodule shape' + nodule_image.shape + ' is not right.')

    return nodule_image.reshape(1, nodule_image.shape[0], nodule_image.shape[1], nodule_image.shape[2])


def save_3d_image(image, name):
    direction = 'saving-imgs'
    if not os.path.exists(direction):
        os.makedirs(direction)
    name = name + '.jpg'
    cv2.imwrite(os.path.join(direction, name), image)


# TODO: Verify it with import entire dataset
def load_data(dataset, save_dir='./'):

    save_dir = os.path.join(save_dir, 'fold%d' % dataset.fold, dataset.phase)
    if dataset.phase == 'train':

        if not os.path.exists(save_dir):
            print "save_direction not exist, saving training data..."
            os.makedirs(save_dir)
            # save training dataset
            for i in tqdm(range(len(dataset))):
                image, label = dataset[i]
                image = np.expand_dims(image, axis=0)
                np.save(os.path.join(save_dir, '%03d_image.npy' % i), image)  # save the image
                np.save(os.path.join(save_dir, '%03d_label.npy' % i), label)  # save the label
            print "Generation of files completed, direction: %s" % save_dir
            print

        # generate data to memory
        print "Move training data into memory..."
        images = []
        labels = []
        for i in tqdm(range(len(dataset))):
            image = np.load(os.path.join(save_dir, '%03d_image.npy' % i))
            label = np.load(os.path.join(save_dir, '%03d_label.npy' % i))
            images.append(image)
            labels.append(label)
        images = np.concatenate(images, axis=0)
        labels = np.asarray(labels)
        assert images.shape[0] == len(dataset)
        return torch.Tensor(images), torch.LongTensor(labels)

    if dataset.phase == 'unlabeled':

        if not os.path.exists(save_dir):
            print "save_directions are empty, saving unlabeled data..."
            os.makedirs(save_dir)
            # save unlabeled dataset
            for i in tqdm(range(len(dataset))):
                image = dataset[i]
                image = np.expand_dims(image, axis=0)
                np.save(os.path.join(save_dir, '%03d_image.npy' % i), image)  # save the image
            print "Generation of files completed, direction: %s" % save_dir
            print

        print "Move unlabeled data into memory..."
        images = []
        for i in tqdm(range(len(dataset))):
            image = np.load(os.path.join(save_dir, '%03d_image.npy' % i))
            images.append(image)
        images = np.concatenate(images, axis=0)
        assert images.shape[0] == len(dataset)
        return torch.Tensor(images)

    if dataset.phase == 'test':

        if not os.path.exists(save_dir):
            print "save_directions are empty, saving testing data..."
            os.makedirs(save_dir)
            # save test dataset
            for i in tqdm(range(len(dataset))):
                image, label, series_uid, centre = dataset[i]
                image = np.expand_dims(image, axis=0)
                np.save(os.path.join(save_dir, '%03d_image.npy' % i), image)
                np.save(os.path.join(save_dir, '%03d_label.npy' % i), label)
                np.save(os.path.join(save_dir, '%03d_uid.npy' % i), series_uid)
                np.save(os.path.join(save_dir, '%03d_centre.npy' % i), centre)

            print "Generation of files completed, direction: %s" % save_dir
            print

        print "Move test data into memory..."
        images = []
        labels = []
        uids = []
        centres = []
        for i in tqdm(range(len(dataset))):
            image = np.load(os.path.join(save_dir, '%03d_image.npy' % i))
            images.append(image)

            label = np.load(os.path.join(save_dir, '%03d_label.npy' % i))
            labels.append(label)

            series_uid = np.load(os.path.join(save_dir, '%03d_uid.npy' % i))
            uids.append(series_uid)

            centre = np.load(os.path.join(save_dir, '%03d_centre.npy' % i))
            centre = centre.reshape(1, 3)
            centres.append(centre)
        images = np.concatenate(images, axis=0)
        labels = np.asarray(labels)
        uids = np.asarray(uids)
        centres = np.concatenate(centres, axis=0)
        return torch.Tensor(images), torch.LongTensor(labels), uids, centres


