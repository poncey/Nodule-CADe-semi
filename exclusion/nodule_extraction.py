import numpy as np
import os
import cv2


def extract_nodule(lung_image, centre, size=(64, 64, 64)):
    if len(size) != 3:
        raise Exception("Wrong shape of size, it should be 3-d vector")
    size = np.asarray(size, dtype=np.int16)
    size_up = size // 2  # Cut as the length start from the centre
    size_down = size - size_up
    centre = np.asarray(centre, dtype=np.int16)

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

    return nodule_image


def save_3d_image(image, name):
    direction = 'saving-imgs'
    if not os.path.exists(direction):
        os.makedirs(direction)
    name = name + '.jpg'
    cv2.imwrite(os.path.join(dir, name), image)
