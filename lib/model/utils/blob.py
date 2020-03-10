# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Blob helper functions."""

import numpy as np
# from scipy.misc import imread, imresize
from model.utils.config import cfg
import skimage.transform
import pdb
import cv2

try:
    xrange          # Python 2
except NameError:
    xrange = range  # Python 3


def im_list_to_blob(ims):
    """Convert a list of images into a network input.

    Assumes images are already prepared (means subtracted, BGR order, ...).
    """
    max_shape = np.array([im.shape for im in ims]).max(axis=0)
    num_images = len(ims)
    blob = np.zeros((num_images, max_shape[0], max_shape[1], 3),
                    dtype=np.float32)
    for i in xrange(num_images):
        im = ims[i]
        blob[i, 0:im.shape[0], 0:im.shape[1], :] = im

    return blob

def prep_im_for_blob(im, pixel_means, target_size, max_size, is_ws, training):
    """Mean subtract and scale an image for use in a blob."""

    """ original """
    # im = im.astype(np.float32, copy=False)
    # im -= pixel_means
    # # im = im[:, :, ::-1]
    # # might need to alter pixel means
    # im_shape = im.shape
    # im_size_min = np.min(im_shape[0:2])
    # im_size_max = np.max(im_shape[0:2])
    # # larger, smaller of the width and height
    # im_scale = float(target_size) / float(im_size_min)
    # # Prevent the biggest axis from being more than MAX_SIZE
    # if np.round(im_scale * im_size_max) > max_size:
    #     im_scale = float(max_size) / float(im_size_max)
    # # im = imresize(im, im_scale)
    # im = cv2.resize(im, None, None, fx=im_scale, fy=im_scale,
    #                 interpolation=cv2.INTER_LINEAR)

    #XXX modified: apply augmentation
    im = im.astype(np.float32, copy=False) / 255.

    if is_ws:
        if cfg.TRAIN.USE_ROTATION:
            im = skimage.transform.rotate(im, np.random.uniform(-cfg.TRAIN.ROTATION_MAX_ANGLE,cfg.TRAIN.ROTATION_MAX_ANGLE), cval=pixel_means[0][0][0]/255.)
                                
        if cfg.TRAIN.USE_CROPPING:
            offsets_u = np.random.random_integers(0,cfg.TRAIN.CROPPING_MAX_MARGIN*im.shape[0])
            offsets_d = np.random.random_integers(1,cfg.TRAIN.CROPPING_MAX_MARGIN*im.shape[0])
            offsets_l = np.random.random_integers(0,cfg.TRAIN.CROPPING_MAX_MARGIN*im.shape[1])
            offsets_r = np.random.random_integers(1,cfg.TRAIN.CROPPING_MAX_MARGIN*im.shape[1])
            im = im[offsets_u:-offsets_d,offsets_l:-offsets_r,:]

    if training:
        if cfg.TRAIN.USE_BRIGHTNESS_ADJUSTMENT:
            im += np.random.uniform(-cfg.TRAIN.BRIGHTNESS_ADJUSTMENT_MAX_DELTA,cfg.TRAIN.BRIGHTNESS_ADJUSTMENT_MAX_DELTA)
            im = np.clip(im, 0, 1)
    
        if cfg.TRAIN.USE_CONTRAST_ADJUSTMENT:
            mm = np.mean(im)
            im = (im-mm)*np.random.uniform(cfg.TRAIN.CONTRAST_ADJUSTMENT_LOWER_FACTOR,cfg.TRAIN.CONTRAST_ADJUSTMENT_UPPER_FACTOR) + mm                         
            im = np.clip(im, 0, 1)

    im -= pixel_means/255.
    im_shape = im.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])
    im_scale = float(target_size) / float(im_size_min)
    # Prevent the biggest axis from being more than MAX_SIZE
    if np.round(im_scale * im_size_max) > max_size:
        im_scale = float(max_size) / float(im_size_max)

    im = skimage.transform.resize(im, [np.round(im_shape[0]*im_scale),np.round(im_shape[1]*im_scale)])*255.

    return im, im_scale