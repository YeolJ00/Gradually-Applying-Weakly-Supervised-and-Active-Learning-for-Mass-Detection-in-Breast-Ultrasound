# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Xinlei Chen
# --------------------------------------------------------

"""Compute minibatch blobs for training a Fast R-CNN network."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import numpy.random as npr
import imageio
from model.utils.config import cfg
from model.utils.blob import prep_im_for_blob, im_list_to_blob
import pdb
def get_minibatch(roidb, num_classes, training):
    """
    Given a roidb, construct a minibatch sampled from it.
    roidb is not roidb, it is minibatch_db from roibatchLoader. size is 1.
    """
    num_images = len(roidb)
    # Sample random scales to use for each image in this batch
    random_scale_inds = npr.randint(0, high=len(cfg.TRAIN.SCALES),
                                    size=num_images)
    assert(cfg.TRAIN.BATCH_SIZE % num_images == 0), \
        'num_images ({}) must divide BATCH_SIZE ({})'. \
        format(num_images, cfg.TRAIN.BATCH_SIZE)

    # Get the input image blob, formatted for caffe
    # return scaled blob and scale of blob, from roidb's information about the images
    im_blob, im_scales = _get_image_blob(roidb, random_scale_inds, training)

    blobs = {'data': im_blob}

    assert len(im_scales) == 1, "Single batch only"
    assert len(roidb) == 1, "Single batch only"
    
    # gt boxes: (x1, y1, x2, y2, cls)
    if cfg.TRAIN.USE_ALL_GT:
        # Include all ground truth boxes
        # we deleted [0] at the end of the code
        gt_inds = np.where(roidb[0]['gt_classes'] != 0)
    else:
        # For the COCO ground truth boxes, exclude the ones that are ''iscrowd'' 
        gt_inds = np.where((roidb[0]['gt_classes'] != 0) & np.all(roidb[0]['gt_overlaps'].toarray() > -1.0, axis=1))
    gt_boxes = np.empty((len(gt_inds[0]), 5), dtype=np.float32)
    gt_boxes[:, 0:4] = roidb[0]['boxes'][gt_inds, :] * im_scales[0]
    gt_boxes[:, 4] = roidb[0]['gt_classes'][gt_inds]
    blobs['gt_boxes'] = gt_boxes
    blobs['im_info'] = np.array([[im_blob[:].shape[1], im_blob[:].shape[2], im_scales[0]]], dtype=np.float32) # pylint bug
        # shape[1] is the width of the widest image
        # shape[2] is the height of a tallest image
        # im_scale[0] is the multiplier from the original image
    blobs['im_label'] = roidb[0]['im_label']
    blobs['file_name'] = roidb[0]['file_name']

    # blobs['img_id'] = roidb[0]['img_id']
    # might need to add img_path
    # OR this part might not be needed at all

    return blobs
    # blobs is the dictionary of
    # 1. 'data'     : blob, 4d list of images
    # 2. 'gt_boxes' : gt_boxes, list of gt_boxes(x1, y1, x2, y2, cls)
    # 3. 'im_info'' : list of blob width and height and scale
    # 4. 'im_label' : im_label of an image
    # 5. 'file_name': file name of an image

def _get_image_blob(roidb, scale_inds, training):
    """Builds an input blob from the images in the roidb at the specified
    scales.
    """
    num_images = len(roidb)

    processed_ims = []
    im_scales = []
    for i in range(num_images):
        '''
        This part might need to be changed
        delete cv2 related code,
        change to 2d if possible, of preferable
        '''
        #im = cv2.imread(roidb[i]['image'])
        im = imageio.imread(roidb[i]['image'])

        if len(im.shape) == 2:
            im = im[:,:,np.newaxis]
            im = np.concatenate((im,im,im), axis=2)
        # 2d image to 3d image

        # flip the channel, since the original one using cv2
        # rgb -> bgr
        im = im[:,:,::-1]

        if roidb[i]['flipped']:
            im = im[:, ::-1, :]
        # flip height-wise
        target_size = cfg.TRAIN.SCALES[scale_inds[i]]
        # 1 is always expected
        im, im_scale = prep_im_for_blob(im, cfg.PIXEL_MEANS, target_size,
                                        cfg.TRAIN.MAX_SIZE, training)
        # im is resized with im_scale ratio
        im_scales.append(im_scale)
        processed_ims.append(im)

    # Create a blob to hold the input images
    # change image lists to blob.
    blob = im_list_to_blob(processed_ims)

    return blob, im_scales
