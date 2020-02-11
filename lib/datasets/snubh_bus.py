from __future__ import print_function
# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

import datasets
import datasets.snubh_bus
import os, sys
from datasets.imdb import imdb
import xml.dom.minidom as minidom
import numpy as np
import scipy.sparse
import subprocess
import pdb
import pickle

class snubh_bus(imdb):
    def __init__(self, image_set, data_path):
        imdb.__init__(self, image_set)
        self._image_set = image_set
        self._data_path = data_path
        self._classes_image = ('__background__','benign','malignant')
        self._classes = ('__background__','benign','malignant')
        self._class_to_ind_image = dict(zip(self._classes_image, range(3)))

        self._image_ext = ['.tiff']

        self._image_index = self._load_image_set_index()
        # Default to roidb handler
        self._roidb_handler = self.gt_roidb

        # Specific config options
        self.config = {'cleanup'  : True,
                       'use_salt' : True,
                       'top_k'    : 2000}

        assert os.path.exists(self._data_path), 'Path does not exist: {}'.format(self._data_path)

    def image_path_at(self, i):
        """
        Return the absolute path to image i in the image sequence.
        """
        return self.image_path_from_index(self._image_index[i])

    def image_path_from_index(self, index):
        """
        Construct an image path from the image's "index" identifier.
        """
        image_path = os.path.join(self._data_path, 'TIFFimages', index + self._image_ext[0])
        assert os.path.exists(image_path), 'path does not exist: {}'.format(image_path)
        return image_path

    def _load_image_set_index(self):
        """
        Load the indexes listed in this dataset's image set file.
        returns list of image file names
        """
        # Example path to image set file:
        # self._data_path + /ImageSets/val.txt

        image_set_file = os.path.join(self._data_path, 'ImageSets', 'Main','{}.txt'.format(self._image_set))
        image_index = []
        if os.path.exists(image_set_file):
            f = open(image_set_file, 'r')
            data = f.read().split()
            for lines in data:
                if lines != '':
                    image_index.append(lines)
            f.close()
            return image_index


    def gt_roidb(self):
        """
        Return the database of ground-truth regions of interest.
        This function loads/saves from/to a cache file to speed up future calls.

        returns gt_roidb : list of dictionaries
        """
        cache_file = os.path.join(self.cache_path, self.name + '_gt_roidb.pkl')
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = pickle.load(fid)
            print('{} gt roidb loaded from {}'.format(self.name, cache_file))
            return roidb

        gt_roidb = [self._load_imagenet_annotation(index)
                    for index in self.image_index]
        with open(cache_file, 'wb') as fid:
            pickle.dump(gt_roidb, fid, pickle.HIGHEST_PROTOCOL)
        print('wrote gt roidb to {}'.format(cache_file))

        return gt_roidb


    def _load_imagenet_annotation(self, index):
        """
        Load image and bounding boxes info from txt files of imagenet.
        """
        filename = os.path.join(self._data_path, 'Annotations', index + '.xml')

        # print 'Loading: {}'.format(filename)
        def get_data_from_tag(node, tag):
            return node.getElementsByTagName(tag)[0].childNodes[0].data

        with open(filename) as f:
            data = minidom.parseString(f.read())

        objs = data.getElementsByTagName('object')
        num_objs = len(objs)

        boxes = np.zeros((num_objs, 4), dtype=np.uint16)
        gt_classes = np.zeros((num_objs), dtype=np.int32)
        overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)

        birads = data.getElementsByTagName('BIRADS')
        label = int(get_data_from_tag(birads[0],'diag'))

        # Load object bounding boxes into a data frame.
        for ix, obj in enumerate(objs):
            if get_data_from_tag(obj, "name").lower().strip() == '__background__':
                continue
            cls = self._class_to_ind_image[str(get_data_from_tag(obj, "name")).lower().strip()]
            x1 = float(get_data_from_tag(obj, 'xmin'))
            y1 = float(get_data_from_tag(obj, 'ymin'))
            x2 = float(get_data_from_tag(obj, 'xmax'))
            y2 = float(get_data_from_tag(obj, 'ymax'))
            boxes[ix, :] = [x1, y1, x2, y2]
            gt_classes[ix] = cls
            overlaps[ix, cls] = 1.0

        overlaps = scipy.sparse.csr_matrix(overlaps)

        return {'boxes' : boxes,
                'gt_classes': gt_classes,
                'gt_overlaps' : overlaps,
                'flipped' : False,
                'im_label': label}