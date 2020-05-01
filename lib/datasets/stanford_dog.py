from __future__ import print_function
# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

import datasets
import datasets.stanford_dog
import os, sys
from datasets.imdb import imdb
import xml.dom.minidom as minidom
import xml.etree.ElementTree as ET
from xml.etree.ElementTree import Element, SubElement, ElementTree
import numpy as np
import scipy.sparse
import subprocess
import pdb
import pickle
import torch
import random

class stanford_dog(imdb):
    def __init__(self, image_set, data_path):
        imdb.__init__(self, 'Stanford_Dog_' + image_set)
        self._image_set = image_set
        self._data_path = data_path
        self._classes_image = ('__background__','bloodhound','English_foxhound')
        self._classes = ('__background__','bloodhound','English_foxhound')
        self._class_to_ind_image = dict(zip(self._classes_image, range(3)))
        self._ind_to_class_image = dict(zip(range(3),self._classes_image))

        self._image_ext = ['.jpg']

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
        image_path = os.path.join(self._data_path, 'Images', index + self._image_ext[0])
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
        if not (self.name.startswith('al')):
            cache_file = os.path.join(self.cache_path, self.name + '_gt_roidb_master.pkl')
            if os.path.exists(cache_file):
                with open(cache_file, 'rb') as fid:
                    roidb = pickle.load(fid)
                print('{} gt roidb loaded from {}'.format(self.name, cache_file))
                return roidb

        gt_roidb = [self._load_imagenet_annotation(index)
                    for index in self.image_index]
        if not (self.name.startswith('al')):
            with open(cache_file, 'wb') as fid:
                pickle.dump(gt_roidb, fid, pickle.HIGHEST_PROTOCOL)
            print('wrote gt roidb to {}'.format(cache_file))

        return gt_roidb


    def _load_imagenet_annotation(self, index):
        """
        Load image and bounding boxes info from txt files of imagenet.
        """
        if self._image_set == 'al_train':
            filename = os.path.join(self._data_path, 'Annotations', index + '_AL.xml')
        else:
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

        file_name = data.getElementsByTagName('filename')[0].childNodes[0].data + 'f'
        
        # Load object bounding boxes into a data frame.
        for ix, obj in enumerate(objs):
            if get_data_from_tag(obj, "name").lower().strip() == '__background__':
                continue
            cls = self._class_to_ind_image[str(get_data_from_tag(obj, "name")).strip()]
            x1 = float(get_data_from_tag(obj, 'xmin'))
            y1 = float(get_data_from_tag(obj, 'ymin'))
            x2 = float(get_data_from_tag(obj, 'xmax'))
            y2 = float(get_data_from_tag(obj, 'ymax'))
            boxes[ix, :] = [x1, y1, x2, y2]
            gt_classes[ix] = cls
            overlaps[ix, cls] = 1.0
            label = cls - 1

        overlaps = scipy.sparse.csr_matrix(overlaps)

        return {'boxes' : boxes,
                'gt_classes': gt_classes,
                'gt_overlaps' : overlaps,
                'flipped' : False,
                'im_label': label,
                'file_name': file_name}

    def active_learning(self, all_boxes, thresh = 0.7, prob = 0.5):
        # all_boxes : (cls, image, boxes, 5) 5 = (x, y, x, y, cls_prob)
        # all_boxes_n:(cls, image_n, boxes, 5)
        # Call all annotations
        annopath = os.path.join(self._data_path,'Annotations','{:s}.xml')
        imagesetfile = os.path.join(self._data_path,'ImageSets','Main',self._image_set + '.txt')
        cachedir = os.path.join(self._data_path, 'annotations_cache')
        al_imageset_dir = os.path.join(self._data_path,'ImageSets','Main')
        al_imageset_file = os.path.join(al_imageset_dir,'al_train.txt')
        al_annot_file = os.path.join(self._data_path)

        if not os.path.isdir(al_imageset_dir):
            os.mkdir(al_imageset_dir)
        if not os.path.isdir(cachedir):
            os.mkdir(cachedir)
        cachefile = os.path.join(cachedir, 'Stanford_Dog_ws_train_annots.pkl')
        # read list of images
        with open(imagesetfile, 'r') as f:
            lines = f.readlines()
        imagenames = [x.strip() for x in lines]

        if not os.path.isfile(cachefile): # loads and save annotations
            recs = {}
            sizes = {}
            for i, imagename in enumerate(imagenames):
                recs[imagename] = self.parse_ws(annopath.format(imagename))
            if i % 100 == 0:
                print('Reading annotation for {:d}/{:d}'.format(
                i + 1, len(imagenames)))
            print('Saving cached annotations to {:s}'.format(cachefile))
            with open(cachefile, 'wb') as f:
                pickle.dump(recs, f)
        else: # load
            with open(cachefile, 'rb') as f:
                try:
                    recs = pickle.load(f)
                except:
                    recs = pickle.load(f, encoding='bytes')
        # recs need to be accessed by imagename(file name)
        # recs[ith imagename] = ith image's object list
        # recs == gt_boxes list

        uniform = torch.distributions.uniform.Uniform(0,1)
        bloodhound_list = []
        foxhound_list = []
        for i, imagename in enumerate(imagenames):
            # pred_boxes_cls = all_boxes[:,i,:,:]
            pred_boxes_cls = np.asarray(all_boxes)[:,i]
            gt_cls = int(recs[imagename]['label']) + 1
            # Variables for checking thesis, Irrelavant to test----
            highest_cls_prob = 0
            highest_xmin = 0
            highest_xmax = 0
            highest_ymin = 0
            highest_ymax = 0
            pred_boxes = pred_boxes_cls[gt_cls].reshape(-1,5)

            #added
            highest_iou = 0
            bloodhounds = pred_boxes_cls[1].reshape(-1,5)
            keep = np.where(bloodhounds[:,-1] > thresh)
            bloodhounds = bloodhounds[keep[0]]

            foxhounds = pred_boxes_cls[2].reshape(-1,5)
            keep = np.where(foxhounds[:,-1] > thresh)
            foxhounds = foxhounds[keep[0]]

            candidate = {}
            candidate['name'] = self._ind_to_class_image[gt_cls]

            for bloodhound in bloodhounds:
                for foxhound in foxhounds:
                    bloodhound_roi = torch.from_numpy(bloodhound[:4])
                    foxhound_roi = torch.from_numpy(foxhound[:4])

                    ixmin, iymin, _, _ = torch.max(bloodhound_roi, foxhound_roi)
                    _, _, ixmax, iymax = torch.min(bloodhound_roi, foxhound_roi)
                    iw = torch.max(ixmax-ixmin, torch.Tensor([0]))
                    ih = torch.max(iymax-iymin, torch.Tensor([0]))
                    inters = iw * ih

                    uni = ((bloodhound_roi[2] - bloodhound_roi[0] + 1.) * (bloodhound_roi[3] - bloodhound_roi[1] + 1.) +
                            (foxhound_roi[2] - foxhound_roi[0] + 1.) * (foxhound_roi[3] - foxhound_roi[1] + 1.) -
                            inters)
                    mutual_iou = inters / uni

                    if mutual_iou > 0.5 and mutual_iou > highest_iou:
                        highest_iou = mutual_iou
                        if gt_cls == 1:
                            xmin, ymin, xmax, ymax = bloodhound_roi[0].cpu().numpy(), bloodhound_roi[1].cpu().numpy(), bloodhound_roi[2].cpu().numpy(), bloodhound_roi[3].cpu().numpy()
                            xmin, ymin, xmax, ymax = int(np.round(xmin)), int(np.round(ymin)), int(np.round(xmax)), int(np.round(ymax))
                            candidate['bndbox'] = [xmin, ymin, xmax, ymax]
                        elif gt_cls == 2:
                            xmin, ymin, xmax, ymax = foxhound_roi[0].cpu().numpy(), foxhound_roi[1].cpu().numpy(), foxhound_roi[2].cpu().numpy(), foxhound_roi[3].cpu().numpy()
                            xmin, ymin, xmax, ymax = int(np.round(xmin)), int(np.round(ymin)), int(np.round(xmax)), int(np.round(ymax))
                            candidate['bndbox'] = [xmin, ymin, xmax, ymax]
            if 'bndbox' in candidate:
                if gt_cls == 1:
                    bloodhound_list.append((imagename,candidate))
                elif gt_cls == 2:
                    foxhound_list.append((imagename,candidate))
            #added

            # for roi_idx in range(len(pred_boxes)):
            #     roi = torch.from_numpy(pred_boxes[roi_idx][:4])
            #     cls_prob = pred_boxes[roi_idx][-1]
            #     if highest_cls_prob < cls_prob:
            #         highest_cls_prob = cls_prob
            #         highest_xmin = int(np.round(roi[0].cpu().numpy()))
            #         highest_ymin = int(np.round(roi[1].cpu().numpy()))
            #         highest_xmax = int(np.round(roi[2].cpu().numpy()))
            #         highest_ymax = int(np.round(roi[3].cpu().numpy()))
            # # Variables for checking thesis, Irrelavant to test----
            
            # if highest_cls_prob > thresh and uniform.sample() <= prob:
            #     obj = {}
            #     obj['name'] = self._ind_to_class_image[gt_cls]
            #     obj['bndbox'] = [highest_xmin,highest_ymin,highest_xmax,highest_ymax]
            #     if gt_cls == 1:
            #         benign_list.append((imagename,obj))
            #     elif gt_cls == 2:
            #         malign_list.append((imagename,obj))
        num_b = len(bloodhound_list)
        num_f = len(foxhound_list)

        with open(al_imageset_file,'w') as f:
            for i in range(num_b):
                bloodhound_img, bloodhound_obj = bloodhound_list[i]
                f.write('{}\n'.format(bloodhound_img))
                self.write_annotation(annopath.format(bloodhound_img+"_AL"), bloodhound_img, recs[bloodhound_img], bloodhound_obj)
        with open(al_imageset_file,'a') as f:
            for i in range(num_f):
                foxhound_img, foxhound_obj = foxhound_list[i]
                f.write('{}\n'.format(foxhound_img))
                self.write_annotation(annopath.format(foxhound_img+"_AL"), foxhound_img, recs[foxhound_img], foxhound_obj)

    def evaluate_detections(self, all_boxes, all_boxes_n, output_dir, thresh):
        # all_boxes : (cls, image, boxes, 5) 5 = (x, y, x, y, cls_prob)
        # all_boxes_n:(cls, image_n, boxes, 5)
        # Call all annotations
        annopath = os.path.join(self._data_path,'Annotations','{:s}.xml')
        imagesetfile = os.path.join(self._data_path,'ImageSets','Main',self._image_set + '.txt')
        cachedir = os.path.join(self._data_path, 'annotations_cache')
        logdir = os.path.join(output_dir,'results' ,'stats')
        logfile = os.path.join(logdir,'log.txt')

        if not os.path.isdir(logdir):
            os.mkdir(logdir)
        if not os.path.isdir(cachedir):
            os.mkdir(cachedir)
        cachefile = os.path.join(cachedir, 'Stanford_Dog_test_annots.pkl')
        # read list of images
        with open(imagesetfile, 'r') as f:
            lines = f.readlines()
        imagenames = [x.strip() for x in lines]

        if not os.path.isfile(cachefile): # loads and save annotations
            recs = {}
            for i, imagename in enumerate(imagenames):
                recs[imagename] = self.parse_rec(annopath.format(imagename))
            if i % 100 == 0:
                print('Reading annotation for {:d}/{:d}'.format(
                i + 1, len(imagenames)))
            print('Saving cached annotations to {:s}'.format(cachefile))
            with open(cachefile, 'wb') as f:
                pickle.dump(recs, f)
        else: # load
            with open(cachefile, 'rb') as f:
                try:
                    recs = pickle.load(f)
                except:
                    recs = pickle.load(f, encoding='bytes')
        # recs need to be accessed by imagename(file name)
        # recs[ith imagename] = ith image's object list
        # recs == gt_boxes list
        lesion_detected = 0
        box_over_object = 0 # predicted a box on an object
        cor_box_over_object = 0# predicted correct class box on object
        box_over_background_n = 0
        
        #metric added
        # metric = sum of (mutual_iou * min(box1, box2))
        metric = 0
        num_metric = 0

        for i in all_boxes_n[1:]:
            for j in i:
                box_over_background_n += len(np.where(np.asarray(j)[:,-1] > thresh)[0])
        # box_over_background_n = len(torch.FloatTensor(all_boxes_n).view(-1,5))# any predicted boxes in normal data is wrong
        all_object = len(recs) # assume an image has only one gt object
        all_detected_boxes = 0
        for i in all_boxes[1:]:
            for j in i:
                all_detected_boxes += len(np.where(np.asarray(j)[:,-1] > thresh)[0])# npwhere returns a tuple thus add [0]
        # all_detected_boxes += box_over_background_n
        # all_detected_boxes = len(torch.FloatTensor(all_boxes).view(-1,5)) + box_over_background_n

        for i, imagename in enumerate(imagenames):
            # pred_boxes_cls = all_boxes[:,i,:,:]
            # pred_boxes_cls[cls]: cls probs of all_boxes
            # ** there can be duplicated boxes over classes. ex) pred_boxes_cls[1][i][:4] == pred_boxes_cls[2][j][:4]            
            pred_boxes_cls = np.asarray(all_boxes)[:,i]
            gt_box = torch.FloatTensor(recs[imagename][0]['bbox']) # assumes only one object exists as gt
            gt_cls = self._class_to_ind_image[recs[imagename][0]['name']]
            # Variables for checking thesis, Irrelavant to test----
            image_detected = False
            highest_cls_prob = 0

            #metric added
            bloodhounds = pred_boxes_cls[1].reshape(-1,5)
            keep = np.where(bloodhounds[:,-1] > thresh)
            bloodhounds = bloodhounds[keep[0]]

            foxhounds = pred_boxes_cls[2].reshape(-1,5)
            keep = np.where(foxhounds[:,-1] > thresh)
            foxhounds = foxhounds[keep[0]]
            for bloodhound in bloodhounds:
                for foxhound in foxhounds:
                    bloodhound_roi = torch.from_numpy(bloodhound[:4])
                    foxhound_roi = torch.from_numpy(foxhound[:4])
                    min_score = bloodhound[-1] if bloodhound[-1] <= foxhound[-1] else foxhound[-1]

                    ixmin, iymin, _, _ = torch.max(bloodhound_roi, foxhound_roi)
                    _, _, ixmax, iymax = torch.min(bloodhound_roi, foxhound_roi)
                    iw = torch.max(ixmax-ixmin, torch.Tensor([0]))
                    ih = torch.max(iymax-iymin, torch.Tensor([0]))
                    inters = iw * ih

                    uni = ((bloodhound_roi[2] - bloodhound_roi[0] + 1.) * (bloodhound_roi[3] - bloodhound_roi[1] + 1.) +
                            (foxhound_roi[2] - foxhound_roi[0] + 1.) * (foxhound_roi[3] - foxhound_roi[1] + 1.) -
                            inters)
                    mutual_iou = inters / uni

                    if mutual_iou > 0.5:
                        metric += mutual_iou.item() * min_score.item()
                        num_metric += 1
            #metric added

            for cls_idx in range(1,self.num_classes):
                pred_boxes = pred_boxes_cls[cls_idx].reshape(-1,5)
                # pdb.set_trace()
                # print(pred_boxes.shape)
                # print(pred_boxes)
                for roi_idx in range(len(pred_boxes)):
                    roi = torch.from_numpy(pred_boxes[roi_idx][:4])
                    cls_prob = pred_boxes[roi_idx][-1]
                    ixmin, iymin, _, _ = torch.max(roi, gt_box)
                    _, _, ixmax, iymax = torch.min(roi, gt_box)
                    iw = torch.max(ixmax-ixmin, torch.Tensor([0]))
                    ih = torch.max(iymax-iymin, torch.Tensor([0]))
                    inters = iw*ih

                    uni = ((roi[2] - roi[0] + 1.) * (roi[3] - roi[1] + 1.) +
                            (gt_box[2] - gt_box[0] + 1.) * (gt_box[3] - gt_box[1] + 1.) - 
                            inters)
                    iou = inters/uni
                    if iou > 0.5 and cls_prob > thresh:# Threshold for detection
                        image_detected = True
                        box_over_object += 1
                        if cls_idx == gt_cls and cls_prob>0.5:
                            cor_box_over_object += 1
                    
                    if cls_idx == gt_cls and highest_cls_prob < cls_prob:
                        highest_cls_prob = cls_prob
                        highest_box_iou = iou
                    
            if image_detected:
                lesion_detected += 1
        # Finished counting for the whole dataset
        box_over_background = all_detected_boxes - box_over_object
        with open(logfile, 'a') as f:
            f.write('{},{},{},{},{},{}, {}, {:.4f}\n'.format(box_over_object, lesion_detected, all_object, cor_box_over_object, all_detected_boxes, box_over_background_n, num_metric, metric))

    def parse_ws(self, filename):
        """Parse BIRADS from xml file """
        tree = ET.parse(filename)
        obj = tree.find('object')
        size = tree.find('size')

        recs = {}
        recs['label'] = self._class_to_ind_image[obj.find('name').text] - 1
        recs['width'] = size.find('width').text
        recs['height'] = size.find('height').text
        recs['depth'] = size.find('depth').text

        return recs

    def parse_rec(self, filename):
        """ Parse a PASCAL VOC xml file """
        tree = ET.parse(filename)
        objects = []
        for obj in tree.findall('object'):
            obj_struct = {}
            obj_struct['name'] = obj.find('name').text
            if obj_struct['name'] == '__background__':
                continue
            bbox = obj.find('bndbox')
            obj_struct['bbox'] = [int(bbox.find('xmin').text),
                                int(bbox.find('ymin').text),
                                int(bbox.find('xmax').text),
                                int(bbox.find('ymax').text)]
            objects.append(obj_struct)

        return objects # list of dictionaries

    def write_annotation(self, file_path, filename, recs, obj):
        with open(file_path,'w') as f:
            root = Element('annotation')
            SubElement(root, 'folder').text = '02088466' if filename.startswith("n02088466") else '02089973'
            SubElement(root, 'filename').text = filename
            
            source = SubElement(root, 'source')
            SubElement(source, 'database').text = 'ImageNet database'

            size = SubElement(root, 'size')
            SubElement(size, 'width').text = str(recs['width'])
            SubElement(size, 'height').text = str(recs['height'])
            SubElement(size, 'depth').text = str(recs['depth'])

            SubElement(root, 'segment').text = '0'
            
            objects = SubElement(root, 'object')
            SubElement(objects, 'name').text = obj['name']
            SubElement(objects, 'pose').text = 'Unspecified'
            SubElement(objects, 'truncated').text = '0'
            SubElement(objects, 'difficult').text = '0'
            
            bbox = SubElement(objects, 'bndbox')
            SubElement(bbox, 'xmin').text = str(obj['bndbox'][0])
            SubElement(bbox, 'ymin').text = str(obj['bndbox'][1])
            SubElement(bbox, 'xmax').text = str(obj['bndbox'][2])
            SubElement(bbox, 'ymax').text = str(obj['bndbox'][3])

            def indent(elements, level=0):
                i = "\n" + level * " "
                if len(elements):
                    if not elements.text or not elements.text.strip():
                        elements.text = i + " "
                    if not elements.tail or not elements.tail.strip():
                        elements.tail = i
                    for elements in elements:
                        indent(elements, level+1)
                    if not elements.tail or not elements.tail.strip():
                        elements.tail = i
                else:
                    if level and (not elements.tail or not elements.tail.strip()):
                        elements.tail = i
            indent(root)
            tree = ElementTree(root)
            tree.write(file_path)