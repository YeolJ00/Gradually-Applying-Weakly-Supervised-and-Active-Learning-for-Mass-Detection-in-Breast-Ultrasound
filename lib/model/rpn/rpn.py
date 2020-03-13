from __future__ import absolute_import
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from model.utils.config import cfg
from .proposal_layer import _ProposalLayer
from .anchor_target_layer import _AnchorTargetLayer
from model.utils.net_utils import _smooth_l1_loss_3d
from .focal_loss import FocalLoss2d

import numpy as np
import math
import pdb
import time

class _RPN(nn.Module):
    """ region proposal network """
    def __init__(self, din):
        super(_RPN, self).__init__()
        
        self.din = din  # get depth of input feature map, e.g., 512
        self.anchor_scales = cfg.ANCHOR_SCALES
        self.anchor_ratios = cfg.ANCHOR_RATIOS
        self.feat_stride = cfg.FEAT_STRIDE[0]

        # define the convrelu layers processing input feature map
        # self.RPN_Conv = nn.Conv2d(self.din, 1024, kernel_size = 3, stride = 1, padding = 1, bias=True)
        self.RPN_Conv = nn.Conv2d(self.din, 512, kernel_size = 3, stride = 1, padding = 1, bias=True)

        # define bg/fg classifcation score layer
        self.nc_score_out = len(self.anchor_scales) * len(self.anchor_ratios) * 2 # 2(bg/fg) * 9 (anchors)
        # self.RPN_cls_score = nn.Conv2d(1024, self.nc_score_out, kernel_size = 3, stride = 1, padding =  1)
        self.RPN_cls_score = nn.Conv2d(512, self.nc_score_out, kernel_size = 1, stride = 1, padding =  1)

        # define anchor box offset prediction layer
        self.nc_bbox_out = len(self.anchor_scales) * len(self.anchor_ratios) * 4 # 4(coords) * 9 (anchors)
        # self.RPN_bbox_pred = nn.Conv2d(1024, self.nc_bbox_out, kernel_size = 3, stride =  1, padding = 1)
        self.RPN_bbox_pred = nn.Conv2d(512, self.nc_bbox_out, kernel_size = 1, stride =  1, padding = 1)

        # define proposal layer
        self.RPN_proposal = _ProposalLayer(self.feat_stride, self.anchor_scales, self.anchor_ratios)

        # define anchor target layer
        self.RPN_anchor_target = _AnchorTargetLayer(self.feat_stride, self.anchor_scales, self.anchor_ratios)

        # define Focal Loss
        self.FocalLoss2d = FocalLoss2d(gamma = 2)

        self.rpn_loss_cls = 0
        self.rpn_loss_box = 0

    @staticmethod
    def reshape(x, d):
        input_shape = x.size()
        x = x.view(
            input_shape[0],
            int(d),
            int(float(input_shape[1] * input_shape[2]) / float(d)),
            input_shape[3]
        )
        return x

    def forward(self, base_feat, im_info, gt_boxes, num_boxes, is_ws):

        batch_size = base_feat.size(0)

        # return feature map after convrelu layer
        rpn_conv1 = F.relu(self.RPN_Conv(base_feat), inplace=True)
        # get rpn classification score
        rpn_cls_score = self.RPN_cls_score(rpn_conv1)
        # rpn_cls_score size is (batch, anchor*2, H, W)

        rpn_cls_score_reshape = self.reshape(rpn_cls_score, 2) # reshaped size is (batch, 2, anchor*H ,W)
        rpn_cls_prob_reshape = F.softmax(rpn_cls_score_reshape, 1)  # softamx only rescales two numbers
        rpn_cls_prob = self.reshape(rpn_cls_prob_reshape, self.nc_score_out) # (batch, anchor*2, H , W)

        # get rpn offsets to the anchor boxes + nms
        rpn_bbox_pred = self.RPN_bbox_pred(rpn_conv1)

        # proposal layer
        cfg_key = 'TRAIN' if self.training else 'TEST'
        rois = self.RPN_proposal((rpn_cls_prob.data, rpn_bbox_pred.data,
                                 im_info, cfg_key))
        # rois : (batch, 300, 5)

        # torch.set_printoptions(threshold=100000)
        # pdb.set_trace()
        # print(rpn_cls_prob_reshape[0,1,:,:])

        self.rpn_loss_cls = 0
        self.rpn_loss_box = 0
        if self.training and is_ws:
            return rois, rpn_cls_prob.data, self.rpn_loss_box
        # generating training labels and build the rpn loss
        if self.training and not is_ws:
            assert gt_boxes is not None
            rpn_data = self.RPN_anchor_target((rpn_cls_score.data, gt_boxes, im_info, num_boxes))
            
            # output [labels, bbox_targets, bbox_inside_weigts, bbox_outside_weights]
            # labels : (batch_size, 1, A*H, W)
            # bbox_targets: (batch_size, A*4, H, W)
            # bbox_inside_weights: (batch_size, A*4, H, W)
            # bbox_outside_weights: (batch_size, A*4, H, W)
            
            # compute classification loss
            rpn_cls_score = rpn_cls_score_reshape.permute(0, 2, 3, 1).contiguous().view(batch_size, -1, 2)
            rpn_label = rpn_data[0].view(batch_size, -1)#  (batch, 9*H*W)
            _rpn_label = rpn_label.view(batch_size, 9, -1)#(batch, 9, H*W)
            # pdb.set_trace()
            # print(_rpn_label.shape)
            # print(_rpn_label[0,:,2])
            # print(_rpn_label[0,:,3])

            rpn_keep = Variable(rpn_label.view(-1).ne(-1).nonzero().view(-1))
            rpn_cls_score = torch.index_select(rpn_cls_score.view(-1,2), 0, rpn_keep)
            rpn_label = torch.index_select(rpn_label.view(-1), 0, rpn_keep.data)
            rpn_label = Variable(rpn_label.long())
            # self.rpn_loss_cls = self.FocalLoss2d(rpn_cls_score, rpn_label)
            self.rpn_loss_cls = F.cross_entropy(rpn_cls_score, rpn_label)
            fg_cnt = torch.sum(rpn_label.data.ne(0)) # useless

            rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights = rpn_data[1:]

            # compute bbox regression loss
            rpn_bbox_inside_weights = Variable(rpn_bbox_inside_weights)
            rpn_bbox_outside_weights = Variable(rpn_bbox_outside_weights)
            rpn_bbox_targets = Variable(rpn_bbox_targets)

            # self.rpn_loss_box = _smooth_l1_loss(rpn_bbox_pred, rpn_bbox_targets, rpn_bbox_inside_weights,
            #                                                 rpn_bbox_outside_weights, sigma=3, dim=[1,2,3])#(1, 9, H, W)
            
            _rpn_loss_box = _smooth_l1_loss_3d(rpn_bbox_pred, rpn_bbox_targets, rpn_bbox_inside_weights,
                                                            rpn_bbox_outside_weights, sigma=3)
            _rpn_loss_box = _rpn_loss_box.view(batch_size, 9, -1)
            _rpn_loss_box = torch.where(_rpn_label == 1, _rpn_loss_box, torch.FloatTensor([0]).cuda()) #(1, 9, H*W)
            self.rpn_loss_box = _rpn_loss_box.sum(2).mean()

        return rois, self.rpn_loss_cls, self.rpn_loss_box
        # rois : proposals sent to faster_rcnn (batch, nms_top_n, 5) 5 is (batch#,x,y,x,y)
        # rpn_loss_cls : sum loss, type = float
        # rpn_loss_box : mean loss, type = float
