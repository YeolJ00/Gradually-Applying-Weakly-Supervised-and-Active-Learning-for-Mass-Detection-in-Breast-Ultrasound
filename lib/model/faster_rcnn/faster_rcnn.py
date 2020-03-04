import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.models as models
from torch.autograd import Variable
import numpy as np
from model.utils.config import cfg
from model.rpn.rpn import _RPN

from model.roi_layers import ROIAlign, ROIPool

# from model.roi_pooling.modules.roi_pool import _RoIPooling
# from model.roi_align.modules.roi_align import RoIAlignAvg

from model.rpn.proposal_target_layer_cascade import _ProposalTargetLayer
import time
import pdb
from model.utils.net_utils import _smooth_l1_loss, _crop_pool_layer, _affine_grid_gen, _affine_theta

class _fasterRCNN(nn.Module):
  """ faster RCNN """
  def __init__(self, classes, class_agnostic):
    super(_fasterRCNN, self).__init__()
    self.classes = classes
    self.n_classes = len(classes)
    self.class_agnostic = class_agnostic
    # loss
    self.RCNN_loss_cls = 0
    self.RCNN_loss_bbox = 0

    # define rpn
    self.RCNN_rpn = _RPN(self.dout_base_model)
    self.RCNN_proposal_target = _ProposalTargetLayer(self.n_classes)

    # self.RCNN_roi_pool = _RoIPooling(cfg.POOLING_SIZE, cfg.POOLING_SIZE, 1.0/16.0)
    # self.RCNN_roi_align = RoIAlignAvg(cfg.POOLING_SIZE, cfg.POOLING_SIZE, 1.0/16.0)

    self.RCNN_roi_pool = ROIPool((cfg.POOLING_SIZE, cfg.POOLING_SIZE), 1.0/16.0)
    self.RCNN_roi_align = ROIAlign((cfg.POOLING_SIZE, cfg.POOLING_SIZE), 1.0/16.0, 0)

  def forward(self, im_data, im_info, gt_boxes, num_boxes, im_label, is_ws = False):
    batch_size = im_data.size(0)

    im_info = im_info.data
    gt_boxes = gt_boxes.data
    num_boxes = num_boxes.data
    im_label = im_label.data
    #print(im_label)

    # feed image data to base model to obtain base feature map
    # (batch, 1024, H, W)
    base_feat = self.RCNN_base(im_data)

    # feed base feature map to RPN to obtain rois
    # if is_ws == True, rpn_loss_cls is ACTUALLY rpn_cls_prob.data
    rois, rpn_loss_cls, rpn_loss_bbox = self.RCNN_rpn(base_feat, im_info, gt_boxes, num_boxes, is_ws)

    # torch.set_printoptions(threshold=5000)
    # pdb.set_trace()
    # print(rois[0,:25])
    # print(gt_boxes)
    # print(gt_boxes.shape)

    # if it is training phase, then use ground truth bboxes for refining
    if self.training and is_ws == False:
      roi_data = self.RCNN_proposal_target(rois, gt_boxes, num_boxes)
      rois, rois_label, rois_target, rois_inside_ws, rois_outside_ws, fg_rois_per_this_image = roi_data
      # rois : (batch, rois per image, 5)
      # rois_label: (batch, rois_per_image)
      # rois_target: (batch, anchor, 4)
      # rois_inside_ws:(batch, anchor, 4)
      # rois_outside_ws:(batch, anchor, 4)
      # fg_rois_per_this_image : int

      # pdb.set_trace()
      # print(rois)
      # print(rois_label)

      rois_label = Variable(rois_label.view(-1).long())
      rois_target = Variable(rois_target.view(-1, rois_target.size(2)))
      rois_inside_ws = Variable(rois_inside_ws.view(-1, rois_inside_ws.size(2)))
      rois_outside_ws = Variable(rois_outside_ws.view(-1, rois_outside_ws.size(2)))
    else:

      rois_label = None
      rois_target = None
      rois_inside_ws = None
      rois_outside_ws = None
      rpn_loss_cls = 0
      rpn_loss_bbox = 0

    rois = Variable(rois)


    # torch.set_printoptions(threshold=5000)
    # pdb.set_trace()
    # print(gt_boxes)
    # print(rois)


    # do roi pooling based on predicted rois
    # base_feat : (batch, 1024, h, w)
    # rois.view(-1,5) : (batch*rois, 5)
    # pooled_feat: (batch*rois, 1024, h, w)

    if cfg.POOLING_MODE == 'align':
      pooled_feat = self.RCNN_roi_align(base_feat, rois.view(-1, 5))
    elif cfg.POOLING_MODE == 'pool':
      pooled_feat = self.RCNN_roi_pool(base_feat, rois.view(-1,5))

    # feed pooled features to top model
    # head to tail is layer4 in resnet, which is a residual block.
    # (batch*rois, 1024, h, w) -> (batch*rois, 2048, h/2, w/2) -> (batch*rois, 2048)
    pooled_feat = self._head_to_tail(pooled_feat)

    # compute bbox offset
    bbox_pred = self.RCNN_bbox_pred(pooled_feat)
    # bbox_pred : (batch*rois, 4*C)
    if self.training and not self.class_agnostic and is_ws == False:
      # select the corresponding columns according to roi labels
      bbox_pred_view = bbox_pred.view(bbox_pred.size(0), int(bbox_pred.size(1) / 4), 4) # (batch*rois, C, 4)
      bbox_pred_select = torch.gather(bbox_pred_view, 1, rois_label.view(rois_label.size(0), 1, 1).expand(rois_label.size(0), 1, 4))
      bbox_pred = bbox_pred_select.squeeze(1)
    # bbox_pred : (batch*rois, 4)
    # compute object classification probability
    # cls_score : (rois, num classes)
    # cls_prob : (rois, num_classes)
    cls_score = self.RCNN_cls_score(pooled_feat)
    cls_prob = F.softmax(cls_score, 1)

    RCNN_loss_cls = 0
    RCNN_loss_bbox = 0
    '''
    if is_ws == True
    1. choose one roi according to cls_prob
    2. assign rois_label
    '''
    if self.training and is_ws:
      # classification loss
      # cls_prob_ws = cls_prob[:,(im_label+1).long()].squeeze() # (rois,)
      cls_prob_ws = cls_prob[:,2].squeeze() # (rois,)
      cls_prob_bg = cls_prob[:,0].squeeze()
      # pdb.set_trace()
      # print(cls_prob)
      # print(cls_prob_ws.shape)
      # print(cls_prob_ws)
      chosen_roi = torch.argmax(cls_prob_ws, dim = 0)
      rois_label = torch.FloatTensor([im_label+1.0]) # (rois,)
      _cls_score = cls_score[chosen_roi].unsqueeze_(0) # (1,3)

      # negative sample with least malignant
      chosen_bg = torch.argmin(cls_prob_ws, dim = 0)
      bg_score = cls_score[chosen_bg].unsqueeze_(0)
      cls_score = torch.cat((_cls_score,bg_score), dim = 0)
      rois_label = torch.FloatTensor([im_label+1.0, 0])

      rois_label = Variable(rois_label.view(-1).long().cuda())
      # pdb.set_trace()
      # print(rois_label.shape)
      # print(cls_score.shape)
      class_weight = torch.FloatTensor([1 , 1./(1-cfg.TRAIN.WS_MAL_PCT), 1./cfg.TRAIN.WS_MAL_PCT]).cuda()
      class_weight = Variable(class_weight, requires_grad = False)
      RCNN_loss_cls = F.cross_entropy(cls_score, rois_label, class_weight)
    if self.training and is_ws == False:
      # bounding box regression L1 loss
      # fg = max(1, fg_rois_per_this_image)
      # bg = max(1, cfg.TRAIN.BATCH_SIZE - fg_rois_per_this_image)
      # class_weight = torch.FloatTensor([1, 0.5*bg/fg, 0.5*bg/fg]).cuda()
      # class_weight = Variable(class_weight, requires_grad = False)
      # RCNN_loss_cls = F.cross_entropy(cls_score, rois_label, class_weight)
      RCNN_loss_cls = F.cross_entropy(cls_score, rois_label)
      RCNN_loss_bbox = _smooth_l1_loss(bbox_pred, rois_target, rois_inside_ws, rois_outside_ws)


    cls_prob = cls_prob.view(batch_size, rois.size(1), -1)
    bbox_pred = bbox_pred.view(batch_size, rois.size(1), -1)

    return rois, cls_prob, bbox_pred, rpn_loss_cls, rpn_loss_bbox, RCNN_loss_cls, RCNN_loss_bbox, rois_label

  def _init_weights(self):
    def normal_init(m, mean, stddev, truncated=False):
      """
      weight initalizer: truncated normal and random normal.
      """
      # x is a parameter
      if truncated:
        m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean) # not a perfect approximation
      else:
        torch.nn.init.xavier_normal_(m.weight)
        # m.weight.data.normal_(mean, stddev)
        # m.bias.data.zero_()
    
      
    def weight_Sequential(m):
      if type(m) == nn.Linear:
        m.weight.data.normal_(0, 0.001)
    

    normal_init(self.RCNN_rpn.RPN_Conv, 0, 0.01, cfg.TRAIN.TRUNCATED)
    normal_init(self.RCNN_rpn.RPN_cls_score, 0, 0.01, cfg.TRAIN.TRUNCATED)
    normal_init(self.RCNN_rpn.RPN_bbox_pred, 0, 0.01, cfg.TRAIN.TRUNCATED)
    normal_init(self.RCNN_cls_score, 0, 0.01, cfg.TRAIN.TRUNCATED)
    normal_init(self.RCNN_bbox_pred, 0, 0.001, cfg.TRAIN.TRUNCATED)
    # self.RCNN_top.apply(weight_Sequential)

  def create_architecture(self):
    self._init_modules()
    self._init_weights()
