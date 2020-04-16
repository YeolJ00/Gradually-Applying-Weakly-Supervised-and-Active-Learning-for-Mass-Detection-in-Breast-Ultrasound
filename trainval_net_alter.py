# --------------------------------------------------------
# Pytorch multi-GPU Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Jiasen Lu, Jianwei Yang, based on code from Ross Girshick
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
import os
import sys
import numpy as np
import argparse
import pprint
import pdb
import time

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim

import torchvision.transforms as transforms
from torch.utils.data.sampler import Sampler

from roi_data_layer.roidb import combined_roidb
from roi_data_layer.roibatchLoader import roibatchLoader
from model.utils.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from model.utils.net_utils import weights_normal_init, save_net, load_net, \
    adjust_learning_rate, save_checkpoint, clip_gradient

from model.faster_rcnn.vgg16 import vgg16
from model.faster_rcnn.resnet import resnet

def parse_args():
  """
  Parse input arguments
  """
  parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
  parser.add_argument('--dataset', dest='dataset',
            help='training dataset',
            default='SNUBH_BUS', type=str)
  parser.add_argument('--net', dest='net',
          help='vgg16, res101',
          default='res101', type=str)
  parser.add_argument('--start_epoch', dest='start_epoch',
            help='starting epoch',
            default=1, type=int)
  parser.add_argument('--epochs', dest='max_epochs',
            help='number of epochs to train',
            default=20, type=int)
  parser.add_argument('--max_iter', dest='max_iter',
            default = 80000, type=int)
  parser.add_argument('--disp_interval', dest='disp_interval',
            help='number of iterations to display',
            default=100, type=int)
  parser.add_argument('--checkpoint_interval', dest='checkpoint_interval',
            help='number of iterations to display',
            default=10000, type=int)

  parser.add_argument('--save_dir', dest='save_dir',
            help='directory to save models', default="models",
            type=str)
  parser.add_argument('--nw', dest='num_workers',
            help='number of workers to load data',
            default=0, type=int)
  parser.add_argument('--cuda', dest='cuda',
            help='whether use CUDA',
            action='store_true')
  parser.add_argument('--ls', dest='large_scale',
            help='whether use large imag scale',
            action='store_true')            
  parser.add_argument('--mGPUs', dest='mGPUs',
            help='whether use multiple GPUs',
            action='store_true')
  parser.add_argument('--bs', dest='batch_size',
            help='batch_size',
            default=1, type=int)
  parser.add_argument('--cag', dest='class_agnostic',
            help='whether to perform class_agnostic bbox regression',
            action='store_true')
  parser.add_argument('--al', dest='active_learning',
            help='whether to use active learning dataset',
            action='store_true')

# config optimization
  parser.add_argument('--o', dest='optimizer',
            help='training optimizer',
            default="adam", type=str)
  parser.add_argument('--lr', dest='lr',
            help='starting learning rate',
            default=0.001, type=float)
  parser.add_argument('--lr_decay_step', dest='lr_decay_step',
            help='step to do learning rate decay, unit is epoch',
            default=10, type=int)
  parser.add_argument('--lr_decay_gamma', dest='lr_decay_gamma',
            help='learning rate decay ratio',
            default=0.1, type=float)
  parser.add_argument('--gamma_for_alpha', dest='gamma_for_alpha',
            help='power of alpha, which is ws weigth',
            default=5, type=float)
# set training session
  parser.add_argument('--s', dest='session',
            help='training session',
            default=1, type=int)

# resume trained model
  parser.add_argument('--r', dest='resume',
            help='resume checkpoint or not',
            default=False, type=bool)
  parser.add_argument('--checksession', dest='checksession',
            help='checksession to load model',
            default=1, type=int)
  parser.add_argument('--checkepoch', dest='checkepoch',
            help='checkepoch to load model',
            default=1, type=int)
  parser.add_argument('--checkpoint', dest='checkpoint',
            help='checkpoint to load model',
            default=0, type=int)
# log and display
  parser.add_argument('--use_tfb', dest='use_tfboard',
            help='whether use tensorboard',
            action='store_true')

  args = parser.parse_args()
  return args


class sampler(Sampler):
  def __init__(self, train_size, batch_size):
    # num_per_batch == num_of_batch
    self.num_data = train_size
    self.num_per_batch = int(train_size / batch_size)
    self.batch_size = batch_size
    self.range = torch.arange(0,batch_size).view(1, batch_size).long()
    self.leftover_flag = False
    if train_size % batch_size:
      self.leftover = torch.arange(self.num_per_batch*batch_size, train_size).long()
      self.leftover_flag = True

  def __iter__(self):
    rand_num = torch.randperm(self.num_per_batch).view(-1,1) * self.batch_size
    # rand_num becomes the index of batch starting point
    self.rand_num = rand_num.expand(self.num_per_batch, self.batch_size) + self.range
    # rand_num is the index of each images; expand broadcasts a single element + range gives them order

    self.rand_num_view = self.rand_num.view(-1)

    if self.leftover_flag:
      self.rand_num_view = torch.cat((self.rand_num_view, self.leftover),0)

    return iter(self.rand_num_view)

  def __len__(self):
    return self.num_data

if __name__ == '__main__':

  args = parse_args()

  print('Called with args:')
  print(args)

  if args.dataset == "imagenet":
    args.imdb_name = "imagenet_train"
    args.imdbval_name = "imagenet_val"
    args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '30']
  elif args.dataset == "SNUBH_BUS":
    args.imdb_name = "SNUBH_BUS"
    args.imdbval_name = "SNUBH_BUS_VAL"
    args.set_cfgs = ['ANCHOR_SCALES','[8, 16, 32]','ANCHOR_RATIOS','[0.5, 1, 2]','MAX_NUM_GT_BOXES','20']

  args.cfg_file = "cfgs/{}_ls.yml".format(args.net) if args.large_scale else "cfgs/{}.yml".format(args.net)

  if args.cfg_file is not None:
    cfg_from_file(args.cfg_file)
  if args.set_cfgs is not None:
    cfg_from_list(args.set_cfgs)

  print('Using config:')
  pprint.pprint(cfg)
  np.random.seed(cfg.RNG_SEED)

  #torch.backends.cudnn.benchmark = True
  if torch.cuda.is_available() and not args.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

  # train set
  # -- Note: Use validation set and disable the flipped to enable faster loading.
  cfg.TRAIN.USE_FLIPPED = True
  cfg.USE_GPU_NMS = args.cuda

  if args.active_learning:
    imdb_al, roidb_al, ratio_list_al, ratio_index_al = combined_roidb(args.imdb_name + '_al_train')
  imdb_s, roidb_s, ratio_list_s, ratio_index_s = combined_roidb(args.imdb_name + '_s_train')
  imdb_ws, roidb_ws, ratio_list_ws, ratio_index_ws = combined_roidb(args.imdb_name + '_ws_train')
  # imdb : instance of image db = snubh_bus
  # roidb: list of dictionaries

  if args.active_learning:
    train_size_al = len(roidb_al)
  train_size_s = len(roidb_s)
  train_size_ws = len(roidb_ws)
  # train_size = number of images
  
  if args.active_learning:
    print('{:d} active learning roidb entries'.format(len(roidb_al)))
  print('{:d} strong roidb entries'.format(len(roidb_s)))
  print('{:d} weak roidb entries'.format(len(roidb_ws)))

  output_dir = args.save_dir + "/" + args.net + "/" + args.dataset
  if not os.path.exists(output_dir):
    os.makedirs(output_dir)

  if args.active_learning:
    sampler_batch_al = sampler(train_size_al, args.batch_size)
  sampler_batch_s = sampler(train_size_s, args.batch_size)
  sampler_batch_ws = sampler(train_size_ws, args.batch_size)

  dataset_s = roibatchLoader(roidb_s, ratio_list_s, ratio_index_s, args.batch_size, \
                 imdb_s.num_classes, is_ws= False, training=True)
  dataloader_s = torch.utils.data.DataLoader(dataset_s, batch_size=args.batch_size, 
                sampler=sampler_batch_s, num_workers=args.num_workers)

  if args.active_learning:
    dataset_al = roibatchLoader(roidb_al, ratio_list_al, ratio_index_al, args.batch_size, \
                  imdb_al.num_classes, is_ws= False, training=True)
    dataloader_al = torch.utils.data.DataLoader(dataset_al, batch_size=args.batch_size, 
                  sampler=sampler_batch_al, num_workers=args.num_workers)

  dataset_ws = roibatchLoader(roidb_ws, ratio_list_ws, ratio_index_ws, args.batch_size,
                imdb_ws.num_classes, is_ws = True, training=True)
  dataloader_ws = torch.utils.data.DataLoader(dataset_ws, batch_size=args.batch_size,
                sampler=sampler_batch_ws, num_workers=args.num_workers)

  # initilize the tensor holder here.
  im_data = torch.FloatTensor(1)
  im_info = torch.FloatTensor(1)
  num_boxes = torch.LongTensor(1)
  gt_boxes = torch.FloatTensor(1)
  im_label = torch.FloatTensor(1)

  # ship to cuda
  if args.cuda:
    im_data = im_data.cuda()
    im_info = im_info.cuda()
    num_boxes = num_boxes.cuda()
    gt_boxes = gt_boxes.cuda()


  # make variable
  im_data = Variable(im_data)
  im_info = Variable(im_info)
  num_boxes = Variable(num_boxes)
  gt_boxes = Variable(gt_boxes)

  if args.cuda:
    cfg.CUDA = True

  # initilize the network here.
  if args.net == 'vgg16':
    fasterRCNN = vgg16(imdb_s.classes, pretrained=True, class_agnostic=args.class_agnostic)
  elif args.net == 'res101':
    fasterRCNN = resnet(imdb_s.classes, 101, pretrained=True, class_agnostic=args.class_agnostic)
  elif args.net == 'res50':
    fasterRCNN = resnet(imdb_s.classes, 50, pretrained=True, class_agnostic=args.class_agnostic)
  elif args.net == 'res152':
    fasterRCNN = resnet(imdb_s.classes, 152, pretrained=True, class_agnostic=args.class_agnostic)
  else:
    print("network is not defined")
    pdb.set_trace()

  fasterRCNN.create_architecture()

  lr = cfg.TRAIN.LEARNING_RATE
  lr = args.lr
  #tr_momentum = cfg.TRAIN.MOMENTUM
  #tr_momentum = args.momentum

  params = []
  for key, value in dict(fasterRCNN.named_parameters()).items():
    if value.requires_grad:
      if 'bias' in key:
        params += [{'params':[value],'lr':lr*(cfg.TRAIN.DOUBLE_BIAS + 1), \
        'weight_decay': cfg.TRAIN.BIAS_DECAY and cfg.TRAIN.WEIGHT_DECAY or 0}]
    else:
      params += [{'params':[value],'lr':lr, 'weight_decay': cfg.TRAIN.WEIGHT_DECAY}]

  if args.cuda:
    fasterRCNN.cuda()
    
  if args.optimizer == "adam":
    lr = lr * 0.1
    optimizer = torch.optim.Adam(params)

  elif args.optimizer == "sgd":
    optimizer = torch.optim.SGD(params, momentum=cfg.TRAIN.MOMENTUM)

  if args.cuda:
    fasterRCNN.cuda()

  if args.resume:
    load_name = os.path.join(output_dir,
    'faster_rcnn_{}_{}_{}.pth'.format(args.checksession, args.checkepoch, args.checkpoint))
    print("loading checkpoint %s" % (load_name))
    checkpoint = torch.load(load_name)
    args.session = checkpoint['session']
    args.start_epoch = checkpoint['epoch']
    fasterRCNN.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    lr = optimizer.param_groups[0]['lr']
    if 'pooling_mode' in checkpoint.keys():
      cfg.POOLING_MODE = checkpoint['pooling_mode']
    print("loaded checkpoint %s" % (load_name))

  if args.mGPUs:
    fasterRCNN = nn.DataParallel(fasterRCNN)

  if args.use_tfboard:
    from tensorboardX import SummaryWriter
    logger = SummaryWriter("logs")

  start = time.time()
  loss_temp = 0
  dataset_cycle = "strong"
  for step in range(args.max_iter + 1):
    # setting to train mode
    fasterRCNN.train()
    # alpha = 1 - (0.99 * (0.9**(step / 2000)))
    # alpha = 0.01 + 0.99 * (step/80000.)
    alpha = 0.01 + 0.99 * ((step/args.max_iter)**args.gamma_for_alpha)

    if step % train_size_s == 0 and dataset_cycle == "strong":
      data_iter_s = iter(dataloader_s)
      dataset_cycle = "al" if args.active_learning else "strong"
    if args.active_learning and step % train_size_al == 0 and dataset_cycle == "al":
      data_iter_al = iter(dataloader_al)
      dataset_cycle = "strong"
    if step % train_size_ws == 0:
      data_iter_ws = iter(dataloader_ws)

    if dataset_cycle == "strong":
      data = next(data_iter_s)
      with torch.no_grad():
        im_data.resize_(data[0].size()).copy_(data[0])
        im_info.resize_(data[1].size()).copy_(data[1])
        gt_boxes.resize_(data[2].size()).copy_(data[2])
        num_boxes.resize_(data[3].size()).copy_(data[3])
        im_label.resize_(data[4].size()).copy_(data[4])  
    elif args.active_learning and dataset_cycle =="al":
      data = next(data_iter_al)
      with torch.no_grad():
        im_data.resize_(data[0].size()).copy_(data[0])
        im_info.resize_(data[1].size()).copy_(data[1])
        gt_boxes.resize_(data[2].size()).copy_(data[2])
        num_boxes.resize_(data[3].size()).copy_(data[3])
        im_label.resize_(data[4].size()).copy_(data[4])  
    # fasterRCNN.zero_grad()
    rois, cls_prob, bbox_pred, \
    rpn_loss_cls_s, rpn_loss_box_s, \
    RCNN_loss_cls_s, RCNN_loss_bbox_s, \
    rois_label_s = fasterRCNN(im_data, im_info, gt_boxes, num_boxes, im_label, is_ws = False)

    loss = rpn_loss_cls_s.mean() + rpn_loss_box_s.mean() \
        + RCNN_loss_cls_s.mean() + RCNN_loss_bbox_s.mean()
    
    data = next(data_iter_ws)
    with torch.no_grad():
      im_data.resize_(data[0].size()).copy_(data[0])
      im_info.resize_(data[1].size()).copy_(data[1])
      gt_boxes.resize_(data[2].size()).copy_(data[2])
      num_boxes.resize_(data[3].size()).copy_(data[3])
      im_label.resize_(data[4].size()).copy_(data[4])
    # fasterRCNN.zero_grad()
    rois, cls_prob, bbox_pred, \
    rpn_loss_cls_ws, rpn_loss_box_ws, \
    RCNN_loss_cls_ws, RCNN_loss_bbox_ws, \
    rois_label_ws = fasterRCNN(im_data, im_info, gt_boxes, num_boxes, im_label, is_ws = True)
    
    loss += alpha * RCNN_loss_cls_ws.mean()

    # backward
    optimizer.zero_grad()
    loss.backward()
    if args.net == "vgg16":
        clip_gradient(fasterRCNN, 10.)
    optimizer.step()
    loss_temp += loss

    if step % args.disp_interval == 0:
      end = time.time()
      loss_temp /= args.disp_interval
      if args.mGPUs:
        loss_rpn_cls_s = rpn_loss_cls_s.mean().item()
        loss_rpn_box_s = rpn_loss_box_s.mean().item()
        loss_rcnn_cls_s = RCNN_loss_cls_s.mean().item()
        loss_rcnn_box_s = RCNN_loss_bbox_s.mean().item()
        loss_rcnn_cls_ws = alpha * RCNN_loss_cls_ws.mean().item()
      else:
        loss_rpn_cls_s = rpn_loss_cls_s.mean().item()
        loss_rpn_box_s = rpn_loss_box_s.mean().item()
        loss_rcnn_cls_s = RCNN_loss_cls_s.mean().item()
        loss_rcnn_box_s = RCNN_loss_bbox_s.mean().item()
        loss_rcnn_cls_ws = alpha * RCNN_loss_cls_ws.mean().item()

      fg_cnt = torch.sum(rois_label_s.data.ne(0))
      bg_cnt = rois_label_s.data.numel() - fg_cnt

      print("[session %d][iter %4d/%4d] loss: %.4f, lr: %.2e" \
                              % (args.session, step, args.max_iter, loss_temp, lr))
      print("\t\t\tfg/bg=(%d/%d), time cost: %f" % (fg_cnt, bg_cnt, end-start))
      print("\t\t\trpn_cls_s: %.4f, rpn_box_s: %.4f,  rcnn_cls_s: %.4f,  rcnn_box_s: %.4f, \
        \n\t\t\trcnn_cls_ws: %.4f" \
                    % (loss_rpn_cls_s, loss_rpn_box_s, loss_rcnn_cls_s, loss_rcnn_box_s, loss_rcnn_cls_ws))
      if args.use_tfboard:
        info = {
          'loss': loss_temp,
          'loss_rpn_cls': loss_rpn_cls_s,
          'loss_rpn_box': loss_rpn_box_s,
          'loss_rcnn_cls': loss_rcnn_cls_s,
          'loss_rcnn_box': loss_rcnn_box_s,
          'loss_rcnn_cls_ws' : loss_rcnn_cls_ws
        }
        logger.add_scalars("logs_s_{}/losses".format(args.session), info, step)

      loss_temp = 0
      start = time.time()

    if step%10000==0 and step != 0:
      save_name = os.path.join(output_dir, 'faster_rcnn_{}_{}_0.pth'.format(args.session, step))
      save_checkpoint({
        'session': args.session,
        'step': step + 1,
        'model': fasterRCNN.module.state_dict() if args.mGPUs else fasterRCNN.state_dict(),
        'optimizer': optimizer.state_dict(),
        'pooling_mode': cfg.POOLING_MODE,
        'class_agnostic': args.class_agnostic,
      }, save_name)
      print('save model: {}'.format(save_name))
  
    # load all images in ws
    # go through faster_rcnn

  if args.use_tfboard:
    logger.close()
