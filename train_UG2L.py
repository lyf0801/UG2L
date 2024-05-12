#!/usr/bin/python3
#coding=utf-8
from misc import AvgMeter, check_mkdir
import sys
import datetime
sys.path.insert(0, '../')
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch import optim
from torch.optim.lr_scheduler import _LRScheduler
from timm.scheduler import create_scheduler
from timm.optim import create_optimizer
from src.UG2L import net as Net
from ORSI_SOD_dataset import ORSI_SOD_Dataset
from parser import parser_args
from evaluator import *

args = {
    'iter_num': 7500,
    'epoch': 300,
    'train_batch_size': 8,
    'last_iter': 0,
    'snapshot': ''
}

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

torch.manual_seed(2022)


"""
  smaps : BCE + wIOU
  edges: BCE
"""
def structure_loss(pred, mask):
    #mask = mask.detach()
    wbce  = F.binary_cross_entropy_with_logits(pred, mask)
    pred  = torch.sigmoid(pred)
    inter = (pred*mask).sum(dim=(2,3))
    union = (pred+mask).sum(dim=(2,3))
    wiou  = 1-(inter+1)/(union-inter+1)
    return wbce.mean()+wiou.mean()#
"""
define calculate uncertainty loss
"""
def cal_usl(seg_logits, seg_gts):
    assert seg_logits.shape == seg_gts.shape, (seg_logits.shape, seg_gts.shape)
    pred = seg_logits.sigmoid()
    #np.log(2 - abs((2*i-1.0))) / np.log(2)
    ucl = torch.log(2 - (2 * pred - 1).abs())
    return ucl.mean() / torch.log(torch.Tensor([2])).cuda()
"""
perform cosine strategy
"""
def get_coef(iter_percentage, method='cos'):
    coef_range = (0, 1)
    min_coef, max_coef = min(coef_range), max(coef_range)
    normalized_coef = 1 + np.cos(iter_percentage * np.pi)
    ual_coef = 1/2 * normalized_coef * (max_coef - min_coef) + min_coef

    return ual_coef


def main(dataset):
    #define dataset
    data_path = '/data/iopen/lyf/SaliencyOD_in_RSIs/'+ dataset +' dataset/'
    train_dataset = ORSI_SOD_Dataset(root = data_path,  mode = 'train', aug = True)
    train_loader = DataLoader(train_dataset, batch_size = args['train_batch_size'], shuffle = True, num_workers = 8)
    test_set = ORSI_SOD_Dataset(root = data_path,  mode = "test", aug = False)
    test_loader = DataLoader(test_set, batch_size = 1, shuffle = True, num_workers = 1)
    args['iter_num'] = args['epoch'] * len(train_loader)

    #############################ResNet pretrained###########################
    #res18[2,2,2,2],res34[3,4,6,3],res50[3,4,6,3],res101[3,4,23,3],res152[3,8,36,3]
    model = Net()
    ##############################Optim setting###############################
    net = model.cuda().train()
    optimizer = create_optimizer(parser_args, net)
    lr_scheduler, _ = create_scheduler(parser_args, optimizer)
    #########################################################################

    curr_iter = args['last_iter']
    for epoch in range(0, args['epoch']): 
        loss_sod_record, loss_ucl_record, total_loss_record = AvgMeter(),  AvgMeter(),  AvgMeter()
        net.train() 
        for i, data in enumerate(train_loader):
            
            # data\binarizing\Variable
            inputs, labels, _, _ = data
            batch_size = inputs.size(0)

            inputs = inputs.type(torch.FloatTensor).cuda()
            labels = labels.type(torch.FloatTensor).cuda()

            optimizer.zero_grad()
            smap1, smap2, smap3, smap4, smap5 = net(inputs)
            ##########loss#############

            loss1_1 = structure_loss(smap1, labels)
            loss1_2 = structure_loss(smap2, labels)
            loss1_3 = structure_loss(smap3, labels)
            loss1_4 = structure_loss(smap4, labels)
            loss1_5 = structure_loss(smap5, labels)


            loss_sod = loss1_1 + loss1_2 + (loss1_3 / 2) + (loss1_4 / 4) + (loss1_5 / 8)   
            loss_ucl = cal_usl(smap1, labels) + cal_usl(smap2, labels) + (cal_usl(smap3, labels) / 2) + (cal_usl(smap4, labels)/ 4) + (cal_usl(smap5, labels) / 8)
            loss_ucl = get_coef(iter_percentage= curr_iter/args['iter_num']) * loss_ucl
            total_loss = loss_sod + loss_ucl
            total_loss.backward()
            
            optimizer.step()
            lr_scheduler.step(epoch)

            loss_sod_record.update(loss_sod.item(), batch_size)
            loss_ucl_record.update(loss_ucl.item(), batch_size)
            total_loss_record.update(total_loss.item(), batch_size)

            #############log###############
            curr_iter += 1

            log = '[epoch: %03d] [iter: %05d] [total loss %.5f]  [sod loss %.5f]  [ucl loss %.5f]  [lr %.13f] ' % \
                  (epoch, curr_iter, total_loss_record.avg,  loss_sod_record.avg, loss_ucl_record.avg,  optimizer.param_groups[0]['lr'])
            print(log)
            


if __name__ == '__main__':
    dataset = "ORS_4199"
    main(dataset)

