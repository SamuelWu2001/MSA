# train.py
#!/usr/bin/env	python3

""" train network using pytorch
    Junde Wu
"""

import os
import sys
import argparse
from datetime import datetime
from collections import OrderedDict
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score, accuracy_score,confusion_matrix
import torchvision
import torchvision.transforms as transforms
from skimage import io
from torch.utils.data import DataLoader
#from dataset import *
from torch.autograd import Variable
from PIL import Image
from tensorboardX import SummaryWriter
#from models.discriminatorlayer import discriminator
from dataset import *
from conf import settings
import time
import cfg
from tqdm import tqdm
from torch.utils.data import DataLoader, random_split
from utils import *
import function 


args = cfg.parse_args()

GPUdevice = torch.device('cuda', args.gpu_device)

net = get_network(args, args.net, use_gpu=args.gpu, gpu_device=GPUdevice, distribution = args.distributed)
if args.pretrain:
    weights = torch.load(args.pretrain)
    net.load_state_dict(weights,strict=False)

freeze_model(net)
# check_freeze_status(net)

optimizer = optim.Adam(net.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5) #learning rate decay

'''load pretrained model'''
if args.weights != 0:
    print(f'=> resuming from {args.weights}')
    assert os.path.exists(args.weights)
    checkpoint_file = os.path.join(args.weights)
    assert os.path.exists(checkpoint_file)
    loc = 'cuda:{}'.format(args.gpu_device)
    checkpoint = torch.load(checkpoint_file, map_location=loc)
    start_epoch = checkpoint['epoch']
    best_tol = checkpoint['best_tol']
    
    net.load_state_dict(checkpoint['state_dict'],strict=False)
    # optimizer.load_state_dict(checkpoint['optimizer'], strict=False)

    args.path_helper = checkpoint['path_helper']
    logger = create_logger(args.path_helper['log_path'])
    print(f'=> loaded checkpoint {checkpoint_file} (epoch {start_epoch})')

args.path_helper = set_log_dir('logs', args.exp_name)
logger = create_logger(args.path_helper['log_path'])
logger.info(args)


'''segmentation data'''
transform_train = transforms.Compose([
    transforms.Resize((args.image_size,args.image_size), antialias=None),
    transforms.ToTensor(),
])

transform_train_seg = transforms.Compose([
    transforms.Resize((args.out_size,args.out_size), antialias=None),
    transforms.ToTensor(),
])

transform_test = transforms.Compose([
    transforms.Resize((args.image_size, args.image_size), antialias=None),
    transforms.ToTensor(),
])

transform_test_seg = transforms.Compose([
    transforms.Resize((args.out_size,args.out_size), antialias=None),
    transforms.ToTensor(),
])


if args.dataset == 'isic':
    '''isic data'''
    isic_train_dataset = ISIC2016(args, args.data_path, transform = transform_train, transform_msk= transform_train_seg, mode = 'Training')
    isic_test_dataset = ISIC2016(args, args.data_path, transform = transform_test, transform_msk= transform_test_seg, mode = 'Test')

    nice_train_loader = DataLoader(isic_train_dataset, batch_size=args.b, shuffle=True, num_workers=8, pin_memory=True)
    nice_test_loader = DataLoader(isic_test_dataset, batch_size=args.b, shuffle=False, num_workers=8, pin_memory=True)
    '''end'''

elif args.dataset == 'decathlon':
    nice_train_loader, nice_test_loader, train_eval_loader, val_eval_loader, transform_train, transform_val, train_list, val_list =get_decath_loader(args)

'''checkpoint path and tensorboard'''
# iter_per_epoch = len(Glaucoma_training_loader)
checkpoint_path = os.path.join(settings.CHECKPOINT_PATH, args.net, settings.TIME_NOW)
#use tensorboard
if not os.path.exists(settings.LOG_DIR):
    os.mkdir(settings.LOG_DIR)
writer = SummaryWriter(log_dir=os.path.join(
        settings.LOG_DIR, args.net, settings.TIME_NOW))
# input_tensor = torch.Tensor(args.b, 3, 256, 256).cuda(device = GPUdevice)
# writer.add_graph(net, Variable(input_tensor, requires_grad=True))

#create checkpoint folder to save model
if not os.path.exists(checkpoint_path):
    os.makedirs(checkpoint_path)
checkpoint_path = os.path.join(checkpoint_path, '{net}-{epoch}-{type}.pth')

'''begain training'''
best_acc = 0.0
best_tol = 1e4
# for epoch in range(settings.EPOCH):
depth_list = []
for batch_idx, pack in enumerate(train_eval_loader):
    imgs = pack['image']
    depth_list.append(imgs.shape[4])
for batch_idx, pack in enumerate(val_eval_loader):
    imgs = pack['image']
    depth_list.append(imgs.shape[4])
import matplotlib.pyplot as plt

count_less_than_64 = sum(1 for elem in depth_list if elem < 64)
print('count', count_less_than_64, len(depth_list))



# 计算每个元素的数量
unique_elements = list(set(depth_list))
counts = [depth_list.count(elem) for elem in unique_elements]

# 创建折线图
plt.bar(unique_elements, counts)

# 设置 x 轴和 y 轴标签
plt.xlabel('Element Values')
plt.ylabel('Count')

# 显示图形
plt.show()

#     if args.mod == 'sam_adpt':
#         net.train()
#         time_start = time.time()
#         loss = function.train_sam(args, net, optimizer, nice_train_loader, epoch, writer, vis = args.vis)
#         logger.info(f'Train loss: {loss}|| @ epoch {epoch}.')
#         time_end = time.time()
#         print('time_for_training ', time_end - time_start)
        
#         net.eval()
#         if epoch and epoch % args.val_freq == 0 or epoch == settings.EPOCH-1:
#             tol = function.validation_sam(args, nice_test_loader, epoch, net, writer)
#             logger.info(f'Valid loss: {tol} || @ epoch {epoch}.')
#             # logger.info(f'Total score: {tol}, DICE: {edice}, IOU: {eiou}, RKID_PF: {organ_dice_list[0]} & {organ_iou_list[0]}, LIVER_PF: {organ_dice_list[1]} & {organ_iou_list[1]}, SPLEEN_PF: {organ_dice_list[2]} & {organ_iou_list[2]}, PANCREAS_PF: {organ_dice_list[3]} & {organ_iou_list[3]} || @ epoch {epoch}.')
#             if args.distributed != 'none':
#                 sd = net.module.state_dict()
#             else:
#                 sd = net.state_dict()

#             if tol < best_tol:
#                 best_tol = tol
#                 is_best = True

#                 save_checkpoint({
#                     'epoch': epoch + 1,
#                     'model': args.net,
#                     'state_dict': sd,
#                     'optimizer': optimizer.state_dict(),
#                     'best_tol': best_tol,
#                     'path_helper': args.path_helper,
#                 }, is_best, args.path_helper['ckpt_path'], filename="best_checkpoint")
#             else:
#                 is_best = False
        
#         if epoch and epoch % args.eval_freq == 0 or epoch == settings.EPOCH-1:
#             overall_dice, pos_pf, neg_pf = function.eval_sam(args, train_eval_loader, epoch, net, writer)
#             logger.info(f'Train || overall_dice : {overall_dice} pos_pf : {pos_pf} neg_pf : {neg_pf} || @ epoch {epoch}.')
#             val_overall_dice, val_pos_pf, val_neg_pf = function.eval_sam(args, val_eval_loader, epoch, net, writer)
#             logger.info(f'Valid || overall_dice : {val_overall_dice} pos_pf : {val_pos_pf} neg_pf : {neg_pf} || @ epoch {epoch}.')

# writer.close()
