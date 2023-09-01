
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
from conf import settings
import time
import cfg
from conf import settings
from tqdm import tqdm
from utils import *
import torch.nn.functional as F
import torch
from einops import rearrange
import pytorch_ssim
import models.sam.utils.transforms as samtrans
from sklearn.metrics import f1_score

# from lucent.modelzoo.util import get_model_layers
# from lucent.optvis import render, param, transform, objectives
# from lucent.modelzoo import inceptionv1

import shutil
import tempfile

import matplotlib.pyplot as plt
from tqdm import tqdm

from monai.losses import DiceCELoss
from monai.inferers import sliding_window_inference
from monai.transforms import (
    AsDiscrete,
)
from monai.metrics import DiceMetric


import torch
from patch import patch
import statistics

args = cfg.parse_args()

GPUdevice = torch.device('cuda', args.gpu_device)
pos_weight = torch.ones([1]).cuda(device=GPUdevice)*2
criterion_G = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
seed = torch.randint(1,11,(args.b,7))

torch.backends.cudnn.benchmark = True
loss_function = DiceCELoss(to_onehot_y=True, softmax=True)
bce_lossfunc = nn.BCELoss()
scaler = torch.cuda.amp.GradScaler()
max_iterations = settings.EPOCH
post_label = AsDiscrete(to_onehot=14)
post_pred = AsDiscrete(argmax=True, to_onehot=14)
dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
dice_val_best = 0.0
global_step_best = 0
epoch_loss_values = []
metric_values = []

def train_sam(args, net: nn.Module, optimizer, train_loader,
          epoch, writer, schedulers=None, vis = 50):
    hard = 0
    epoch_loss = 0
    ind = 0
    # train mode
    net.train()
    optimizer.zero_grad()

    GPUdevice = torch.device('cuda:' + str(args.gpu_device))
    device = GPUdevice
    accumulation_steps = 8/args.b  # 梯度累积的步数
    accumulated_gradients = 0 # 計算當前累積步數
    intersection_total = 0
    union_total = 0
    seg_dice_total = 0
    seg_iou_total = 0
    seg_dice_list = []
    seg_iou_list = []
    pos_print = True
    neg_print = True
    first_pos_index = -1
    first_neg_index = -1


    if args.thd:
        lossfunc = DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')
    else:
        lossfunc = criterion_G

    with tqdm(total=len(train_loader), desc=f'Epoch {epoch}', unit='img') as pbar:
        for batch_idx, pack in enumerate(train_loader):
            imgs = pack['image'].to(dtype = torch.float32, device = GPUdevice)
            masks = pack['label'].to(dtype = torch.float32, device = GPUdevice)
            # for k,v in pack['image_meta_dict'].items():
            #     print(k)
            if 'pt' not in pack:
                imgs, pt, masks, pos_neg = generate_click_prompt(imgs, masks)  
            else:
                pt = pack['pt']
                point_labels = pack['p_label']
            name = pack['image_meta_dict']['filename_or_obj']
            if args.thd:
                pt = rearrange(pt, 'b n d -> (b d) n')
                pos_neg = rearrange(pos_neg, 'b d -> (b d)')
                imgs = rearrange(imgs, 'b c h w d -> (b d) c h w ')
                masks = rearrange(masks, 'b c h w d -> (b d) c h w ')
                imgs = imgs.repeat(1,3,1,1)
                point_labels = torch.ones(imgs.size(0))
                # imgs = torchvision.transforms.Resize((args.image_size,args.image_size))(imgs)
                # masks = torchvision.transforms.Resize((args.out_size,args.out_size))(masks)
            showp = pt
            if torch.sum(pos_neg) != 0 and torch.sum(pos_neg) != 2*args.chunk :
                first_pos_index = torch.nonzero(pos_neg == 1)[0].item()
                first_neg_index = torch.nonzero(pos_neg == 0)[0].item()


            mask_type = torch.float32
            ind += 1
            b_size,c,w,h = imgs.size()
            longsize = w if w >=h else h

            if point_labels[0] != -1:
                # point_coords = samtrans.ResizeLongestSide(longsize).apply_coords(pt, (h, w))
                point_coords = pt
                coords_torch = torch.as_tensor(point_coords, dtype=torch.float, device=GPUdevice)
                labels_torch = torch.as_tensor(point_labels, dtype=torch.int, device=GPUdevice)
                coords_torch, labels_torch = coords_torch[None, :, :], labels_torch[None, :]
                pt = (coords_torch, labels_torch)

            '''init'''
            if hard:
                true_mask_ave = (true_mask_ave > 0.5).float()
                #true_mask_ave = cons_tensor(true_mask_ave)
            imgs = imgs.to(dtype = mask_type,device = GPUdevice)
            
            '''Train'''
            # for n, value in net.image_encoder.named_parameters():
            #     if "Adapter" not in n:
            #         value.requires_grad = False
            imge= net.image_encoder(imgs)

            with torch.no_grad():
                se, de = net.prompt_encoder(
                    points=pt,
                    boxes=None,
                    masks=None,
                )
            pred, _ = net.mask_decoder(
                image_embeddings=imge,
                image_pe=net.prompt_encoder.get_dense_pe(), 
                sparse_prompt_embeddings=se,
                dense_prompt_embeddings=de, 
                multimask_output=False,
              )
            # print('seg', pred.shape, masks.shape)
            # post process
            pred = F.interpolate(
                pred,
                (args.out_size, args.out_size),
                mode="bilinear",
                align_corners=False,
            ) 
            
            dice_loss = lossfunc(pred, masks)
            bce_loss = bce_lossfunc(torch.sigmoid(pred), masks)
            loss = dice_loss + bce_loss

            # draw masks
            if pos_print and first_pos_index >= 0 :
                sigmoid_pred = torch.sigmoid(pred)
                binary_pred = (sigmoid_pred >= args.threshold).float()    
                pred_mask = binary_pred[first_pos_index][0].detach().cpu().numpy()
                gt_mask = masks[first_pos_index][0].detach().cpu().numpy()
                img = imgs[first_pos_index][0].detach().cpu().numpy()
                point_loc = showp[first_pos_index].cpu().numpy()
                # 使用 Matplotlib 绘制图像
                plt.figure(figsize=(10, 3))  # 调整图像大小             

                # 绘制第一个张量的图像
                plt.subplot(131)  # 分割为 1x3 子图，选择第一个子图
                plt.imshow(pred_mask, cmap='gray')
                plt.title('pred_mask')               

                # 绘制第二个张量的图像
                plt.subplot(132)  # 选择第二个子图
                plt.imshow(gt_mask, cmap='gray')
                plt.scatter(point_loc[1], point_loc[0], color='red', marker='o')
                plt.title('gt_mask')               

                # 绘制第三个张量的图像
                plt.subplot(133)  # 选择第三个子图
                plt.imshow(img, cmap='gray')
                plt.scatter(point_loc[1], point_loc[0], color='red', marker='o')
                plt.title('img')               

                # 调整子图布局
                plt.tight_layout()              

                # 保存图像
                plt.savefig(f'image/train/Positive_{epoch}.png')
                pos_print = False

            if neg_print and first_neg_index >= 0 :
                sigmoid_pred = torch.sigmoid(pred)
                binary_pred = (sigmoid_pred >= args.threshold).float()    
                pred_mask = binary_pred[first_neg_index][0].detach().cpu().numpy()
                gt_mask = masks[first_neg_index][0].detach().cpu().numpy()
                img = imgs[first_neg_index][0].detach().cpu().numpy()
                point_loc = showp[first_neg_index].cpu().numpy()
                # 使用 Matplotlib 绘制图像
                plt.figure(figsize=(10, 3))  # 调整图像大小             

                # 绘制第一个张量的图像
                plt.subplot(131)  # 分割为 1x3 子图，选择第一个子图
                plt.imshow(pred_mask, cmap='gray')
                plt.title('pred_mask')               

                # 绘制第二个张量的图像
                plt.subplot(132)  # 选择第二个子图
                plt.imshow(gt_mask, cmap='gray')
                plt.scatter(point_loc[1], point_loc[0], color='red', marker='o')
                plt.title('gt_mask')               

                # 绘制第三个张量的图像
                plt.subplot(133)  # 选择第三个子图
                plt.imshow(img, cmap='gray')
                plt.scatter(point_loc[1], point_loc[0], color='red', marker='o')
                plt.title('img')               

                # 调整子图布局
                plt.tight_layout()              

                # 保存图像
                plt.savefig(f'image/train/Negative_{epoch}.png')
                neg_print = False

            # cal dice, iou score
            # intersection, union = cal_intersection_union(pred, masks)
            # temp = eval_seg(pred, masks, [0.5])
            # intersection_total += intersection
            # union_total += union
            # seg_dice = temp[1]
            # seg_iou = temp[0]
            # seg_dice_total += seg_dice
            # seg_iou_total += seg_iou
            # seg_dice_list.append(seg_dice)
            # seg_iou_list.append(seg_iou)

            pbar.set_postfix(**{'loss (batch)': loss.item()})
            epoch_loss += loss.item()
            loss.backward()

            # nn.utils.clip_grad_value_(net.parameters(), 0.1)
            # optimizer.step()
            # optimizer.zero_grad()
            accumulated_gradients += 1
            if accumulated_gradients == accumulation_steps:
            # 当累积的步数达到设定的步数时，应用累积的梯度更新参数
                optimizer.step()
                optimizer.zero_grad()
                accumulated_gradients = 0  # 重置累积的梯度
                # epoch_loss += loss.item()

            '''vis images'''
            # if vis:
            #     if ind % vis == 0:
            #         namecat = 'Train'
            #         for na in name:
            #             namecat = namecat + na.split('/')[-1].split('.')[0] + '+'
            #         vis_image(imgs,pred,masks, os.path.join(args.path_helper['sample_path'], namecat+'epoch+' +str(epoch) + '.jpg'), reverse=False, points=showp)

            pbar.update()
            # torch.cuda.empty_cache()
    # print('dice_score', (2.0 * intersection_total) / (union_total + 1e-10), intersection_total, union_total)
    # dice_score_avg = (2.0 * intersection_total) / (union_total + 1e-10)
    # iou_score_avg = intersection_total / (union_total - intersection_total)
    # seg_dice_avg = seg_dice_total / len(train_loader)
    # seg_iou_avg = seg_iou_total / len(train_loader) 
    # # print('epoch_loss', epoch_loss, epoch_loss/len(train_loader))
    # seg_dice_median = statistics.median(seg_dice_list)
    # seg_iou_median = statistics.median(seg_iou_list)
    # seg_dice_std = statistics.stdev(seg_dice_list)
    # seg_iou_std = statistics.stdev(seg_iou_list)
    return epoch_loss/len(train_loader) #, dice_score_avg, seg_dice_avg, seg_dice_median, seg_dice_std, iou_score_avg, seg_iou_avg, seg_iou_median, seg_iou_std

def test_sam(args, val_loader, epoch, net: nn.Module, clean_dir=True):
     # eval mode
    net.eval()

    mask_type = torch.float32
    n_val = len(val_loader)  # the number of batch
    ave_res, mix_res = (0,0,0,0), (0,0,0,0)
    rater_res = [(0,0,0,0) for _ in range(6)]
    tot = 0
    hard = 0
    threshold = (0.1, 0.3, 0.5, 0.7, 0.9)
    GPUdevice = torch.device('cuda:' + str(args.gpu_device))
    device = GPUdevice
    dice_score = 0
    intersection_total = 0
    union_total = 0
    if args.thd:
        lossfunc = DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')
    else:
        lossfunc = criterion_G

    with tqdm(total=n_val, desc='test round', unit='batch', leave=False) as pbar:
        for ind, pack in enumerate(val_loader):
            imgsw = pack['image'].to(dtype = torch.float32, device = GPUdevice)
            masksw = pack['label'].to(dtype = torch.float32, device = GPUdevice)
            # for k,v in pack['image_meta_dict'].items():
            #     print(k)
            name = pack['image_meta_dict']['filename_or_obj']
            # 切 patch
            imgsw = imgsw.squeeze()
            masksw = masksw.squeeze()
            imgsw_np = imgsw.cpu().numpy()
            masksw_np = masksw.cpu().numpy()
            patch_size = (64, 64, 64)
            img_patch = patch(imgsw_np, patch_size, -175)
            mask_patch = patch(masksw_np, patch_size, 0)
            imgsw = torch.tensor(img_patch)
            masksw = torch.tensor(mask_patch)


            # print('point', pack['pt'])
            

            for i in range(imgsw.shape[0]):
                imgs = imgsw[i]
                masks = masksw[i]
                imgs, pt, masks, pos_neg = generate_click_prompt(imgs, masks, args.plabel)
                if args.thd:
                    pt = rearrange(pt, 'b n d -> (b d) n')
                    pos_neg = rearrange(pos_neg, 'b d -> (b d)')
                    imgs = rearrange(imgs, 'b c h w d -> (b d) c h w ')
                    masks = rearrange(masks, 'b c h w d -> (b d) c h w ')
                    imgs = imgs.repeat(1,3,1,1)
                    point_labels = torch.ones(imgs.size(0))
                    imgs = torchvision.transforms.Resize((args.image_size,args.image_size))(imgs)
                    masks = torchvision.transforms.Resize((args.out_size,args.out_size))(masks)
                showp = pt

                mask_type = torch.float32
                # ind += 1
                b_size,c,w,h = imgs.size()
                longsize = w if w >=h else h

                if point_labels[0] != -1:
                    # point_coords = samtrans.ResizeLongestSide(longsize).apply_coords(pt, (h, w))
                    point_coords = pt
                    coords_torch = torch.as_tensor(point_coords, dtype=torch.float, device=GPUdevice)
                    labels_torch = torch.as_tensor(point_labels, dtype=torch.int, device=GPUdevice)
                    coords_torch, labels_torch = coords_torch[None, :, :], labels_torch[None, :]
                    pt = (coords_torch, labels_torch)

                '''init'''
                if hard:
                    true_mask_ave = (true_mask_ave > 0.5).float()
                    #true_mask_ave = cons_tensor(true_mask_ave)
                imgs = imgs.to(dtype = mask_type,device = GPUdevice)
                masks = masks.to(dtype = mask_type,device = GPUdevice)
                
                '''test'''
                with torch.no_grad():
                    imge= net.image_encoder(imgs)
                    se, de = net.prompt_encoder(
                        points=pt,
                        boxes=None,
                        masks=None,
                    )
                    pred, _ = net.mask_decoder(
                        image_embeddings=imge,
                        image_pe=net.prompt_encoder.get_dense_pe(),
                        sparse_prompt_embeddings=se,
                        dense_prompt_embeddings=de, 
                        multimask_output=False,
                    )
                    # tot += lossfunc(pred, masks)
                    intersection, union = cal_intersection_union(pred, masks)
                    intersection_total += intersection
                    union_total += union
                    print('dice_score', (2.0 * intersection) / (union + 1e-10), intersection, union)
                    '''vis images'''
                    # if ind % args.vis == 0:
                    #     namecat = 'Test'
                    #     for na in name:
                    #         img_name = na.split('/')[-1].split('.')[0]
                    #         namecat = namecat + img_name + '+'
                    #     vis_image(imgs,pred, masks, os.path.join(args.path_helper['sample_path'], namecat+'epoch+' +str(epoch) + '.jpg'), reverse=False, points=showp)
                    

                    temp = eval_seg(pred, masks, threshold)
                    mix_res = tuple([sum(a) for a in zip(mix_res, temp)])
            pbar.update()
    dice_score =  (2.0 * intersection_total) / (union_total + 1e-10)   
    # if args.evl_chunk:
    #     n_val = n_val * (imgsw.size(-1) // evl_ch)

    return dice_score

def validation_sam(args, val_loader, epoch, net: nn.Module, clean_dir=True):
     # eval mode
    net.eval()

    mask_type = torch.float32
    n_val = len(val_loader)  # the number of batch
    ave_res, mix_res = (0,0,0,0), (0,0,0,0)
    rater_res = [(0,0,0,0) for _ in range(6)]
    tot = 0
    hard = 0
    threshold = (0.1, 0.3, 0.5, 0.7, 0.9)
    GPUdevice = torch.device('cuda:' + str(args.gpu_device))
    device = GPUdevice
    epoch_loss = 0
    pos_print = True
    neg_print = True
    first_pos_index = -1
    first_neg_index = -1

    if args.thd:
        lossfunc = DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')
    else:
        lossfunc = criterion_G

    with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
        for batch_idx, pack in enumerate(val_loader):
            imgs = pack['image'].to(dtype = torch.float32, device = GPUdevice)
            masks = pack['label'].to(dtype = torch.float32, device = GPUdevice)
            # for k,v in pack['image_meta_dict'].items():
            #     print(k)
            if 'pt' not in pack:
                imgs, pt, masks, pos_neg = generate_click_prompt(imgs, masks)
            else:
                pt = pack['pt']
                point_labels = pack['p_label']

            if args.thd:
                pt = rearrange(pt, 'b n d -> (b d) n')
                pos_neg = rearrange(pos_neg, 'b d -> (b d)')
                imgs = rearrange(imgs, 'b c h w d -> (b d) c h w ')
                masks = rearrange(masks, 'b c h w d -> (b d) c h w ')
                imgs = imgs.repeat(1,3,1,1)
                point_labels = torch.ones(imgs.size(0))
                # imgs = torchvision.transforms.Resize((args.image_size,args.image_size))(imgs)
                # masks = torchvision.transforms.Resize((args.out_size,args.out_size))(masks)
            showp = pt
            if torch.sum(pos_neg) != 0 and torch.sum(pos_neg) != 2*args.chunk :
                first_pos_index = torch.nonzero(pos_neg == 1)[0].item()
                first_neg_index = torch.nonzero(pos_neg == 0)[0].item()

            mask_type = torch.float32
            b_size,c,w,h = imgs.size()
            longsize = w if w >=h else h

            if point_labels[0] != -1:
                # point_coords = samtrans.ResizeLongestSide(longsize).apply_coords(pt, (h, w))
                point_coords = pt
                coords_torch = torch.as_tensor(point_coords, dtype=torch.float, device=GPUdevice)
                labels_torch = torch.as_tensor(point_labels, dtype=torch.int, device=GPUdevice)
                coords_torch, labels_torch = coords_torch[None, :, :], labels_torch[None, :]
                pt = (coords_torch, labels_torch)

            '''init'''
            if hard:
                true_mask_ave = (true_mask_ave > 0.5).float()
                #true_mask_ave = cons_tensor(true_mask_ave)
            imgs = imgs.to(dtype = mask_type,device = GPUdevice)
            masks = masks.to(dtype = mask_type,device = GPUdevice)
            
            '''Validate'''
            # for n, value in net.image_encoder.named_parameters():
            #     if "Adapter" not in n:
            #         value.requires_grad = False
            # imge= net.image_encoder(imgs)

            with torch.no_grad():
                imge= net.image_encoder(imgs)
                se, de = net.prompt_encoder(
                    points=pt,
                    boxes=None,
                    masks=None,
                )
                pred, _ = net.mask_decoder(
                    image_embeddings=imge,
                    image_pe=net.prompt_encoder.get_dense_pe(), 
                    sparse_prompt_embeddings=se,
                    dense_prompt_embeddings=de, 
                    multimask_output=False,
                  )
                # print('seg', pred.shape, masks.shape)
                # post process
                pred = F.interpolate(
                    pred,
                    (args.out_size, args.out_size),
                    mode="bilinear",
                    align_corners=False,
                ) 
            
            dice_loss = lossfunc(pred, masks)
            bce_loss = bce_lossfunc(torch.sigmoid(pred), masks)
            loss = dice_loss + bce_loss

            pbar.set_postfix(**{'loss (batch)': loss.item()})
            epoch_loss += loss.item()
            # draw masks
            if pos_print and first_pos_index >= 0 :
                sigmoid_pred = torch.sigmoid(pred)
                binary_pred = (sigmoid_pred >= args.threshold).float()    
                pred_mask = binary_pred[first_pos_index][0].detach().cpu().numpy()
                gt_mask = masks[first_pos_index][0].detach().cpu().numpy()
                img = imgs[first_pos_index][0].detach().cpu().numpy()
                point_loc = showp[first_pos_index].cpu().numpy()
                # 使用 Matplotlib 绘制图像
                plt.figure(figsize=(10, 3))  # 调整图像大小             

                # 绘制第一个张量的图像
                plt.subplot(131)  # 分割为 1x3 子图，选择第一个子图
                plt.imshow(pred_mask, cmap='gray')
                plt.title('pred_mask')               

                # 绘制第二个张量的图像
                plt.subplot(132)  # 选择第二个子图
                plt.imshow(gt_mask, cmap='gray')
                plt.scatter(point_loc[1], point_loc[0], color='red', marker='o')
                plt.title('gt_mask')               

                # 绘制第三个张量的图像
                plt.subplot(133)  # 选择第三个子图
                plt.imshow(img, cmap='gray')
                plt.scatter(point_loc[1], point_loc[0], color='red', marker='o')
                plt.title('img')               

                # 调整子图布局
                plt.tight_layout()              

                # 保存图像
                plt.savefig(f'image/valid/Positive_{epoch}.png')
                pos_print = False

            if neg_print and first_neg_index >= 0 :
                sigmoid_pred = torch.sigmoid(pred)
                binary_pred = (sigmoid_pred >= args.threshold).float()    
                pred_mask = binary_pred[first_neg_index][0].detach().cpu().numpy()
                gt_mask = masks[first_neg_index][0].detach().cpu().numpy()
                img = imgs[first_neg_index][0].detach().cpu().numpy()
                point_loc = showp[first_neg_index].cpu().numpy()
                # 使用 Matplotlib 绘制图像
                plt.figure(figsize=(10, 3))  # 调整图像大小             

                # 绘制第一个张量的图像
                plt.subplot(131)  # 分割为 1x3 子图，选择第一个子图
                plt.imshow(pred_mask, cmap='gray')
                plt.title('pred_mask')               

                # 绘制第二个张量的图像
                plt.subplot(132)  # 选择第二个子图
                plt.imshow(gt_mask, cmap='gray')
                plt.scatter(point_loc[1], point_loc[0], color='red', marker='o')
                plt.title('gt_mask')               

                # 绘制第三个张量的图像
                plt.subplot(133)  # 选择第三个子图
                plt.imshow(img, cmap='gray')
                plt.scatter(point_loc[1], point_loc[0], color='red', marker='o')
                plt.title('img')               

                # 调整子图布局
                plt.tight_layout()              

                # 保存图像
                plt.savefig(f'image/valid/Negative_{epoch}.png')
                neg_print = False
            pbar.update()
            # torch.cuda.empty_cache()
    # plabel_dice_avg = [dice / n_val for dice in plabel_dice_total]
    # plabel_iou_avg = [iou / n_val for iou in plabel_iou_total]
    return epoch_loss / len(val_loader) #, iou_score_avg, dice_score_avg #plabel_dice_avg, plabel_iou_avg


def eval_sam(args, eval_loader, epoch, net: nn.Module, clean_dir=True):
     # eval mode
    net.eval()

    mask_type = torch.float32
    n_val = len(eval_loader)  # the number of batch
    hard = 0
    threshold = (0.1, 0.3, 0.5, 0.7, 0.9)
    GPUdevice = torch.device('cuda:' + str(args.gpu_device))
    device = GPUdevice
    plabel = 1
    patch_num_list = []

    with tqdm(total=n_val, desc='evaluate round', unit='batch', leave=False) as pbar:
        pos_num = 0
        pos_intersection_total = 0
        pos_union_total = 0
        total_dice_patch_list = []
        neg_intersection_total = 0
        neg_union_total = 0
        neg_num = 0

        for ind, pack in enumerate(eval_loader):
            imgsw = pack['image'].to(dtype = torch.float32, device = GPUdevice)
            masksw = pack['label'].to(dtype = torch.float32, device = GPUdevice)
            total_intersection_total = 0
            total_union_total = 0

            imgsw = imgsw.squeeze()
            masksw = masksw.squeeze()
            imgsw_np = imgsw.cpu().numpy()
            masksw_np = masksw.cpu().numpy()
            patch_size = (args.roi_size, args.roi_size, args.evl_chunk)
            img_patch = patch(imgsw_np, patch_size, -175)
            mask_patch = patch(masksw_np, patch_size, 0)
            imgsw = torch.tensor(img_patch)
            masksw = torch.tensor(mask_patch)
            patch_num_list.append(imgsw.shape[0])

            for i in range(imgsw.shape[0]):
                imgs = imgsw[i]
                masks = masksw[i]
                imgs, pt, masks, pos_neg = generate_click_prompt(imgs, masks, plabel)
                if args.thd:
                    pt = rearrange(pt, 'b n d -> (b d) n')
                    pos_neg = rearrange(pos_neg, 'b d -> (b d)')
                    imgs = rearrange(imgs, 'b c h w d -> (b d) c h w ')
                    masks = rearrange(masks, 'b c h w d -> (b d) c h w ')
                    imgs = imgs.repeat(1,3,1,1)
                    point_labels = torch.ones(imgs.size(0))

                mask_type = torch.float32

                if point_labels[0] != -1:
                    # point_coords = samtrans.ResizeLongestSide(longsize).apply_coords(pt, (h, w))
                    point_coords = pt
                    coords_torch = torch.as_tensor(point_coords, dtype=torch.float, device=GPUdevice)
                    labels_torch = torch.as_tensor(point_labels, dtype=torch.int, device=GPUdevice)
                    coords_torch, labels_torch = coords_torch[None, :, :], labels_torch[None, :]
                    pt = (coords_torch, labels_torch)
                    
                imgs = imgs.to(dtype = mask_type,device = GPUdevice)
                masks = masks.to(dtype = mask_type,device = GPUdevice)

                '''test'''
                with torch.no_grad():
                    imge= net.image_encoder(imgs)
                    se, de = net.prompt_encoder(
                        points=pt,
                        boxes=None,
                        masks=None,
                    )
                    pred, _ = net.mask_decoder(
                        image_embeddings=imge,
                        image_pe=net.prompt_encoder.get_dense_pe(),
                        sparse_prompt_embeddings=se,
                        dense_prompt_embeddings=de, 
                        multimask_output=False,
                    )
                    # post process
                    pred = F.interpolate(
                        pred,
                        (args.out_size, args.out_size),
                        mode="bilinear",
                        align_corners=False,
                    ) 
                    # comaper slice based 3d dice 
                    # metric = DiceMetric(include_background=True, reduction="mean")
                    # d1 = cal_3dslice_dice(pred, masks)
                    # binary_y_true = (masks > 0.5).float()
                    # sigmoid_pred = torch.sigmoid(pred)
                    # binary_y_pred = (sigmoid_pred >= 0.5).float()
                    # d2 = metric(binary_y_pred, binary_y_true)
                    # print('compare', d1, d2)

                    for i in  range(pos_neg.shape[0]) :
                        intersection, union = cal_intersection_union(pred[i], masks[i])
                        total_intersection_total += intersection
                        total_union_total += union
                        if  pos_neg[i] == 0:
                            neg_intersection_total += intersection
                            neg_union_total += union
                            neg_num += 1
                        else :
                            pos_intersection_total += intersection
                            pos_union_total += union
                            pos_num += 1
            total_dice_patch = (2.0 * total_intersection_total) / (total_union_total + 1e-10)
            total_dice_patch_list.append(total_dice_patch)
            pbar.update()

        neg_ratio = neg_num / (pos_num + neg_num)
        pos_pf = (2.0 * pos_intersection_total) / (pos_union_total + 1e-10)
        neg_pf = (neg_union_total / neg_num) / (args.roi_size * args.roi_size)
        overall_dice = sum(total_dice_patch_list) / len(total_dice_patch_list)
        print('neg / total', neg_ratio, pos_num, neg_num)  
        print('pos_pf', pos_pf) 
        print('neg_pf', neg_pf)
        print('overall_dice', overall_dice) 
            
    return overall_dice, pos_pf, neg_pf

# def validation_sam(args, val_loader, epoch, net: nn.Module, clean_dir=True):
#      # eval mode
#     net.eval()

#     mask_type = torch.float32
#     n_val = len(val_loader)  # the number of batch
#     ave_res, mix_res = (0,0,0,0), (0,0,0,0)
#     rater_res = [(0,0,0,0) for _ in range(6)]
#     tot = 0
#     hard = 0
#     threshold = (0.1, 0.3, 0.5, 0.7, 0.9)
#     GPUdevice = torch.device('cuda:' + str(args.gpu_device))
#     device = GPUdevice

#     if args.thd:
#         lossfunc = DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')
#     else:
#         lossfunc = criterion_G

#     with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
#         for ind, pack in enumerate(val_loader):
#             imgsw = pack['image'].to(dtype = torch.float32, device = GPUdevice)
#             masksw = pack['label'].to(dtype = torch.float32, device = GPUdevice)
#             # for k,v in pack['image_meta_dict'].items():
#             #     print(k)
#             if 'pt' not in pack:
#                 imgsw, ptw, masksw = generate_click_prompt(imgsw, masksw)
#             else:
#                 ptw = pack['pt']
#                 point_labels = pack['p_label']
#             name = pack['image_meta_dict']['filename_or_obj']
            
#             buoy = 0
#             if args.evl_chunk:
#                 evl_ch = int(args.evl_chunk)
#             else:
#                 evl_ch = int(imgsw.size(-1))

#             while (buoy + evl_ch) <= imgsw.size(-1):
#                 if args.thd:
#                     pt = ptw[:,:,buoy: buoy + evl_ch]
#                 else:
#                     pt = ptw

#                 imgs = imgsw[...,buoy:buoy + evl_ch]
#                 masks = masksw[...,buoy:buoy + evl_ch]
#                 buoy += evl_ch

#                 if args.thd:
#                     pt = rearrange(pt, 'b n d -> (b d) n')
#                     imgs = rearrange(imgs, 'b c h w d -> (b d) c h w ')
#                     masks = rearrange(masks, 'b c h w d -> (b d) c h w ')
#                     imgs = imgs.repeat(1,3,1,1)
#                     point_labels = torch.ones(imgs.size(0))

#                     imgs = torchvision.transforms.Resize((args.image_size,args.image_size))(imgs)
#                     masks = torchvision.transforms.Resize((args.out_size,args.out_size))(masks)
                
#                 showp = pt

#                 mask_type = torch.float32
#                 ind += 1
#                 b_size,c,w,h = imgs.size()
#                 longsize = w if w >=h else h

#                 if point_labels[0] != -1:
#                     # point_coords = samtrans.ResizeLongestSide(longsize).apply_coords(pt, (h, w))
#                     point_coords = pt
#                     coords_torch = torch.as_tensor(point_coords, dtype=torch.float, device=GPUdevice)
#                     labels_torch = torch.as_tensor(point_labels, dtype=torch.int, device=GPUdevice)
#                     coords_torch, labels_torch = coords_torch[None, :, :], labels_torch[None, :]
#                     pt = (coords_torch, labels_torch)

#                 '''init'''
#                 if hard:
#                     true_mask_ave = (true_mask_ave > 0.5).float()
#                     #true_mask_ave = cons_tensor(true_mask_ave)
#                 imgs = imgs.to(dtype = mask_type,device = GPUdevice)
                
#                 '''test'''
#                 with torch.no_grad():
#                     imge= net.image_encoder(imgs)

#                     se, de = net.prompt_encoder(
#                         points=pt,
#                         boxes=None,
#                         masks=None,
#                     )

#                     pred, _ = net.mask_decoder(
#                         image_embeddings=imge,
#                         image_pe=net.prompt_encoder.get_dense_pe(),
#                         sparse_prompt_embeddings=se,
#                         dense_prompt_embeddings=de, 
#                         multimask_output=False,
#                     )
#                     # post process
#                     pred = F.interpolate(
#                         pred,
#                         (args.out_size, args.out_size),
#                         mode="bilinear",
#                         align_corners=False,
#                     ) 
                
#                     tot += lossfunc(pred, masks)

#                     '''vis images'''
#                     if ind % args.vis == 0:
#                         namecat = 'Test'
#                         for na in name:
#                             img_name = na.split('/')[-1].split('.')[0]
#                             namecat = namecat + img_name + '+'
#                         vis_image(imgs,pred, masks, os.path.join(args.path_helper['sample_path'], namecat+'epoch+' +str(epoch) + '.jpg'), reverse=False, points=showp)
                    

#                     temp = eval_seg(pred, masks, threshold)
#                     mix_res = tuple([sum(a) for a in zip(mix_res, temp)])

#             pbar.update()

#     if args.evl_chunk:
#         print('b4', n_val)
#         n_val = n_val * (imgsw.size(-1) // evl_ch)
#         print('after', n_val, imgsw.size(-1), evl_ch)

#     return tot/ n_val , tuple([a/n_val for a in mix_res])

            # for j in range(len(plabel_list)):
            #     plable_intersection_total = 0
            #     plable_union_total = 0
            #     for i in range(imgsw.shape[0]):
            #         imgs = imgsw[i]
            #         masks = masksw[i]
            #         imgs, pt, masks = generate_click_prompt(imgs, masks, plabel_list[j])
            #         if args.thd:
            #             pt = rearrange(pt, 'b n d -> (b d) n')
            #             imgs = rearrange(imgs, 'b c h w d -> (b d) c h w ')
            #             masks = rearrange(masks, 'b c h w d -> (b d) c h w ')
            #             imgs = imgs.repeat(1,3,1,1)
            #             point_labels = torch.ones(imgs.size(0))

            #         mask_type = torch.float32
            #         # ind += 1
            #         b_size,c,w,h = imgs.size()
            #         longsize = w if w >=h else h

            #         if point_labels[0] != -1:
            #             # point_coords = samtrans.ResizeLongestSide(longsize).apply_coords(pt, (h, w))
            #             point_coords = pt
            #             coords_torch = torch.as_tensor(point_coords, dtype=torch.float, device=GPUdevice)
            #             labels_torch = torch.as_tensor(point_labels, dtype=torch.int, device=GPUdevice)
            #             coords_torch, labels_torch = coords_torch[None, :, :], labels_torch[None, :]
            #             pt = (coords_torch, labels_torch)

            #         '''init'''
            #         if hard:
            #             true_mask_ave = (true_mask_ave > 0.5).float()
            #             #true_mask_ave = cons_tensor(true_mask_ave)
            #         imgs = imgs.to(dtype = mask_type,device = GPUdevice)
            #         masks = masks.to(dtype = mask_type,device = GPUdevice)

            #         '''test'''
            #         with torch.no_grad():
            #             imge= net.image_encoder(imgs)
            #             se, de = net.prompt_encoder(
            #                 points=pt,
            #                 boxes=None,
            #                 masks=None,
            #             )
            #             pred, _ = net.mask_decoder(
            #                 image_embeddings=imge,
            #                 image_pe=net.prompt_encoder.get_dense_pe(),
            #                 sparse_prompt_embeddings=se,
            #                 dense_prompt_embeddings=de, 
            #                 multimask_output=False,
            #             )
            #             # post process
            #             pred = F.interpolate(
            #                 pred,
            #                 (args.out_size, args.out_size),
            #                 mode="bilinear",
            #                 align_corners=False,
            #             ) 
                        
            #             intersection, union = cal_intersection_union(pred, masks)
            #             plable_intersection_total += intersection
            #             plable_union_total += union

            #     dice_score = (2.0 * plable_intersection_total) / (plable_union_total + 1e-10)
            #     iou_score = plable_intersection_total / (plable_union_total - plable_intersection_total)
            #     plabel_dice_total[j] += dice_score
            #     plabel_iou_total[j] += iou_score