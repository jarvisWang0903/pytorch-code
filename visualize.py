import torch
import torch.nn as nn
import torch.nn.functional as F
from options.test_options import cfg
import json
import os.path as osp
import os
import numpy as np
from tqdm import tqdm
import time
import argparse
from model.deeplabv2 import get_deeplab_v2
from model.DISE.model import SharedEncoder, PrivateEncoder
from model.DISE.seg_model import Classifier_Module
from model.DISE.model import Memory
import cv2
import matplotlib.pyplot as plt


input_size_target = (512, 256)

def get_single_feature(c_memory, i):

    print(c_memory.shape)
    interp = nn.Upsample(size=(input_size_target[1], input_size_target[0]), mode='bilinear',
                         align_corners=True)
    features = interp(c_memory)
    feature=features[:,i,:,:]
    print(feature.shape)

    feature=feature.view(feature.shape[1],feature.shape[2])
    print(feature.shape)

    return feature


def save_feature_to_img(c_memory):
    #to numpy
    feature=get_single_feature(c_memory, 0)
    feature=feature.cpu().data.numpy()

    #use sigmod to [0,1]
    feature = 1.0/(1+np.exp(-1*feature))

    # to [0,255]
    feature=np.round(feature*255)
    print(feature[0])

    cv2.imwrite('/home/groupprofzli/data1/ycwang/baseline_dst/vis/img.jpg',feature)


def save_feature_to_het_img(c_memory):
    fig = plt.figure(figsize=(16, 16))
    fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95, wspace=0.05, hspace=0.05)
    # to numpy
    feature = get_single_feature(c_memory, 0)
    feature = feature.cpu().data.numpy()

    pmin = np.min(feature)
    pmax = np.max(feature)
    img = ((feature - pmin) / (pmax - pmin + 0.000001)) * 255  # float在[0，1]之间，转换成0-255
    img = img.astype(np.uint8)  # 转成unit8
    img = cv2.applyColorMap(img, cv2.COLORMAP_JET)  # 生成heat map
    #img = img[:, :, ::-1]  # 注意cv2（BGR）和matplotlib(RGB)通道是相反的
    # plt.imshow(img)
    # fig.savefig(savename, dpi=100)
    cv2.imwrite('/home/groupprofzli/data1/ycwang/baseline_dst/vis/img.jpg', img)

def draw_features(x, savename):
    tic = time.time()
    fig = plt.figure(figsize=(16, 16))
    fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95, wspace=0.05, hspace=0.05)
    for i in range(8*8):
        plt.subplot(8, 8, i + 1)
        plt.axis('off')
        feature = get_single_feature(x, i)
        img = feature.cpu().data.numpy()
        pmin = np.min(img)
        pmax = np.max(img)
        img = ((img - pmin) / (pmax - pmin + 0.000001)) * 255  # float在[0，1]之间，转换成0-255
        img = img.astype(np.uint8)  # 转成unit8
        img = cv2.applyColorMap(img, cv2.COLORMAP_JET)  # 生成heat map
        img = img[:, :, ::-1]  # 注意cv2（BGR）和matplotlib(RGB)通道是相反的
        plt.imshow(img)
        print("{}/{}".format(i, 1))
    fig.savefig(savename, dpi=100)
    fig.clf()
    plt.close()
    print("time:{}".format(time.time() - tic))

if __name__ == '__main__':
    MODEL_DIR = r'/home/groupprofzli/data1/ycwang/baseline_dst/baseline_dst_exp_6/snapshots'
    i_iter = 4000
    c_memory_restor_from = osp.join(MODEL_DIR, f'common_memory_main_{i_iter}.pth.tar')

    if c_memory_restor_from is not None:
        c_memory = Memory(cfg.EVAL.COMMON_MEMORY_SIZE, c_memory_restor_from)
    savepath = '/home/groupprofzli/data1/ycwang/baseline_dst/vis'
    #feature = F.adaptive_avg_pool2d(c_memory.memory, (c_memory.memory.size(2), c_memory.memory.size(2)))
    draw_features(c_memory.memory, "{}/f7_layer3.png".format(savepath))