import torch
import torch.nn as nn
import torch.nn.functional as F
from data import CreateTrgDataLoader
from PIL import Image
import json
import os.path as osp
import os
import numpy as np
from model import CreateModel
from tqdm import tqdm
import time
from tensorboardX import SummaryWriter
import matplotlib as mpl
mpl.use('Agg')
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
import argparse


filter_list = [7, 14]

marker = { 0:'Δ', 1:'x', 2:'o' }
            #red                    #blue                  #green
map_data = {0:plt.cm.Set1(0), 1:plt.cm.Set1(1), 2:plt.cm.Set1(2)}
def get_label(dataset, _class):
    return str(dataset) + ' ' +str(_class)

def plot_embedding(data, label):
    #label = label.numpy()
    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)
    fig = plt.figure()
    for i in range(data.shape[0]):
        # plt.scatter(data[i, 0], data[i, 1], c=map_data[label[i][0]],
        #             marker=marker[label[i][1]], label=get_label(label[i][0],label[i][1]))
        #plt.cm.Set1((map_data.index(label[i][0]) + 1) / 10.)
        plt.text(data[i, 0], data[i, 1],
                 marker[label[i].item()],
                 color= map_data[label[i].item()], fontdict={'weight': 'bold', 'size': 9})
    plt.xticks([])
    plt.yticks([])
    plt.title('t-SNE embedding')
    plt.savefig('./t.png')



def tsne(vector, labels):
    print("Generating T-SNE...")
    tsne = TSNE(n_components=2, init='pca', random_state=0)
    tSNE_result = tsne.fit_transform(vector)
    tSNE_label = labels
    plot_embedding(tSNE_result, tSNE_label)

def main():
    model_restore_from = '/home/groupprofzli/data1/ycwang/baseline_dst_new/exp_4/private_memory_84000.pth.tar'
    state_dict = torch.load(model_restore_from, map_location=lambda storage, loc: storage)
    mem = state_dict['memory']
    n_lbl, N, n_dim = mem.size()
    print(n_lbl, N, n_dim)

    lbl = torch.LongTensor(list(range(3))).view(3, 1).repeat(1, N).view(-1, 1)
    mem = mem.view(-1, n_dim)
    mem = mem[100:399]
    print(lbl.size(), mem.size())
    tsne(mem, lbl)


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = '1'
    main()


