# -*- coding: utf-8 -*-
'''
@Time    : 2022/4/20 18:26
@Author  : LYZ
@FileName: main.py
@Software: PyCharm
'''

import os

import rawpy
import skimage
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader

from MyDataset import MyDataset
from Network import RIDNet
import numpy as np
from Eval import inv_normalization, write_image
from Eval import pre
from tqdm import tqdm
from loss import *

# noisy_data_path = "dataset/noisy/[0-99]_noise.dng"
# origin_data_path = "dataset/ground_truth/[0-99]_gt.dng"

# windows os test
noisy_data_path = "C:/Users/LYZ/Desktop/dataset/noisy/[0-99]_noise.dng"
origin_data_path = "C:/Users/LYZ/Desktop/dataset/ground_truth/[0-99]_gt.dng"

lr = 0.0001
epochs = 1000
white_level = 16383
black_level = 1024
start_epoch = 0
batch_size = 1
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

if __name__ == "__main__":
    net = RIDNet(4, 4, 32)
    net = nn.DataParallel(net)
    net = net.cuda()

    # 取出模型
    # net.load_state_dict(torch.load(model_path))

    optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-5)
    criterion = Smooth_L1_Loss().cuda()
    # 动态学习率
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20, eta_min=1e-5, verbose=1)

    # 加载数据
    train_dataset = MyDataset(noisy_data_path, origin_data_path)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    for epoch in range(epochs):

        net.train()

        train_loss = 0.0
        loop = tqdm(enumerate(train_loader), total=len(train_loader), leave=False, ncols=80)
        for i, data in loop:
            noisy = data['noisy'].cuda()
            gt = data['ground_truth'].cuda()
            optimizer.zero_grad()
            pred = net(noisy)
            loss = criterion(pred, gt)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            loop.set_description(f'Epoch [{epoch}/{epochs}')

        # 动态学习率
        current_lr = scheduler.optimizer.param_groups[0]['lr']
        scheduler.step()

        net.eval()
        with torch.no_grad():
            noisy_path = "dataset/noisy/0_noise.dng"
            gt_path = "dataset/ground_truth/0_gt.dng"
            XX, height, width = pre(noisy_path)
            XX = XX.cuda()
            YY = net(XX)
            result_data = YY.detach().to("cpu").numpy().transpose(0, 2, 3, 1)
            result_data = inv_normalization(result_data, black_level=1024, white_level=16383)
            result_write_data = write_image(result_data, height, width)
            gt = rawpy.imread(gt_path).raw_image_visible
            psnr = skimage.metrics.peak_signal_noise_ratio(
                gt.astype(np.float64), result_write_data.astype(np.float64), data_range=white_level)
            ssim = skimage.metrics.structural_similarity(
                gt.astype(np.float64), result_write_data.astype(np.float64), channel_axis=True, data_range=white_level)
            print('psnr:', psnr)
            print('ssim:', ssim)

        # 保存模型
        torch.save(net.state_dict(), "models/RID-" + str(epoch + start_epoch) + ".pth")
