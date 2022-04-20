# -*- coding: utf-8 -*-
'''
@Time    : 2022/4/20 18:26
@Author  : LYZ
@FileName: main.py
@Software: PyCharm
'''

import os
import torch.nn as nn
from Network import RIDNet
import torch
from Eval import pre
from tqdm import tqdm
from pytorch_mssim import MS_SSIM

noisy_data_path = "dataset/noisy/"
origin_data_path = "dataset/ground_truth/"
NoisyFiles = os.listdir(noisy_data_path)
OriginFiles = os.listdir(origin_data_path)
NoisyFiles_len = len(NoisyFiles)
device = "cuda:0"
lr = 0.0001
# loss = nn.L1Loss()
epochs = 1000
# model_path = "CBD-880.pth"
white_level = 16383
black_level = 1024

Loss = MS_SSIM(data_range=white_level, size_average=True, channel=4)

if __name__ == "__main__":
    net = RIDNet(4, 4, 32).to(device)
    # net.load_state_dict(torch.load(model_path))

    for epoch in tqdm(range(epochs)):
        if epoch < 100:
            lr = 0.0001
        elif epoch >= 100 and epoch < 200:
            lr = 0.00005
        elif epoch >= 200 and epoch < 400:
            lr = 0.00001
        elif epoch > 400:
            lr = 0.000005

        optimizer = torch.optim.Adam(net.parameters(), lr=lr)
        net.train()
        net.to(device)

        running_loss = 0.0
        for i in tqdm(range(NoisyFiles_len)):
            X, X_height, X_width = pre(input_path=noisy_data_path + str(i) + "_noise.dng")
            Y, Y_height, Y_width = pre(input_path=origin_data_path + str(i) + "_gt.dng")
            X, Y = X.to(device), Y.to(device)
            optimizer.zero_grad()
            Y_HAT = net(X)
            l = 1 - Loss(Y_HAT, Y)
            l.backward()
            optimizer.step()

            running_loss += l.item()
        print("Epoch{}\tloss {}".format(epoch, running_loss / NoisyFiles_len))

        with open("loss.txt", 'a+') as f:
            f.writelines("Epoch{}\tloss {}".format(epoch, running_loss / NoisyFiles_len))

        print()
        torch.save(net.state_dict(), 'models/RID-L1-' + str(epoch) + '.pth')
