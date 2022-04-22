# -*- coding: utf-8 -*-
'''
@Time    : 2022/4/21 19:37
@Author  : LYZ
@FileName: MyDataset.py
@Software: PyCharm
'''
from torch.utils.data import Dataset
import glob
from Eval import pre


class MyDataset(Dataset):
    def __init__(self, noise_dir, gt_dir):
        super(Dataset, self).__init__()
        self.noise_dir = [f for f in glob.glob(noise_dir)]
        self.gt_dir = [f for f in glob.glob(gt_dir)]

    def __len__(self):
        return len(self.noise_dir)

    def __getitem__(self, index):
        noise_dir = self.noise_dir[index]
        noise_img, _, _ = pre(noise_dir)
        gt_dir = self.gt_dir[index]
        gt_img, _, _ = pre(gt_dir)
        data = {'noisy': noise_img, 'ground_truth': gt_img}
        return data
