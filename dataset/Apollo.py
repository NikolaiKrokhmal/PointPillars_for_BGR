import numpy as np
import os
import torch
from torch.utils.data import Dataset

import sys

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(BASE))

from utils import read_pickle, read_points, bbox_camera2lidar
from dataset import data_augment


class BaseSampler():
    def __init__(self, sampled_list, shuffle=True):
        self.total_num = len(sampled_list)
        self.sampled_list = np.array(sampled_list)
        self.indices = np.arange(self.total_num)
        if shuffle:
            np.random.shuffle(self.indices)
        self.shuffle = shuffle
        self.idx = 0

    def sample(self, num):
        if self.idx + num < self.total_num:
            ret = self.sampled_list[self.indices[self.idx:self.idx + num]]
            self.idx += num
        else:
            ret = self.sampled_list[self.indices[self.idx:]]
            self.idx = 0
            if self.shuffle:
                np.random.shuffle(self.indices)
        return ret


class Apollo(Dataset):
    def __init__(self, data_root, split, percent = 80):
        assert split in ['train', 'val']
        self.data_root = data_root
        self.split = split
        self.data = read_pickle(os.path.join(data_root))
        self.frame_names = list(self.data.keys())
        split_index = int(len(self.frame_names) * float(percent)/100)
        if self.split == 'train':
            self.frame_names = self.frame_names[:split_index]
        elif self.split == 'val':
            self.frame_names = self.frame_names[split_index:]
        self.data = {key: self.data[key] for key in self.frame_names}


        db_sampler = BaseSampler(self.frame_names, shuffle=True)

        self.data_aug_config = dict(
            db_sampler=db_sampler,
            object_noise=dict(
                num_try=100,
                translation_std=[0.25, 0.25, 0.25],
                rot_range=[-0.15707963267, 0.15707963267]
                ),
            random_flip_ratio=0.5,
            global_rot_scale_trans=dict(
                rot_range=[-0.78539816, 0.78539816],
                scale_ratio_range=[0.95, 1.05],
                translation_std=[0, 0, 0]
                ),
            object_range_filter=[0, -39.68, -3, 69.12, 39.68, 1]
        )

    def __getitem__(self, index):
        data_info = self.data[self.frame_names[index]]
        # annotations input
        pts = read_points(data_info["path"])
        # annos_name = annos_info['name']
        # annos_location = annos_info['location']
        # annos_dimension = annos_info['dimensions']
        # rotation_y = annos_info['rotation_y']
        # gt_bboxes = np.concatenate([annos_location, annos_dimension, rotation_y[:, None]], axis=1).astype(np.float32)
        # gt_bboxes_3d = bbox_camera2lidar(gt_bboxes, tr_velo_to_cam, r0_rect)
        # gt_labels = [self.CLASSES.get(name, -1) for name in annos_name]
        data_dict = {
            'pts': pts,
            'gt_labels': data_info['cones']
        }

        return data_dict

    def __len__(self):
        return len(self.frame_names)


if __name__ == '__main__':
    apollo = Apollo(data_root="C:\\Yuval\\Me\\Projects\\Final Project\\Data-ApolloScape\\PCD_MAP.pkl",
                   split="train")
    x = apollo[9]
