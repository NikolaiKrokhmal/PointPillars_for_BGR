import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from functools import partial


def collate_fn(list_data):
    batched_pts_list = []
    batched_labels_list = []
    batched_names_list = []
    batched_gt_bboxes_list = []
    for data_dict in list_data:
        pts = data_dict['pts']

        cones = data_dict['gt_labels']
        gt_bboxes = []
        gt_names = []
        for cone in cones:
            gt_bboxes.append(cones[cone]['location'])
            gt_names.append(cones[cone]['class'])
        gt_labels = np.array([0] * len(cones))
        batched_pts_list.append(torch.from_numpy(pts))
        batched_labels_list.append(torch.from_numpy(gt_labels))
        batched_names_list.append(gt_names)  # List(str)
        batched_gt_bboxes_list.append(torch.from_numpy(np.array(gt_bboxes)))

    rt_data_dict = dict(
        batched_pts=batched_pts_list,
        batched_labels=batched_labels_list,
        batched_names=batched_names_list,
        batched_gt_bboxes=batched_gt_bboxes_list,
    )

    return rt_data_dict


def get_dataloader(dataset, batch_size, num_workers, shuffle=True, drop_last=False):
    collate = collate_fn
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=drop_last,
        collate_fn=collate,
    )
    return dataloader
