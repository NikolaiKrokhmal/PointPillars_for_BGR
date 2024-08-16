import argparse
import cv2
import numpy as np
import os
import torch
import pdb
import time
from tqdm import tqdm

from utils import setup_seed, read_points, read_pickle,shuffle_pickle,  read_calib, read_label, \
    keep_bbox_from_image_range, keep_bbox_from_lidar_range, vis_pc, \
     bbox3d2corners_camera, points_camera2image, \
    bbox_camera2lidar
from model import PointPillarsPollo


def point_range_filter(pts, point_range=[-40, -40, -2, 40, 40, 2]):
    '''
    data_dict: dict(pts, gt_bboxes_3d, gt_labels, gt_names, difficulty)
    point_range: [x1, y1, z1, x2, y2, z2]
    '''
    flag_x_low = pts[:, 0] > point_range[0]
    flag_y_low = pts[:, 1] > point_range[1]
    flag_z_low = pts[:, 2] > point_range[2]
    flag_x_high = pts[:, 0] < point_range[3]
    flag_y_high = pts[:, 1] < point_range[4]
    flag_z_high = pts[:, 2] < point_range[5]
    keep_mask = flag_x_low & flag_y_low & flag_z_low & flag_x_high & flag_y_high & flag_z_high
    pts = pts[keep_mask]
    return pts

def dict2numpy(dict, adding):
    data = list(dict.keys())
    adding = adding.tolist()
    cones = np.zeros((len(data),7), dtype=float)
    for i,cone in enumerate(data):
        cones[i, :] = dict[cone]['location'][:2]+adding
    return cones



def main(args):
    if not args.no_cuda:
        model = PointPillarsPollo().cuda()
        model.load_state_dict(torch.load(args.ckpt))
    else:
        model = PointPillarsPollo()
        model.load_state_dict(torch.load(args.ckpt, map_location=torch.device('cpu')))

    if not os.path.exists(args.pc_path):
        raise FileNotFoundError

    pickle = (shuffle_pickle(args.pc_path))
    frame = pickle[next(iter(pickle))]
    pc = read_points(frame['path'])
    pc = point_range_filter(pc)
    pc_torch = torch.from_numpy(pc)

    model.eval()
    with torch.no_grad():
        if not args.no_cuda:
            pc_torch = pc_torch.cuda()
        start_time = time.time()                                ###############################################
        result_filter = model(batched_pts=[pc_torch],
                              mode='test')[0]
        end_time = time.time()                                  ###############################################
        frame_time = end_time - start_time
        print(frame_time)
    lidar_bboxes = result_filter['lidar_bboxes']
    lidar_bboxes[:, 2] = sum(cone['location'][2] for cone in frame['cones'].values()) / len(frame['cones'])
    real_bbox = dict2numpy(frame['cones'], lidar_bboxes[0, 2:])
    print(f"predicted cones: {len(lidar_bboxes)}, real cones: {len(real_bbox)}")

    vis_pc(pc, lidar_bboxes, real_bbox)
    print(f"frame time is: {frame_time}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Configuration Parameters')
    parser.add_argument('--ckpt', default='./pillar_logs/checkpoints/epoch_60.pth') #'./logs_backup/epoch_60_niko.pth'
    parser.add_argument('--pc_path', default='../../Data-ApolloScape/PCD_MAP.pkl')
    parser.add_argument('--no_cuda', action='store_true', help='whether to use cuda')
    args = parser.parse_args()

    main(args)
