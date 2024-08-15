import numpy as np
import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.anchors_pollo import AnchorsPollo, anchor_target, anchors2bboxes
from ops import Voxelization, nms_cuda
from utils import limit_period

torch.set_default_dtype(torch.float32)

class PillarLayer(nn.Module):
    def __init__(self, voxel_size, point_cloud_range, max_num_points, max_voxels):
        super().__init__()
        self.voxel_layer = Voxelization(voxel_size=voxel_size,
                                        point_cloud_range=point_cloud_range,
                                        max_num_points=max_num_points,
                                        max_voxels=max_voxels)

    @torch.no_grad()
    def forward(self, batched_pts):
        '''
        batched_pts: list[tensor], len(batched_pts) = bs
        return:
               pillars: (p1 + p2 + ... + pb, num_points, c - raw point featuers (x,y,z, intensity),
               coors_batch: (p1 + p2 + ... + pb, 1 + 3),
               num_points_per_pillar: (p1 + p2 + ... + pb, ), (b: batch size)
        '''
        pillars, coors, npoints_per_pillar = [], [], []
        for i, pts in enumerate(batched_pts):
            voxels_out, coors_out, num_points_per_voxel_out = self.voxel_layer(pts)
            # voxels_out: (max_voxel, num_points, point coordinates), coors_out: (max_voxel, 3)
            # num_points_per_voxel_out: (max_voxel, )
            pillars.append(voxels_out)
            coors.append(coors_out.long())
            npoints_per_pillar.append(num_points_per_voxel_out)

        pillars = torch.cat(pillars, dim=0)  # (p1 + p2 + ... + pb, num_points, 3)
        npoints_per_pillar = torch.cat(npoints_per_pillar, dim=0)  # (p1 + p2 + ... + pb, )
        coors_batch = []
        for i, cur_coors in enumerate(coors):
            coors_batch.append(F.pad(cur_coors, (1, 0), value=i))
        coors_batch = torch.cat(coors_batch, dim=0)  # (p1 + p2 + ... + pb, 1 + 3)

        return pillars, coors_batch, npoints_per_pillar


class PillarEncoder(nn.Module):
    def __init__(self, voxel_size, point_cloud_range, in_channel, out_channel):
        super().__init__()
        self.out_channel = out_channel
        self.vx, self.vy = voxel_size[0], voxel_size[1]
        self.x_offset = voxel_size[0] / 2 + point_cloud_range[0]
        self.y_offset = voxel_size[1] / 2 + point_cloud_range[1]
        self.x_l = int((point_cloud_range[3] - point_cloud_range[0]) / voxel_size[0])
        self.y_l = int(np.round((point_cloud_range[4] - point_cloud_range[1]) / voxel_size[1]))
        self.conv = nn.Conv1d(in_channel, out_channel, 1, bias=False)
        self.bn = nn.BatchNorm1d(out_channel, eps=1e-3, momentum=0.01)

    def forward(self, pillars, coors_batch, npoints_per_pillar):
        """
        pillars: (p1 + p2 + ... + pb, num_points, c), c = 3
        coors_batch: (p1 + p2 + ... + pb, 1 + 3)
        npoints_per_pillar: (p1 + p2 + ... + pb, )
        return:  (bs, out_channel, y_l, x_l)
        """
        device = pillars.device
        # 1. calculate offset to the points mean (in each pillar)
        offset_pt_center = pillars[:, :, :3] - torch.sum(pillars[:, :, :3], dim=1, keepdim=True) / npoints_per_pillar[:,
                                                                                                   None,
                                                                                                   None]  # (p1 + p2 + ... + pb, num_points, 3)

        # 2. calculate offset to the pillar center
        x_offset_pi_center = pillars[:, :, :1] - (coors_batch[:, None, 1:2] * self.vx + self.x_offset)  # (p1 + p2 + ... + pb, num_points, 1)
        y_offset_pi_center = pillars[:, :, 1:2] - (
                    coors_batch[:, None, 2:3] * self.vy + self.y_offset)  # (p1 + p2 + ... + pb, num_points, 1)

        # 3. encoder
        features = torch.cat([pillars, offset_pt_center, x_offset_pi_center, y_offset_pi_center],
                             dim=-1)  # (p1 + p2 + ... + pb, num_points, 8)
        features[:, :, 0:1] = x_offset_pi_center  # tmp
        features[:, :, 1:2] = y_offset_pi_center  # tmp

        # 4. find mask for (0, 0, 0) and update the encoded features
        voxel_ids = torch.arange(0, pillars.size(1)).to(device)  # (num_points, )
        mask = voxel_ids[:, None] < npoints_per_pillar[None, :]  # (num_points, p1 + p2 + ... + pb)
        mask = mask.permute(1, 0).contiguous()  # (p1 + p2 + ... + pb, num_points)
        features *= mask[:, :, None]

        # 5. embedding
        features = features.permute(0, 2, 1).contiguous().float()  # (p1 + p2 + ... + pb, 8, num_points)
        features = F.relu(self.bn(self.conv(features)))  # (p1 + p2 + ... + pb, out_channels, num_points)
        pooling_features = torch.max(features, dim=-1)[0]  # (p1 + p2 + ... + pb, out_channels)

        # 6. pillar scatter
        batched_canvas = []
        bs = coors_batch[-1, 0] + 1
        for i in range(bs):
            cur_coors_idx = coors_batch[:, 0] == i
            cur_coors = coors_batch[cur_coors_idx, :]
            cur_features = pooling_features[cur_coors_idx]

            canvas = torch.zeros((self.x_l, self.y_l, self.out_channel), dtype=torch.float32, device=device)
            canvas[cur_coors[:, 1], cur_coors[:, 2]] = cur_features
            canvas = canvas.permute(2, 1, 0).contiguous()
            batched_canvas.append(canvas)
        batched_canvas = torch.stack(batched_canvas, dim=0)  # (bs, in_channel, self.y_l, self.x_l)
        return batched_canvas


class Backbone(nn.Module):
    def __init__(self, in_channel, out_channels, layer_nums, layer_strides=None):
        super().__init__()
        if layer_strides is None:
            layer_strides = [2, 2]
        assert len(out_channels) == len(layer_nums)
        assert len(out_channels) == len(layer_strides)

        self.multi_blocks = nn.ModuleList()
        for i in range(len(layer_strides)):
            blocks = []
            blocks.append(nn.Conv2d(in_channel, out_channels[i], 3, stride=layer_strides[i], bias=False, padding=1))
            blocks.append(nn.BatchNorm2d(out_channels[i], eps=1e-3, momentum=0.01))
            blocks.append(nn.ReLU(inplace=True))

            for _ in range(layer_nums[i]):
                blocks.append(nn.Conv2d(out_channels[i], out_channels[i], 3, bias=False, padding=1))
                blocks.append(nn.BatchNorm2d(out_channels[i], eps=1e-3, momentum=0.01))
                blocks.append(nn.ReLU(inplace=True))

            in_channel = out_channels[i]
            self.multi_blocks.append(nn.Sequential(*blocks))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        '''
        x: (b, c, y_l, x_l). Default: (6, 64, 496, 432)
        return: list[]. Default: [(6, 64, 248, 216), (6, 128, 124, 108), (6, 256, 62, 54)]
        '''
        outs = []
        for i in range(len(self.multi_blocks)):
            x = self.multi_blocks[i](x)
            outs.append(x)
        return outs


class Neck(nn.Module):
    def __init__(self, in_channels, upsample_strides, out_channels):
        super().__init__()
        assert len(in_channels) == len(upsample_strides)
        assert len(upsample_strides) == len(out_channels)

        self.decoder_blocks = nn.ModuleList()
        for i in range(len(in_channels)):
            decoder_block = []
            decoder_block.append(nn.ConvTranspose2d(in_channels[i],
                                                    out_channels[i],
                                                    upsample_strides[i],
                                                    stride=upsample_strides[i],
                                                    bias=False))
            decoder_block.append(nn.BatchNorm2d(out_channels[i], eps=1e-3, momentum=0.01))
            decoder_block.append(nn.ReLU(inplace=True))

            self.decoder_blocks.append(nn.Sequential(*decoder_block))

        # in consitent with mmdet3d
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        '''
        x: [(bs, 64, h/2, w/2), (bs, 128, h/4, w/4)]
        return: (bs, 256, h/2, w/2)
        '''
        outs = []
        for i in range(len(self.decoder_blocks)):
            xi = self.decoder_blocks[i](x[i])  # (bs, 128, 248, 216)
            outs.append(xi)
        out = torch.cat(outs, dim=1)
        return out


class Head(nn.Module):
    def __init__(self, in_channel, n_anchors, n_classes):
        super().__init__()

        self.conv_det = nn.Conv2d(in_channel, n_anchors * n_classes, 1)
        self.sigmoid = nn.Sigmoid()
        self.conv_reg = nn.Conv2d(in_channel, n_anchors * 2, 1)

        conv_layer_id = 0
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0, std=0.01)
                if conv_layer_id == 0:
                    prior_prob = 0.01
                    bias_init = float(-np.log((1 - prior_prob) / prior_prob))
                    nn.init.constant_(m.bias, bias_init)
                else:
                    nn.init.constant_(m.bias, 0)
                conv_layer_id += 1

    def forward(self, x):
        '''
        x: (bs, 256, h/2, w/2)
        return:
              det_prob_pred: (bs, n_anchors*n_classes, h/2, w/2)
              bbox_pred: (bs, n_anchors*n_classes, h/2, w/2)
        '''
        det_prob_pred = self.sigmoid(self.conv_det(x))
        bbox_pred = self.conv_reg(x)
        return det_prob_pred, bbox_pred


class PointPillarsPollo(nn.Module):
    def __init__(self,
                 nclasses=1,
                 voxel_size=(0.2, 0.2, 3),  #(0.2, 0.2, 3)
                 point_cloud_range=(-40, -40, -2, 40, 40, 1), #(0, -4.8, -2, 80, 4.8, 1)
                 max_num_points=32,
                 max_voxels=(16000, 40000)):
        super().__init__()
        self.nclasses = nclasses
        self.pillar_layer = PillarLayer(voxel_size=voxel_size,
                                        point_cloud_range=point_cloud_range,
                                        max_num_points=max_num_points,
                                        max_voxels=max_voxels)
        self.pillar_encoder = PillarEncoder(voxel_size=voxel_size,
                                            point_cloud_range=point_cloud_range,
                                            in_channel=8,
                                            out_channel=64)
        self.backbone = Backbone(in_channel=64,
                                 out_channels=[64, 128],
                                 layer_nums=[3, 5])
        self.neck = Neck(in_channels=[64, 128],
                         upsample_strides=[1, 2],
                         out_channels=[128, 128])
        self.head = Head(in_channel=256, n_anchors=1 * nclasses, n_classes=nclasses)

        # anchors
        ranges = [[-40, -40, -2, 40, 40, 1]]
        sizes = [[0.4, 0.4, 3]]
        rotations = [0]
        self.anchors_generator = AnchorsPollo(ranges=ranges,
                                         sizes=sizes,
                                         rotations=rotations)

        # train
        self.assigners = [
            {'pos_thr': 0.2, 'neg_thr': 0.6}
        ]

        # val and test
        self.max_det = 40
        self.score_thr = 0.95

    def get_predicted_bboxes_single(self, det_prob_pred, bbox_pred, anchors):
        """
        det_prob_pred: (n_anchors*nclasses, 200, 200)
        bbox_pred: (n_anchors*7, 200, 200)
        bbox_dir_cls_pred: (n_anchors*2, 200, 200)
        anchors: (y_l, x_l, 3, 2, 2)
        return:
            bboxes: (k, 2)
            labels: (k, )
            scores: (k, )
        """
        # 0. pre-process
        det_prob_pred = det_prob_pred.permute(1, 2, 0).reshape(-1, self.nclasses)
        bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 2)
        anchors = anchors.reshape(-1, 7)

        # 1. obtain self.nms_pre bboxes based on scores
        inds = det_prob_pred.max(1)[0].topk(self.max_det)[1]
        det_score = det_prob_pred.max(1)[0].topk(self.max_det)[0]
        det_mask = det_score > self.score_thr
        if det_mask.sum().item() == 0:
            result = {'lidar_bboxes': None, 'scores': None}
            return result
        inds = inds[det_mask]
        det_prob_pred = det_prob_pred[inds]
        bbox_pred = bbox_pred[inds, :]
        anchors = anchors[inds, :]

        # 2. decode predicted offsets to bboxes
        bbox_pred = anchors2bboxes(anchors, bbox_pred)

        result = {
            'lidar_bboxes': bbox_pred.detach().cpu().numpy(),
            'scores': det_prob_pred.detach().cpu().numpy()
        }
        return result

    def get_predicted_bboxes(self, det_prob_pred, bbox_pred, batched_anchors):
        '''
        det_prob_pred: (bs, n_anchors*3, 248, 216)
        bbox_pred: (bs, n_anchors*7, 248, 216)
        bbox_dir_cls_pred: (bs, n_anchors*2, 248, 216)
        batched_anchors: (bs, y_l, x_l, 3, 2, 7)
        return:
            bboxes: [(k1, 7), (k2, 7), ... ]
            labels: [(k1, ), (k2, ), ... ]
            scores: [(k1, ), (k2, ), ... ]
        '''
        results = []
        bs = det_prob_pred.size(0)
        for i in range(bs):
            result = self.get_predicted_bboxes_single(det_prob_pred=det_prob_pred[i],
                                                      bbox_pred=bbox_pred[i],
                                                      anchors=batched_anchors[i])
            results.append(result)
        return results

    def forward(self, batched_pts, mode='test', batched_gt_bboxes=None, batched_gt_labels=None):
        batch_size = len(batched_pts)
        # batched_pts: list[tensor] -> pillars: (p1 + p2 + ... + pb, num_points, c),
        #                              coors_batch: (p1 + p2 + ... + pb, 1 + 3),
        #                              num_points_per_pillar: (p1 + p2 + ... + pb, ), (b: batch size)
        pillars, coors_batch, npoints_per_pillar = self.pillar_layer(batched_pts)

        # pillars: (p1 + p2 + ... + pb, num_points, c), c = 4
        # coors_batch: (p1 + p2 + ... + pb, 1 + 3)
        # npoints_per_pillar: (p1 + p2 + ... + pb, )
        #                     -> pillar_features: (bs, out_channel, y_l, x_l)
        pillar_features = self.pillar_encoder(pillars, coors_batch, npoints_per_pillar)

        # xs:  [(bs, 64, 248, 216), (bs, 128, 124, 108), (bs, 256, 62, 54)]
        xs = self.backbone(pillar_features)

        # x: (bs, 384, 248, 216)
        x = self.neck(xs)

        # det_prob_pred: (bs, n_anchors*3, 248, 216)
        # bbox_pred: (bs, n_anchors*7, 248, 216)
        det_prob_pred, bbox_pred = self.head(x)

        # anchors
        device = det_prob_pred.device
        feature_map_size = torch.tensor(list(det_prob_pred.size()[-2:]), device=device)
        anchors = self.anchors_generator.get_multi_anchors(feature_map_size)
        batched_anchors = [anchors for _ in range(batch_size)]

        if mode == 'train':
            anchor_target_dict = anchor_target(batched_anchors=batched_anchors,
                                               batched_gt_bboxes=batched_gt_bboxes,
                                               batched_gt_labels=batched_gt_labels,
                                               assigners=self.assigners,
                                               nclasses=self.nclasses)

            return det_prob_pred, bbox_pred, anchor_target_dict
        elif mode == 'val':
            results = self.get_predicted_bboxes(det_prob_pred=det_prob_pred,
                                                bbox_pred=bbox_pred,
                                                batched_anchors=batched_anchors)
            return results

        elif mode == 'test':
            results = self.get_predicted_bboxes(det_prob_pred=det_prob_pred,
                                                bbox_pred=bbox_pred,
                                                batched_anchors=batched_anchors)
            return results
        else:
            raise ValueError
