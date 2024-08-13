import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F


class LossPollo(nn.Module):
    def __init__(self, beta=1 / 9, det_w=1.0, reg_w=20.0):
        super().__init__()
        self.det_w = det_w
        self.reg_w = reg_w
        self.smooth_l1_loss = nn.SmoothL1Loss(reduction='none',
                                              beta=beta)
        self.bce_loss = nn.BCELoss()

    def forward(self,
                det_prob_pred,
                loc_pred,
                batched_obj_presence,
                batched_loc_reg):
        '''
        det_prob_pred = (n, )
        loc_pred = (n,2)
        batched_obj_presence = (n, )
        batched_loc_reg = (n, 2)
        return: loss, float.
        '''
        # 1. detection probability loss
        det_loss = self.bce_loss(det_prob_pred, batched_obj_presence)

        # 2. regression loss
        loc_loss = self.smooth_l1_loss(loc_pred, batched_loc_reg).mean()
        loc_loss = (loc_loss * batched_obj_presence.unsqueeze(-1)).sum() / (batched_obj_presence.sum() + 1e-6)

        # 4. total loss
        total_loss = self.det_w * det_loss + self.reg_w * loc_loss

        loss_dict = {'det_loss': det_loss,
                     'loc_loss': loc_loss,
                     'total_loss': total_loss}
        return loss_dict

