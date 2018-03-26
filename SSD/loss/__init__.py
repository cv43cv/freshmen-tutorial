import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from .common import *
from .focal_loss import *

def my_loss(loss_name, num_classes, **kwargs):
    if loss_name == 'CE_loss_hnm':
        return CE_loss_hnm(num_classes, **kwargs)
    elif loss_name == 'focal_loss':
        return focal_loss(num_classes, **kwargs)
    else:
        raise ValueError('loss not implemented')

class CE_loss_hnm(nn.Module):
    def __init__(self, num_classes, negpos_ratio=3):
        super(CE_loss_hnm, self).__init__()
        self.num_classes = num_classes
        self.negpos_ratio = negpos_ratio
        
    def forward(self, x, y):
        """
        Args:
            x : tuple of tensors (x_loc, x_conf)
            y : tensor shape(N, num_obj, 5)
    
        """
        x_loc, x_conf = x
        N = x_loc.size(0)
        C = self.num_classes

        y = Variable(match_db(y.data))

        pos = y[:,:,0] > 0

        num_pos = pos.long().sum(dim=1, keepdim=True)

        if num_pos.sum().data[0] == 0:
            return Variable(torch.FloatTensor([0])), Variable(torch.FloatTensor([0]))
        
        #Localization Loss
        pos_idx = pos.unsqueeze(2).expand_as(x_loc)
        x_loc = x_loc[pos_idx].view(-1, 4)
        y_loc = y[:,:,1:][pos_idx].view(-1,4)
        loss_loc = F.smooth_l1_loss(x_loc, y_loc, size_average=False)


        #Confidence Loss
        l = softmax_loss(x_conf.contiguous().view(-1, self.num_classes), y[:,:,0].long().view(-1)).view(N, -1)
        
        l[pos] = 0

        _, loss_idx = l.sort(1, descending=True)
        _, idx_rank = loss_idx.sort(1)
        num_neg = torch.clamp(self.negpos_ratio*num_pos, max=pos.size(1)-1)
        neg = idx_rank < num_neg
        pos_idx = pos.unsqueeze(2).expand_as(x_conf)
        neg_idx = neg.unsqueeze(2).expand_as(x_conf)
        x_conf = x_conf[(pos_idx + neg_idx).gt(0)].view(-1, self.num_classes)
        y_conf = y[:,:,0][(pos + neg).gt(0)].long()
        loss_conf = F.cross_entropy(x_conf, y_conf, size_average=False)

        N = num_pos.sum().float()
        loss_loc /= N
        loss_conf /= N

        return loss_loc, loss_conf