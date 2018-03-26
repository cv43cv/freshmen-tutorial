import torch
import torch.nn  as nn
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp

from loss.common import *

class focal_loss(nn.Module):
    def __init__(self, num_classes, gamma=2):
        super(focal_loss, self).__init__()
        self.num_classes = num_classes
        self.gamma=gamma
        
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
        l = F.softmax(x_conf, dim=2)


        l = torch.clamp(l, min = exp(-100))

        l = ((1-l)**self.gamma) * torch.log(l)

        

        y_conf = y[:,:,0].long()
        loss_conf = F.nll_loss(l.view(-1,C), y_conf.view(-1), size_average=False)

        N = num_pos.sum().float()
        loss_loc /= N

        return loss_loc, loss_conf