import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from math import exp

def _pearson(x, y, size_average = True):
    x = x.view(-1, x.size(1) * x.size(2) * x.size(3))
    y = y.view(-1, y.size(1) * y.size(2) * y.size(3))

    mean_x = torch.mean(x)
    mean_y = torch.mean(y)
    std_x = torch.std(x)
    std_y = torch.std(y)
    xm = x.sub(mean_x)
    ym = y.sub(mean_y)
    r_num = xm.mul(ym)
    mean_r_num = torch.mean(r_num)
    #r_den = torch.norm(xm, 2) * torch.norm(ym, 2)
    r_val = mean_r_num / (std_x * std_y)

    if size_average:
        return r_val.mean()
    else:
        return r_val.mean(1).mean(1).mean(1)

class PEARSON(torch.nn.Module):
    def __init__(self, size_average = True):
        super(PEARSON, self).__init__()
        self.size_average = size_average

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        return _pearson(img1, img2, self.size_average)

def pearson(img1, img2, size_average = True):
    (_, channel, _, _) = img1.size()

    return _pearson(img1, img2, size_average)
