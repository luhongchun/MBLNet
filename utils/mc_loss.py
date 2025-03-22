import torch.nn as nn
import torch
import numpy as np
import random
from torch.autograd import Variable
from torch.nn.modules.module import Module
from torch.nn.modules.utils import _single, _pair, _triple
import torch.nn.functional as F
criterion = nn.CrossEntropyLoss()
#CCMP
class my_MaxPool2d(Module):

    def __init__(self, kernel_size, stride=None, padding=0, dilation=1,
                 return_indices=False, ceil_mode=False):
        super(my_MaxPool2d, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride or kernel_size
        self.padding = padding
        self.dilation = dilation
        self.return_indices = return_indices
        self.ceil_mode = ceil_mode

    def forward(self, input):
        input = input.transpose(3,1)
        input = F.max_pool2d(input, self.kernel_size, self.stride,
                            self.padding, self.dilation, self.ceil_mode,
                            self.return_indices)
        input = input.transpose(3,1).contiguous()
        return input
def Mask(nb_batch, channels):
    foo = [1] * 2 + [0] *  1
    bar = []
    for i in range(100):
        random.shuffle(foo)
        bar += foo
    bar = [bar for i in range(nb_batch)]
    bar = np.array(bar).astype("float32")
    bar = bar.reshape(nb_batch, 100 * channels, 1, 1)
    bar = torch.from_numpy(bar)
    bar = bar.cuda()
    bar = Variable(bar)
    return bar
def Mask1(nb_batch, channels):
    foo = [1] * 2 + [0] * 2
    bar = []
    for i in range(128):#2048/4
        random.shuffle(foo)
        bar += foo
    bar = [bar for i in range(nb_batch)]
    bar = np.array(bar).astype("float32")
    bar = bar.reshape(nb_batch, 128 * channels, 1, 1)
    bar = torch.from_numpy(bar)
    bar = bar.cuda()
    bar = Variable(bar)
    return bar
def Mask2(nb_batch, channels):
    foo = [1] * 1 + [0] * 1
    bar = []
    for i in range(256):
        random.shuffle(foo)
        bar += foo
    bar = [bar for i in range(nb_batch)]
    bar = np.array(bar).astype("float32")
    bar = bar.reshape(nb_batch, 256 * channels, 1, 1)
    bar = torch.from_numpy(bar)
    bar = bar.cuda()
    bar = Variable(bar)
    return bar

def supervisor(x, targets, height, cnum):
    if cnum == 2:
        mask = Mask2(x.size(0), cnum)
    else:
        mask = Mask1(x.size(0), cnum)
    branch = x
    branch = branch.reshape(branch.size(0),branch.size(1), branch.size(2) * branch.size(3))
    branch = F.softmax(branch, 2)
    branch = branch.reshape(branch.size(0), branch.size(1), x.size(2), x.size(2))
    branch = my_MaxPool2d(kernel_size=(1, cnum), stride=(1, cnum))(branch)
    branch = branch.reshape(branch.size(0),branch.size(1), branch.size(2) * branch.size(3))
    loss_2 = 1.0 - 1.0 * torch.mean(torch.sum(branch, 2)) / cnum# set margin = 3.0
    branch_1 = x * mask
    branch_1 = my_MaxPool2d(kernel_size=(1,cnum), stride=(1,cnum))(branch_1)
    branch_1 = nn.AvgPool2d(kernel_size=(height,height))(branch_1)
    branch_1 = branch_1.view(branch_1.size(0), -1)
    loss_1 = criterion(branch_1, targets)
    return [loss_1, loss_2]