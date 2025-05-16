import torch.nn as nn
from torchvision import models
import torch
from utils.mc_loss import supervisor
import torch.nn.init as init
from pretrained.networks import resnet

class MainNet(nn.Module):
    def __init__(self, num_classes, channels):
        super(MainNet, self).__init__()
        self.num_classes = num_classes
        self.pretrained_model = resnet.resnet18(pretrained=True, pth_path=pretrain_path)
        self.rawcls_net = nn.Linear(channels, num_classes)

    def forward(self, x):
        fm, embeding = self.pretrained_model(x)
        raw_logits = self.rawcls_net(embeding)
        return raw_logits

class ResNet(nn.Module):
    def __init__(self, hash_bit, res_model="ResNet18"):
        super(ResNet, self).__init__()
        model_resnet = resnet_dict[res_model](pretrained=True)
        self.conv1 = model_resnet.conv1
        self.bn1 = model_resnet.bn1
        self.relu = model_resnet.relu
        self.maxpool = model_resnet.maxpool
        self.layer1 = model_resnet.layer1
        self.layer2 = model_resnet.layer2
        self.layer3 = model_resnet.layer3
        self.layer4 = model_resnet.layer4
        self.avgpool = model_resnet.avgpool
        self.feature_layers = nn.Sequential(self.conv1, self.bn1, self.relu, self.maxpool, \
                                            self.layer1, self.layer2, self.layer3, self.layer4, self.avgpool)

        self.hash_layer = nn.Linear(model_resnet.fc.in_features, hash_bit)
        self.hash_layer.weight.data.normal_(0, 0.01)
        self.hash_layer.bias.data.fill_(0.0)

    def forward(self, x):
        x = self.feature_layers(x)
        x = x.view(x.size(0), -1)
        y = self.hash_layer(x)
        return y
#MBLNet
class ResNet18_MBLNet(nn.Module):
    def __init__(self, bits, classes, class_mask_rate):
        super(ResNet18_MBLNet, self).__init__()
        self.model = models.resnet18(pretrained=True)
        # del self.model.fc
        # del self.model.avgpool
        self.model.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.model.fc = nn.Linear(512, classes)
        self.model.b = nn.Linear(classes, bits)
        self.class_mask_rate = class_mask_rate

    def forward(self, x, targets=None):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        border = x
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        fm = x
        # ------------------------channel filtering------------------------------------
        bz, c, w, h = x.shape[0:]
        nonzeros = (x != 0).float().sum(dim=(2, 3)) / 1.0 / (w * h) + 1e-6
        # nonzeros_mean = torch.mean(nonzeros, dim=1, keepdim=True)
        nonzeros_mean = torch.ones([bz, 1]).detach().cuda() * 0.2
        Mask = (nonzeros > nonzeros_mean).float().detach()
        x = x * Mask.unsqueeze(-1).unsqueeze(-1)
        # ------------------------------------------------------------
        #---------spatial erase------------------------------
        A = torch.sum(fm.detach(), dim=1, keepdim=True)
        a = torch.mean(A, dim=[2, 3], keepdim=True)
        M = (A > a).float().detach() + (A < a).float().detach() * 0.5
        x = x * M

        if self.training:
            # MC-Loss
            MC_loss = supervisor(fm, targets, height=7, cnum=2)#Note that cnum=2 for datasets cars and cub, and cnum=4 for air, or else an error classification mismatch is reported

        x = self.model.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.model.fc(x)
        #-----hash-out-----100 channels represent 100 classes -> take the maximum value of each channel as 1 and assign 0.7 to the rest,, generate the label x_mask,, and multiply by x一-------
        x_mask = torch.ones(x.size()).detach().cuda() * self.class_mask_rate#0.7
        x_mask_hanmin = torch.zeros(x.size()).detach().cuda()
        for i in range(x_mask.size()[0]):
            x_mask[i, torch.argmax(x[i])] = 1
            x_mask_hanmin[i, torch.argmax(x[i])] = 1

        x_b = x * x_mask
        b = self.model.b(x_b)
        if self.training:
            return fm, x, b, border, MC_loss, x_mask_hanmin
        else:
            return fm, x, b, border