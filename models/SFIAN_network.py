import torch
import torch.nn as nn
import torch.nn.functional as F
from .networks import get_norm_layer, init_net
from .defConv import DeformConv2d

class FuseGFFConvBlock123(nn.Module):
    def __init__(self, inputs, n_filters=128, kernel=3, stride=1):
        super(FuseGFFConvBlock123, self).__init__()
        self.FuseGFFConvBlock123 = nn.Sequential(
            DeformConv2d(inputs, n_filters),
            nn.BatchNorm2d(n_filters),
        )
    def forward(self,x):
        return self.FuseGFFConvBlock123(x)

class FuseGFFConvBlock45(nn.Module):
    def __init__(self,inputs, n_filters = 128, kernel_size = 3, stride=1,padding = 1):
        super(FuseGFFConvBlock45, self).__init__()
        self.FuseGFFConvBlock45 = nn.Sequential(
            nn.Conv2d(inputs, n_filters, kernel_size, stride=stride,padding = padding),
            nn.BatchNorm2d(n_filters),
        )
    def forward(self,x):
        return self.FuseGFFConvBlock45(x)


class SFIANNet(nn.Module):
    def __init__(self, in_nc= 3, num_classes = 1, ngf=16,  norm='batch'):
        super(SFIANNet, self).__init__()
        self.conv_1 = nn.Conv2d(ngf, 16, 1)
        self.conv_2 = nn.Conv2d(ngf * 2,16, 1)
        self.conv_3 = nn.Conv2d(ngf * 4,16, 1)
        self.conv_4 = nn.Conv2d(ngf * 8,16, 1)
        self.conv_5 = nn.Conv2d(ngf * 8,16, 1)

        norm_layer = get_norm_layer(norm_type=norm)
        self.conv1 = nn.Sequential(*self._conv_block(in_nc, ngf, norm_layer, num_block=2, flag=False))
        self.side_conv1 = nn.Conv2d(ngf, num_classes, kernel_size=1, padding=0, stride=1, bias=False)

        self.conv2 = nn.Sequential(*self._conv_block(ngf, ngf * 2, norm_layer, num_block=2, flag=False))
        self.side_conv2 = nn.Conv2d(ngf * 2, num_classes, kernel_size=1,  padding=0, stride=1, bias=False)

        self.conv3 = nn.Sequential(*self._conv_block(ngf * 2, ngf * 4, norm_layer, num_block=3, flag=False))
        self.side_conv3 = nn.Conv2d(ngf * 4, num_classes, kernel_size=1,padding=0, stride=1, bias=False)

        self.conv4 = nn.Sequential(*self._conv_block(ngf * 4, ngf * 8, norm_layer, num_block=3, flag=False))
        self.side_conv4 = nn.Conv2d(ngf * 8, num_classes, kernel_size=1, padding=0, stride=1, bias=False)

        self.conv5 = nn.Sequential(*self._conv_block(ngf * 8, ngf * 8, norm_layer, num_block=3, flag=False))
        self.side_conv5 = nn.Conv2d(ngf * 16, num_classes, kernel_size=1, padding=0, stride=1, bias=False)

        self.FuseGFFConvBlock123 = FuseGFFConvBlock123(16, 1, 3)
        self.FuseGFFConvBlock45 = FuseGFFConvBlock45(16, 1, 3)

        self.fuse_conv = nn.Conv2d(num_classes * 5, num_classes, kernel_size=1, padding=0, stride=1, bias=False)

        self.maxpool = nn.MaxPool2d(2, stride=2)


    def _conv_block(self, in_nc, out_nc, norm_layer, num_block=2, kernel_size=3,
                    stride=1, padding=1, bias=False,flag = False):
        if flag:
            conv = []
            for i in range(num_block):
                cur_in_nc = in_nc if i == 0 else out_nc
                if i == num_block-1:
                    conv += [DeformConv2d(cur_in_nc, out_nc, kernel_size=kernel_size, padding=padding, stride=stride, bias=bias),
                                 norm_layer(out_nc),
                                 nn.ReLU(True)]
                else:
                    conv += [nn.Conv2d(cur_in_nc, out_nc, kernel_size=kernel_size, stride=stride,
                                       padding=padding, bias=bias),
                             norm_layer(out_nc),
                         nn.ReLU(True)]
        else:
            conv = []
            for i in range(num_block):
                cur_in_nc = in_nc if i == 0 else out_nc
                conv += [nn.Conv2d(cur_in_nc, out_nc, kernel_size=kernel_size, stride=stride,
                                   padding=padding, bias=bias),
                         norm_layer(out_nc),
                     nn.ReLU(True)]
        return conv

    def forward(self, x):

        h, w = x.size()[2:]

        x1 = self.conv1(x)
        x1n = self.conv_1(x1)
        x1n = F.interpolate(x1n, size=(h, w), mode='bilinear',
                            align_corners=True)
        g1 = torch.sigmoid(x1n)

        x2 = self.conv2(self.maxpool(x1))
        x2n = self.conv_2(x2)
        x2n = F.interpolate(x2n, size=(h, w), mode='bilinear',
                            align_corners=True)
        g2 = torch.sigmoid(x2n)

        x3 = self.conv3(self.maxpool(x2))
        x3n = self.conv_3(x3)
        x3n = F.interpolate(x3n, size=(h, w), mode='bilinear',
                            align_corners=True)
        g3 = torch.sigmoid(x3n)

        x4 = self.conv4(self.maxpool(x3))
        x4n = self.conv_4(x4)
        x4n = F.interpolate(x4n, size=(h, w), mode='bilinear',
                            align_corners=True)
        g4 = torch.sigmoid(x4n)

        x5 = self.conv5(self.maxpool(x4))
        x5n = self.conv_5(x5)
        x5n = F.interpolate(x5n, size=(h, w), mode='bilinear',
                            align_corners=True)

        g5 = torch.sigmoid(x5n)


        x1gff = (1 + g1) * x1n + (1 - g1) * (g2 * x2n + g3 * x3n + g4 * x4n + g5 * x5n)
        x2gff = (1 + g2) * x2n + (1 - g2) * (g1 * x1n + g3 * x3n + g4 * x4n + g5 * x5n)
        x3gff = (1 + g3) * x3n + (1 - g3) * (g2 * x2n + g1 * x1n + g4 * x4n + g5 * x5n)
        x4gff = (1 + g4) * x4n + (1 - g4) * (g2 * x2n + g3 * x3n + g1 * x1n + g5 * x5n)
        x5gff = (1 + g5) * x4n + (1 - g5) * (g2 * x2n + g3 * x3n + g1 * x1n + g4 * x4n)

        x1gff = self.FuseGFFConvBlock123(x1gff)
        x2gff = self.FuseGFFConvBlock123(x2gff)
        x3gff = self.FuseGFFConvBlock123(x3gff)
        x4gff = self.FuseGFFConvBlock123(x4gff)
        x5gff = self.FuseGFFConvBlock123(x5gff)


        fused = self.fuse_conv(torch.cat([x1gff,
                                          x2gff,
                                          x3gff,
                                          x4gff,
                                          x5gff], dim=1))

        return x1gff, x2gff, x3gff, x4gff, x5gff, fused



def define_SFIAN(in_nc,
                     num_classes,
                     ngf,
                     norm='batch',
                     init_type='xavier',
                     init_gain=0.02,
                     gpu_ids=[]):
    net = SFIANNet(in_nc, num_classes, ngf, norm)
    return init_net(net, init_type, init_gain, gpu_ids)


class BinaryFocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, logits=False, size_average=True):
        super(BinaryFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.size_average = size_average
        self.criterion = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, inputs, targets):
        BCE_loss = self.criterion(inputs, targets)
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss

        if self.size_average:
            return F_loss.mean()
        else:
            return F_loss.sum()
