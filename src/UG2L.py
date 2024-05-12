
#!/usr/bin/python3
#coding=utf-8
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.GCM import GCM
from src.pyramid_vig import pvig_s_224_gelu

def weight_init(module):
    for n, m in module.named_children():
        try:
            #print('initialize: '+n)
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d)):
                nn.init.ones_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Sequential):
                weight_init(m)
            elif isinstance(m, (nn.ReLU,nn.PReLU, nn.Unfold, nn.Sigmoid, nn.AdaptiveAvgPool2d,nn.AvgPool2d, nn.Softmax,nn.Dropout2d)):
                pass
            else:
                m.initialize()
        except:
            pass

class Bottleneck(nn.Module):
    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1):
        super(Bottleneck, self).__init__()
        self.conv1      = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1        = nn.BatchNorm2d(planes)
        self.conv2      = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=(3*dilation-1)//2, bias=False, dilation=dilation)
        self.bn2        = nn.BatchNorm2d(planes)
        self.conv3      = nn.Conv2d(planes, planes*4, kernel_size=1, bias=False)
        self.bn3        = nn.BatchNorm2d(planes*4)
        self.downsample = downsample

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = F.relu(self.bn2(self.conv2(out)), inplace=True)
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            x = self.downsample(x)
        return F.relu(out+x, inplace=True)


class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.conv1    = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1      = nn.BatchNorm2d(64)
        self.layer1   = self.make_layer( 64, 3, stride=1, dilation=1)
        self.layer2   = self.make_layer(128, 4, stride=2, dilation=1)
        self.layer3   = self.make_layer(256, 6, stride=4, dilation=2)
        self.DGCN =  pvig_s_224_gelu(in_channels = 512)

        self.Global   = GCM(dim=1024, in_dim=640)


    def make_layer(self, planes, blocks, stride, dilation):
        downsample    = nn.Sequential(nn.Conv2d(self.inplanes, planes*4, kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(planes*4))
        layers        = [Bottleneck(self.inplanes, planes, stride, downsample, dilation=dilation)]
        self.inplanes = planes*4
        for _ in range(1, blocks):
            layers.append(Bottleneck(self.inplanes, planes, dilation=dilation))
        return nn.Sequential(*layers)
        
    def forward(self, x):
        out1 = F.relu(self.bn1(self.conv1(x)), inplace=True) ##64 x 224 x 224
        out2 = F.max_pool2d(out1, kernel_size=3, stride=2, padding=1)  
        out2 = self.layer1(out2)  #256 x 112 x 112
        out3 = self.layer2(out2)###512 channels#512 x 56 x 56
        out4, out5 = self.DGCN(out3) # 8x8 patch
        out6 = self.Global(self.layer3(out3))  #torch.Size([2, 256, 14, 14])
        #out3 = self.layer3(out3)
        return out1, out2, out3, out4, out5, out6

    def initialize(self):
        self.load_state_dict(torch.load('/data/iopen/lyf/SaliencyOD_in_RSIs/zhuanli/resnet50-19c8e357.pth'), strict=False)
        print("init ResNet")
        """
        load DGN's weight
        """
        weights2 = {}
        DGCN_weights = torch.load("./src/pvig_s_82.1.pth.tar")
        for k,v in DGCN_weights.items():
            if "backbone.6" in k:
                k = "backbone.1" + k[10:]
                """
                size mismatch for backbone.1.0.relative_pos: 
                """
                if "backbone.1.0.relative_pos" not in k:
                    weights2.update({k:v})
            elif "backbone.7" in k:
                k = "backbone.2" + k[10:]
                """
                size mismatch for backbone.2.0.relative_pos: 
                """
                if "backbone.2.0.relative_pos" not in k:
                    weights2.update({k:v})
            elif "backbone.8" in k:
                k = "backbone.3" + k[10:]
                """
                size mismatch for backbone.3.0.relative_pos: 
                """
                if "backbone.3.0.relative_pos" not in k:
                    weights2.update({k:v})
            elif "backbone.9" in k:
                k = "backbone.4" + k[10:]
                """
                size mismatch for backbone.4.0.relative_pos: 
                """
                if "backbone.4.0.relative_pos" not in k:
                    weights2.update({k:v})
            elif "backbone.10" in k:
                k = "backbone.5" + k[10:]
                weights2.update({k:v})
            elif "backbone.11" in k:
                k = "backbone.6" + k[10:]
                weights2.update({k:v})
            elif "backbone.12" in k:
                k = "backbone.7" + k[10:]
                weights2.update({k:v})
            elif "backbone.13" in k:
                k = "backbone.8" + k[10:]
                weights2.update({k:v})
            elif "backbone.14" in k:
                k = "backbone.9" + k[10:]
                weights2.update({k:v})
        self.DGCN.load_state_dict(weights2, False) 
        print("init DGCN")


class SOD_Head(nn.Module):
    def __init__(self,):
        super(SOD_Head, self).__init__()
        self.process = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 1, kernel_size = 3, padding = 1)
        )
        self.initialize()

    def forward(self, x):
        smaps = self.process(x)        
        return smaps

    def initialize(self):
        weight_init(self)

class FuseBlock(nn.Module):
    def __init__(self, in_channel1, in_channel2):
        super(FuseBlock, self).__init__()
        self.in_channel1 = in_channel1
        self.in_channel2 = in_channel2
        self.fuse = nn.Conv2d(self.in_channel1 + self.in_channel2, 256, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(256)
       
        self.initialize()
    def forward(self, x, y):
        out = F.relu(self.bn1(self.fuse(torch.cat((x,y), dim = 1))))
        return out

    def initialize(self):
        weight_init(self)


class net(nn.Module):
    def __init__(self):
        super(net, self).__init__()
        
        self.bkbone  = ResNet()
        self.fuse5 = FuseBlock(in_channel1 = 640,  in_channel2 = 640)
        self.fuse4 = FuseBlock(in_channel1 = 400,  in_channel2 = 256)
        self.fuse3 = FuseBlock(in_channel1 = 512,  in_channel2 = 256)
        self.fuse2 = FuseBlock(in_channel1 = 256,  in_channel2 = 256)
        self.fuse1 = FuseBlock(in_channel1 = 64,  in_channel2 = 256)

        self.SOD_head1 = SOD_Head()
        self.SOD_head2 = SOD_Head()
        self.SOD_head3 = SOD_Head()
        self.SOD_head4 = SOD_Head()
        self.SOD_head5 = SOD_Head()
        

        self.initialize()

    def forward(self, x):
        """
        baseline operations
        """
        s1,s2,s3,s4,s5,s6 = self.bkbone(x)
        #return s3, s4, s5, s6
        out5 =  self.fuse5(s5, F.interpolate(s6, size = s5.size()[2:], mode='bilinear',align_corners=True))

        out4 =  self.fuse4(s4, F.interpolate(out5, size = s4.size()[2:], mode='bilinear',align_corners=True))

        out3 =  self.fuse3(s3, F.interpolate(out4, size = s3.size()[2:], mode='bilinear',align_corners=True))

        out2  = self.fuse2(s2, F.interpolate(out3, size = s2.size()[2:], mode='bilinear',align_corners=True))

        out1  = self.fuse1(s1, F.interpolate(out2, size = s1.size()[2:], mode='bilinear',align_corners=True))
        
        if self.training:
            smap1 = self.SOD_head1(out1)
            smap2 = self.SOD_head2(out2)
            smap3 = self.SOD_head3(out3)
            smap4 = self.SOD_head4(out4)
            smap5 = self.SOD_head5(out5)
        
            ### interpolate
            smap1 = F.interpolate(smap1, size = x.size()[2:], mode='bilinear',align_corners=True)
            smap2 = F.interpolate(smap2, size = x.size()[2:], mode='bilinear',align_corners=True)
            smap3 = F.interpolate(smap3, size = x.size()[2:], mode='bilinear',align_corners=True)
            smap4 = F.interpolate(smap4, size = x.size()[2:], mode='bilinear',align_corners=True)
            smap5 = F.interpolate(smap5, size = x.size()[2:], mode='bilinear',align_corners=True)

        
            return smap1, smap2, smap3, smap4, smap5
        
        else:
            smap1 = self.SOD_head1(out1)
            smap1 = F.interpolate(smap1, size = x.size()[2:], mode='bilinear',align_corners=True)
            
            return torch.sigmoid(smap1)
        

    def initialize(self):
        weight_init(self)


if __name__ == "__main__":
    import thop
    model = net()
    x = torch.Tensor(1, 3, 448, 448)
    flops, params = thop.profile(model,inputs=(x,))
    flops, params = thop.clever_format([flops, params], "%.3f")
    print(flops, params) #102.526G 57.175M