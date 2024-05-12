#!/usr/bin/python3
#coding=utf-8
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Conv2d, Parameter, Softmax
from functools import partial

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

class GCM(nn.Module):
    def __init__(self, dim,in_dim):
        super(GCM, self).__init__()
        self.down_conv = nn.Sequential(nn.Conv2d(dim, in_dim, 3,padding=1),nn.BatchNorm2d(in_dim),
             nn.PReLU())
        down_dim = in_dim // 2

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_dim, down_dim, kernel_size=1), nn.BatchNorm2d(down_dim), nn.PReLU()
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_dim, down_dim, kernel_size=3, dilation=2, padding=2), nn.BatchNorm2d(down_dim), nn.PReLU()
        )
        

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_dim, down_dim, kernel_size=3, dilation=4, padding=4), nn.BatchNorm2d(down_dim), nn.PReLU()
        )
        

        self.conv4 = nn.Sequential(
            nn.Conv2d(in_dim, down_dim, kernel_size=3, dilation=6, padding=6), nn.BatchNorm2d(down_dim), nn.PReLU()
        )


        self.conv5 = nn.Sequential(
            nn.Conv2d(in_dim, down_dim, kernel_size=1),nn.BatchNorm2d(down_dim),  nn.PReLU()  #if batch=1 ，batchnorm will deliver bug
        )

        self.fuse = nn.Sequential(
            nn.Conv2d(5 * down_dim, in_dim, kernel_size=1), nn.BatchNorm2d(in_dim), nn.PReLU()
        )
        self.gamma = Parameter(torch.zeros(1))

        self.softmax = Softmax(dim=-1)

        """
        input  -> x:[B, D, H, W]
        output ->   [B, D, H, W]
        in_channels:    int -> Embedding Dimension
        key_channels:   int -> Key Embedding Dimension,   Best: (in_channels)
        value_channels: int -> Value Embedding Dimension, Best: (in_channels or in_channels//2)
        head_count:     int -> It divides the embedding dimension by the head_count and process each part individually
        Conv2D # of Params:  ((k_h * k_w * C_in) + 1) * C_out)
        """
        
        self.in_channels = in_channels = 640
        self.key_channels = key_channels = 640
        self.head_count = 1
        self.value_channels = value_channels = 640

        self.keys = nn.Conv2d(in_channels, key_channels, 1)
        self.queries = nn.Conv2d(in_channels, key_channels, 1)
        self.values = nn.Conv2d(in_channels, value_channels, 1)
        self.reprojection = nn.Conv2d(value_channels, in_channels, 1)

    def initialize(self):
        weight_init(self)

    def forward(self, x):
        x = self.down_conv(x)

        conv1 = self.conv1(x)
        conv2 = self.conv2(x)
        conv3 = self.conv3(x)
        conv4 = self.conv4(x)
        conv5 = F.upsample(self.conv5(F.adaptive_avg_pool2d(x, 1)), size=x.size()[2:], mode='bilinear', align_corners=True) #if batch=1 ，batchnorm will deliver bug

        y = self.fuse(torch.cat((conv1, conv2, conv3, conv4, conv5), 1))

        n, _, h, w = y.size()

        keys = self.keys(y).reshape((n, self.key_channels, h * w))
        queries = self.queries(y).reshape(n, self.key_channels, h * w)
        values = self.values(y).reshape((n, self.value_channels, h * w))

        head_key_channels = self.key_channels // self.head_count
        head_value_channels = self.value_channels // self.head_count

        attended_values = []
        for i in range(self.head_count):
            key = F.softmax(keys[:, i * head_key_channels : (i + 1) * head_key_channels, :], dim=2)

            query = F.softmax(queries[:, i * head_key_channels : (i + 1) * head_key_channels, :], dim=1)

            value = values[:, i * head_value_channels : (i + 1) * head_value_channels, :]

            context = key @ value.transpose(1, 2)  # dk*dv
            attended_value = (context.transpose(1, 2) @ query).reshape(n, head_value_channels, h, w)  # n*dv
            attended_values.append(attended_value)

        aggregated_values = torch.cat(attended_values, dim=1)
        attention = self.reprojection(aggregated_values)
        
        """
        residual learning
        """
        y = self.gamma * attention + y
        return y
