import logging
from matplotlib.pyplot import axis
import torch
import torch.nn as nn
import torchvision.models as models
import sys
import numpy as np
import os
from torchvision import models
import segmentation_models_pytorch as smp
sys.path.append("./../")
import albumentations as A
from utils import multiPoolPrepare, multiMaxPooling, unwrapPrepare, unwrapPool
import random


class BaseNet(nn.Module):
    def __init__(self, latent_dim, first_channel=4):
        super(BaseNet, self).__init__()
        torch.manual_seed(10)
        torch.cuda.manual_seed(10)
        self.conv1 = nn.Conv2d(first_channel, 8, kernel_size=3, stride=1)
        self.act1 = nn.ReLU()
        self.max_pool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, stride=1)
        self.act2 = nn.ReLU()
        self.max_pool2 = nn.MaxPool2d(kernel_size=2)
        self.conv3 = nn.Conv2d(16, 128, kernel_size=2, stride=1)
        self.act3 = nn.ReLU()
        self.conv4 = nn.Conv2d(128, latent_dim, kernel_size=1)

    def forward(self, x):
        torch.manual_seed(10)
        torch.cuda.manual_seed(10)
        x = self.conv1(x)
        x = self.act1(x)
        x = self.max_pool1(x)
        torch.manual_seed(10)
        torch.cuda.manual_seed(10)
        x = self.conv2(x)
        x = self.act2(x)
        x = self.max_pool2(x)
        torch.manual_seed(10)
        torch.cuda.manual_seed(10)
        x = self.conv3(x)
        x = self.act3(x)
        torch.manual_seed(10)
        torch.cuda.manual_seed(10)
        y = self.conv4(x)
        return y
        

class SlimNet(nn.Module):
    def __init__(self, pH=15, pW=15, sL1=2, sL2=2, imH=256, imW=256, latent_dim=100, first_channel=4):
        base_net = BaseNet(latent_dim, first_channel)
        super(SlimNet, self).__init__()
        imW = int(np.ceil(imW / (sL1 * sL2)) * sL1 * sL2)
        imH = int(np.ceil(imH / (sL1 * sL2)) * sL1 * sL2)
        self.imH = imH
        self.imW = imW
        self.multiPoolPrepare = multiPoolPrepare(pH, pW)
        self.conv1 = list(base_net.modules())[1]
        self.act1 = list(base_net.modules())[2]
        self.multiMaxPooling1 = multiMaxPooling(sL1, sL1, sL1, sL1)
        self.conv2 = list(base_net.modules())[4]
        self.act2 = list(base_net.modules())[5]
        self.multiMaxPooling2 = multiMaxPooling(sL2, sL2, sL2, sL2)
        self.conv3 = list(base_net.modules())[7]
        self.act3 = list(base_net.modules())[8]
        self.conv4 = list(base_net.modules())[9]
        self.outChans = list(base_net.modules())[9].out_channels
        self.unwrapPrepare = unwrapPrepare()
        self.unwrapPool2 = unwrapPool(self.outChans, imH / (sL1 * sL2), imW / (sL1 * sL2), sL2, sL2)
        self.unwrapPool3 = unwrapPool(self.outChans, imH / sL1, imW / sL1, sL1, sL1)
        

    def forward(self, x):
        bs = x.shape[0]
        x = self.multiPoolPrepare(x)
        torch.manual_seed(10)
        torch.cuda.manual_seed(10)
        x = self.conv1(x)
        x = self.act1(x)
        x = self.multiMaxPooling1(x)

        torch.manual_seed(10)
        torch.cuda.manual_seed(10)
        x = self.conv2(x)
        x = self.act2(x)
        x = self.multiMaxPooling2(x)
        torch.manual_seed(10)
        torch.cuda.manual_seed(10)
        x = self.conv3(x)
        x = self.act3(x)
        x = self.conv4(x)
        x = self.unwrapPrepare(x)
        x = self.unwrapPool2(x)
        x = self.unwrapPool3(x)
        y = x.view(bs,-1, self.imH, self.imW).squeeze(0)

        return y


class ASD_Encoder(nn.Module):
    def __init__(self, pH=15, pW=15, sL1=2, sL2=2, imH=256, imW=256, scales='0.5, 1.0, 1.5', class_number = 3, latent_dim=100, device='cpu', first_channel=4) -> None:
        super().__init__()
        self.models = []
        self.scales = list(map(float, scales.split(', ')))
        logging.info(self.scales)
        self.imW = imW
        self.imH = imH
        self.scale_num = len(self.scales)
        for scale in self.scales:
            imH_t = imH * scale
            imW_t = imW * scale
            model = SlimNet(pH, pW, sL1, sL2, imH_t, imW_t, latent_dim, first_channel).to(device)
            self.models.append(model)
        
        self.bn = nn.BatchNorm2d(latent_dim * self.scale_num)
        self.conv1 = nn.Conv2d(latent_dim * self.scale_num, latent_dim, kernel_size=1, stride=1, padding='same')

        self.conv2 = nn.Conv2d(latent_dim* self.scale_num, first_channel, kernel_size=1, stride=1, padding='same')

        self.conv3 = nn.Conv2d(latent_dim * self.scale_num, class_number, kernel_size=1, stride=1, padding='same')
        self.softmax = torch.nn.Softmax(dim=1)
        self.latent_dim = latent_dim

        self._initialize_weights()

    def forward(self, x):
        self.multi_scale_outputs = []
        for i in range(self.scale_num):
            scale = self.scales[i]
            imH_t = int(self.imH * scale)
            imW_t = int(self.imW * scale)
            out = self.models[i](nn.functional.interpolate(x, size=(imW_t, imH_t))).unsqueeze(0)
            out = nn.functional.interpolate(out, size=(self.imW, self.imH))
            self.multi_scale_outputs.append(out)

        combine_multi_scale = torch.cat(self.multi_scale_outputs, axis=1)
        x2 = self.bn(combine_multi_scale)

        svdd_features = self.conv1(x2)
 
        recons_map = self.conv2(x2)

        return svdd_features, recons_map
        
    def _initialize_weights(self, mode='fan_in'):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode=mode, nonlinearity='relu')
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1) 
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()