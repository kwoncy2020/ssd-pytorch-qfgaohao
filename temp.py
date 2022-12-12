
import torch
from torch.nn import Conv2d, Sequential, ModuleList, BatchNorm2d
from torch import nn
from vision.nn.mobilenetv3 import MobileNetV3_Large, MobileNetV3_Small, Block, hswish
from vision.nn.mobilenet import MobileNetV1

from vision.ssd.ssd import SSD
from vision.ssd.predictor import Predictor
from vision.ssd.config import mobilenetv1_ssd_config as config
from pytorch_model_summary import summary

from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights


if __name__ == '__main__':
    weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1
    
    model = mobilenet_v3_small(weights=weights)
    print(summary(model, torch.randn(1,3,224,224)))

    # base_net = MobileNetV3_Small()
    # base_net = MobileNetV1()
    # print(base_net.features)
    # print(summary(base_net, torch.randn(1,3,300,300)))