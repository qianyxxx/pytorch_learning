import torch
import torchvision
from torch import nn
from model_save import *

# Load the model method 1
model = torch.load("vgg16_method1.pth")
# print(model)

# Load the model method 2
model2 = torch.load("vgg16_method2.pth")
# print(model2)  # print dictionary

vgg16 = torchvision.models.vgg16(weights=None)
# vgg16 = torch.load("vgg16_method2.pth")
vgg16.load_state_dict(torch.load("vgg16_method2.pth"))
# print(vgg16)  # print model


# trap

# class Model(nn.Module):
#     def __init__(self):
#         super(Model, self).__init__()
#         self.conv1 = conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
#
#     def forward(self, x):
#         output = self.conv1(x)
#         return x

model = torch.load("myModel.pth")
print(model)