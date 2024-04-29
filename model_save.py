import torch
import torchvision
from torch import nn

vgg16 = torchvision.models.vgg16(weights=None)

# Save the model method 1
torch.save(vgg16, "vgg16_method1.pth")

# Save the model method 2 (recommended)
torch.save(vgg16.state_dict(), "vgg16_method2.pth")  # Save the model's parameters to dictionary

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        output = self.conv1(x)
        return x

model = Model()
torch.save(model, "myModel.pth")