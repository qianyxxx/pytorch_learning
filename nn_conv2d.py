import torch
import torchvision
from torch import nn
from torch.nn import Conv2d
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10(root="./dataset", train=False, download=True,
                                       transform=torchvision.transforms.ToTensor())

dataloader = DataLoader(dataset=dataset, batch_size=64, shuffle=True, num_workers=0, drop_last=True)


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = Conv2d(in_channels=3, out_channels=6, kernel_size=3, stride=1, padding=0)

    def forward(self, x):
        x = self.conv1(x)
        return x


model = Model()  # 初始化模型
# print(model)

writer = SummaryWriter("conv2d")
step = 0

for data in dataloader:
    imgs, targets = data
    output = model(imgs)
    print(imgs.shape)  # torch.Size([64, 3, 32, 32]) 64: batch_size 3: input_channel 32: height 32: width
    print(output.shape)  # torch.Size([64, 6, 30, 30]) 64: batch_size 6: output_channel 30: height 30: width
    writer.add_images("input", imgs, step)
    # Torch.Size([64, 6, 30, 30]) -> Torch.Size([xxx, 3, 30, 30])
    output = torch.reshape(output, (-1, 3, 30, 30)) # -1: batch_size 3: output_channel 30: height 30: width
    writer.add_images("output", output, step)
    print(output.shape)
    step += 1
