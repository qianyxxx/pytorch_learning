import torch
import torchvision
from torch import nn
from torch.nn import ReLU, Sigmoid
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

input = torch.Tensor([[1, -0.5],
                      [-1, 3]])

input = torch.reshape(input, (-1, 1, 2, 2))  # -1: batch_size 1: channel 2: height 2: width
print(input)
print(input.shape)  # torch.Size([1, 1, 2, 2])

dataset = torchvision.datasets.CIFAR10(root="./dataset", train=False, download=True,
                                       transform=torchvision.transforms.ToTensor())

dataloader = DataLoader(dataset=dataset, batch_size=64)

class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.relu1 = ReLU()
        self.sigmoid1 = Sigmoid()

    def forward(self, input):
        output = self.sigmoid1(input)
        return output


model = Model()  # 初始化模型
# output = model(input)
# print(output)

writer = SummaryWriter("sigmoid")
step = 0
for data in dataloader:
    imgs, targets = data
    writer.add_images("input", imgs, step)
    output = model(imgs)
    writer.add_images("output", output, step)
    step += 1

writer.close()