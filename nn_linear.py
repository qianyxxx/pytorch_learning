import torch
import torchvision
from torch.nn import Linear
from torch.utils.data import DataLoader

dataset = torchvision.datasets.CIFAR10(root="./dataset", train=False, download=True,
                                       transform=torchvision.transforms.ToTensor())

dataloader = DataLoader(dataset, batch_size=64)


class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear1 = Linear(3 * 32 * 32 * 64, 10)

    def forward(self, input):
        output = self.linear1(input)
        return output


model = Model()

for data in dataloader:
    imgs, targets = data
    print(imgs.shape)  # torch.Size([64, 3, 32, 32]) 64: batch_size 3: input_channel 32: height 32: width
    # output = torch.reshape(imgs, (1, 1, 1, -1))  # 1: batch_size 1: channel 1: height -1: width
    output = torch.flatten(imgs)
    print(output.shape)
    output = model(output)
    print(output.shape)
