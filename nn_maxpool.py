import torch
import torchvision
from torch import nn
from torch.nn import MaxPool2d
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10(root="./dataset", train=False, download=True,
                                       transform=torchvision.transforms.ToTensor())

dataloader = DataLoader(dataset=dataset, batch_size=64)

# # 5x5 input
# input = torch.tensor([[1, 2, 0, 3, 1],
#                       [0, 1, 2, 3, 1],
#                       [1, 2, 1, 0, 0],
#                       [5, 2, 3, 1, 1],
#                       [2, 1, 0, 1, 1]], dtype=torch.float32)
#
# input = torch.reshape(input, (-1, 1, 5, 5))  # 1: batch_size 1: channel 5: height 5: width
# print(input.shape)  # torch.Size([1, 1, 5, 5])


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.maxpool = nn.MaxPool2d(kernel_size=3, ceil_mode=True)

    def forward(self, input):
        output = self.maxpool(input)
        return output


model = Model()
# output = model(input)
# print(output)

writer = SummaryWriter("logs_maxpool")
step = 0
for data in dataloader:
    imgs, targets = data
    writer.add_images("input", imgs, step)
    output = model(imgs)
    writer.add_images("output", output, step)
    step += 1
    print(imgs.shape)  # torch.Size([64, 3, 32, 32]) 64: batch_size 3: input_channel 32: height 32: width
    print("-------------------")
    print(output.shape)  # torch.Size([64, 3, 10, 10]) 64: batch_size 3: output_channel 10: height 10: width

writer.close()

print(imgs[0].shape)
print(output[0].shape)