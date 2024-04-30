import time

import torch
import torchvision
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter


# from model import Model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 准备数据集
train_data = torchvision.datasets.CIFAR10(root="../dataset", train=True, download=True,
                                          transform=torchvision.transforms.ToTensor())

test_data = torchvision.datasets.CIFAR10(root="../dataset", train=False, download=True,
                                         transform=torchvision.transforms.ToTensor())

# length of data
train_data_size = len(train_data)
test_data_size = len(test_data)
# if train_data_size=10, test_data_size=10
print("length of train_data: {}".format(train_data_size))
print("length of test_data: {}".format(test_data_size))

# use dataloader to load data
train_dataloader = DataLoader(dataset=train_data, batch_size=64)
test_dataloader = DataLoader(dataset=test_data, batch_size=64)


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.model1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, stride=1, padding=2),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, stride=1, padding=2),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2),
            nn.MaxPool2d(kernel_size=2),
            nn.Flatten(),
            nn.Linear(1024, 64),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        x = self.model1(x)
        return x


# build model
# model = Model()
model = Model()
model = model.to(device)

# define loss function
loss_fn = CrossEntropyLoss()
loss_fn = loss_fn.to(device)

# define optimizer
learning_rate = 1e-2  # 1e-2=1*10^-2=0.01
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# parameters setting
# record training step
total_train_step = 0
# record test step
total_test_step = 0
# training epoch
epoch = 100

start_time = time.time()

# use tensorboard to record loss
writer = SummaryWriter("logs_train")

for i in range(epoch):
    print("-------------epoch: {}-------------".format(i + 1))
    # training
    for data in train_dataloader:
        imgs, targets = data

        imgs = imgs.to(device)
        targets = targets.to(device)

        output = model(imgs)

        loss = loss_fn(output, targets)
        # optimizer
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_train_step += 1
        # print loss every 100 steps
        if total_train_step % 100 == 0:
            print("training step: {}, loss: {}".format(total_train_step, loss.item()))
            end_time = time.time()
            print("time: {}".format(end_time - start_time))
            writer.add_scalar("train_loss", loss.item(), total_train_step)

    # testing
    total_test_loss = 0
    total_accuracy = 0

    with torch.no_grad():
        for data in test_dataloader:
            imgs, targets = data

            imgs = imgs.to(device)
            targets = targets.to(device)

            output = model(imgs)
            loss = loss_fn(output, targets)
            total_test_loss += loss.item()
            accuracy = (output.argmax(1) == targets).sum().item()
            total_accuracy += accuracy
    print("total test loss: {}".format(total_test_loss))
    print("total accuracy: {}".format(total_accuracy / test_data_size))
    writer.add_scalar("test_loss", total_test_loss, total_test_step)
    writer.add_scalar("test_accuracy", total_accuracy / test_data_size, total_test_step)
    total_test_step += 1
    # save model in every 10 epoch
    if (i + 1) % 10 == 0:
        torch.save(model, "../saved_model/CIFAR10_{}.pth".format(i + 1))
        print("model saved in ../saved_model/CIFAR10_{}.pth".format(i + 1))

writer.close()
