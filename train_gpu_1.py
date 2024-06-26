import time

import torch
import torchvision
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter


# from model import Model

# check cuda
def check_model_device(model):
    # Get the device of the first parameter of the model
    device = next(model.parameters()).device
    print(f"The model is on {device}")


# 准备数据集
train_data = torchvision.datasets.CIFAR10(root="./dataset", train=True, download=True,
                                          transform=torchvision.transforms.ToTensor())

test_data = torchvision.datasets.CIFAR10(root="./dataset", train=False, download=True,
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
if torch.cuda.is_available():
    model = model.cuda()

# Call the function to check the device of the model
check_model_device(model)

# define loss function
loss_fn = CrossEntropyLoss()
if torch.cuda.is_available():
    loss_fn = loss_fn.cuda()

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
# writer = SummaryWriter("logs_train")

for i in range(epoch):
    print("-------------epoch: {}-------------".format(i + 1))
    # training
    for data in train_dataloader:
        imgs, targets = data

        if torch.cuda.is_available():
            imgs = imgs.cuda()
            targets = targets.cuda()

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
            # writer.add_scalar("train_loss", loss.item(), total_train_step)

    # testing
    total_test_loss = 0
    total_accuracy = 0

    with torch.no_grad():
        for data in test_dataloader:
            imgs, targets = data
            if torch.cuda.is_available():
                imgs = imgs.cuda()
                targets = targets.cuda()

            output = model(imgs)
            loss = loss_fn(output, targets)
            total_test_loss += loss.item()
            accuracy = (output.argmax(1) == targets).sum().item()
            total_accuracy += accuracy
    print("total test loss: {}".format(total_test_loss))
    print("total accuracy: {}".format(total_accuracy / test_data_size))
    # writer.add_scalar("test_loss", total_test_loss, total_test_step)
    # writer.add_scalar("test_accuracy", total_accuracy / test_data_size, total_test_step)
    total_test_step += 1
    # save model in every epoch
    torch.save(model, "saved_model/CFAR10_{}.pth".format(i + 1))
    print("model saved in saved_model/CFAR10_{}.pth".format(i + 1))

# writer.close()
