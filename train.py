import torch
import torchvision
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from model import Model

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

# build model
model = Model()

# define loss function
loss_fn = nn.CrossEntropyLoss()

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

# use tensorboard to record loss
writer = SummaryWriter("logs_train")

for i in range(epoch):
    print("-------------epoch: {}-------------".format(i + 1))
    # training
    for data in train_dataloader:
        imgs, targets = data
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
            writer.add_scalar("train_loss", loss.item(), total_train_step)

    # testing
    total_test_loss = 0
    total_accuracy = 0

    with torch.no_grad():
        for data in test_dataloader:
            imgs, targets = data
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
    # save model in every epoch
    torch.save(model, "saved_model/CFAR10_{}.pth".format(i + 1))
    print("model saved in saved_model/CFAR10_{}.pth".format(i + 1))

writer.close()
