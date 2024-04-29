import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torchvision.models import VGG16_Weights

# train_data = torchvision.datasets.ImageNet("/data_ImageNet", split='train', download=False,
#                                            transform=torchvision.transforms.ToTensor())

vgg16_false = torchvision.models.vgg16(weights=None)
vgg16_true = torchvision.models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1)

print("OK")

# print(vgg16_false)
# print(vgg16_true)

train_data = torchvision.datasets.CIFAR10(root="./dataset", train=False, download=True,
                                       transform=torchvision.transforms.ToTensor())

dataloader = DataLoader(train_data, batch_size=64)

vgg16_true.classifier.add_module("add_linear", nn.Linear(1000, 10))
print(vgg16_true)

vgg16_false.classifier[6] = nn.Linear(4096, 10)
print(vgg16_false)
