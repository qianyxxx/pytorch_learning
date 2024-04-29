import torch
import torchvision
from PIL import Image
from torch import nn

from test_tensorboard import image_path

image_path = "imgs/dogs.png"
image = Image.open(image_path)
print(image)
image.show()

transform = torchvision.transforms.Compose([torchvision.transforms.Resize((32, 32)),
                                            torchvision.transforms.ToTensor()])

image = transform(image)
print(image.shape)


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


model = torch.load("saved_model/CIFAR10_10.pth", map_location=torch.device("cpu"))

print(model)

image = torch.reshape(image, (1, 3, 32, 32)) # batch_size=1, channel=3, height=32, width=32
output = model(image)
model.eval()
with torch.no_grad():
    output = model(image)
print(output)

print(output.argmax(1))
