import torchvision
from torch.utils.tensorboard import SummaryWriter

dataset_transform = torchvision.transforms.Compose([
    # torchvision.transforms.Resize((64, 64)),
    torchvision.transforms.ToTensor()
])

train_set = torchvision.datasets.CIFAR10(root = "./dataset", train = True, download = True, transform = dataset_transform)
test_set = torchvision.datasets.CIFAR10(root = "./dataset", train = False, download = True, transform = dataset_transform)

# print(test_set[0])
# print(test_set.classes)
#
# img, target = test_set[0]
# print(img)
# print(target)
# print(test_set.classes[target])
#
# img.show() # 32 * 32

# print(test_set[0][0].shape)

writer = SummaryWriter("p10")

for i in range(10):
    img, target = test_set[i]
    writer.add_image("test_set_10", img, i)

writer.close()