from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

img_path = "dataset/train/ants_image/0013035.jpg"
img = Image.open(img_path)
# print(img)

# 如何使用transforms.ToTensor()将PIL图片转换为tensor

writer = SummaryWriter("logs")

tensor_trans = transforms.ToTensor()
tensor_img = tensor_trans(img)

# print(tensor_img)

writer.add_image("Tensor_img", tensor_img)

writer.close()