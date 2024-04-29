import torch
from torch.utils.data import Dataset
from PIL import Image
import platform
import os


# help(Dataset)
# print(torch.__version__)
# print(platform.python_version())

class MyData(Dataset):

    def __init__(self, root_dir, label_dir):
        self.root_dir = root_dir
        self.label_dir = label_dir
        self.path = os.path.join(self.root_dir, self.label_dir)
        self.img_path = os.listdir(self.path)

    def __getitem__(self, idx):
        img_mane = self.img_path[idx]
        img_item_path = os.path.join(self.root_dir, self.label_dir, img_mane)
        img = Image.open(img_item_path)
        label = self.label_dir

    def __len__(self):
        return len(self.img_path)


root_dir = "dataset/train"
ants_label_dir = "ants"
bees_label_dir = "bees"
ants_dataset = MyData(root_dir, ants_label_dir)
bees_dataset = MyData(root_dir, bees_label_dir)

train_dataset = ants_dataset + bees_dataset
