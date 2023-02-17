import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

class ImageDataset(Dataset):
    def __init__(self, root, classes=None, transform=None, target_transform=None):
        self.root = root
        self.classes = classes
        self.transform = transform
        self.target_transform = target_transform
        if classes is None or len(classes) == 0 or len(classes[0]) == 0:
            real_dir = root
        else:
            real_dir = os.path.join(root, classes[0])
        if not os.path.exists(real_dir):
            raise ValueError(f"Path not exist: {real_dir}")
        if not os.path.isdir(real_dir):
            raise ValueError(f"Path not dir: {real_dir}")
        name_arr = os.listdir(real_dir)
        name_arr.sort()
        self.img_path_arr = [os.path.join(real_dir, n) for n in name_arr]

    def __getitem__(self, index):
        img_path = self.img_path_arr[index]
        image = np.array(Image.open(img_path).convert("RGB"))
        target = 0  # label, or class index
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            target = self.target_transform(target)
        return image, target

    def __len__(self):
        return len(self.img_path_arr)
