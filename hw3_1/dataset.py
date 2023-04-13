from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
import imageio.v2 as imageio

import os

class Dataset(Dataset):

    def __init__(self, transform=None):
        self.filename_img = []
        self.transform = transform
        self.path = '../hw3_data/p1_data/val'
        self.filename_img = sorted([file for file in os.listdir(self.path)
                                    if file.endswith('.png')])

    def __len__(self):
        return len(self.filename_img)

    def __getitem__(self, idx):
        image = imageio.imread(os.path.join(self.path, self.filename_img[idx]))
        image = transforms.ToPILImage()(image)
        if self.transform:
            image = self.transform(image)
        return image,self.filename_img[idx]