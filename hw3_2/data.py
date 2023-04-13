import json
import os
import random

import torch
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import imageio.v2 as imageio
import numpy as np
import pandas as pd
from tokenizers import Tokenizer

from models.utils import nested_tensor_from_tensor_list

tokenizer = Tokenizer.from_file('../hw3_data/caption_tokenizer.json')
# print(tokenizer.encode().pad())
MAX_DIM = 224
class dataset(Dataset):

    def __init__(self, path,transform):
        self.path = path
        self.data = None
        self.get_data()
        self.transform = transform

    def __len__(self):
        return len(self.data.index)
        # return 100
    def __getitem__(self, idx):
        out = self.data.loc[idx]
        # print(out['file_name'])
        image = imageio.imread(os.path.join(self.path, "%s" %out['file_name']))
        if len(image.shape) !=3:
            image = torch.from_numpy(np.array([image,image,image]))
        if self.transform:
            image = self.transform(image)
        image = nested_tensor_from_tensor_list(image.unsqueeze(0))
        cap = out["caption"]
        cap = tokenizer.encode(cap)
        cap.pad(length=129)

        cap_mask = (
                1 - np.array(cap.attention_mask)).astype(bool)
        # print(len(cap.ids),len(cap.attention_mask))
        return image.tensors.squeeze(0), image.mask.squeeze(0) ,np.array(cap.ids),cap_mask

    def get_data(self):
        if self.path.split('/')[-1] == 'train':
            with open('../hw3_data/p2_data/train.json') as f:
                data = json.load(f)
                ann = pd.DataFrame(data['annotations'])
                img_id = pd.DataFrame(data['images'])
                self.data = pd.merge(ann,img_id,left_on = 'image_id',right_on='id',how='outer').drop(['id_x','id_y'],axis=1)
                # print(self.data)
        else:
            with open('../hw3_data/p2_data/val.json') as f:
                data = json.load(f)
                ann = pd.DataFrame(data['annotations'])
                img_id = pd.DataFrame(data['images'])
                self.data = pd.merge(ann, img_id, left_on='image_id', right_on='id', how='outer').drop(['id_x', 'id_y'],axis=1)
                # print(self.data)




def get_dataset(type='train'):
    img_size=299
    if type == 'train':

        train_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((img_size, img_size)),
            RandomRotation(),
            # transforms.RandomHorizontalFlip(),
            # transforms.Lambda(under_max),
            transforms.ColorJitter(brightness=[0.5, 1.3], contrast=[
                0.8, 1.5], saturation=[0.2, 1.5]),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])



        return dataset(path='../hw3_data/p2_data/images/train',
                       transform=train_transform)
    else:
        val_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((img_size, img_size)),
            # transforms.Lambda(under_max),
            transforms.ToTensor(),
            # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        return dataset(path='../hw3_data/p2_data/images/val',
                       transform=val_transform)


class RandomRotation:
    def __init__(self, angles=[0, 90, 180, 270]):
        self.angles = angles

    def __call__(self, x):
        angle = random.choice(self.angles)
        return TF.rotate(x, angle, expand=True)

def under_max(image):
    if image.mode != 'RGB':
        image = image.convert("RGB")

    shape = np.array(image.size, dtype=np.float)
    long_dim = max(shape)
    scale = MAX_DIM / long_dim

    new_shape = (shape * scale).astype(int)
    image = image.resize(new_shape)

    return image
