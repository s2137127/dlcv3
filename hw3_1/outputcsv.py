import numpy as np
import torch
import clip

from tqdm import tqdm
import json
from os import mkdir
import csv,sys
from os.path import isdir, dirname
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
import imageio.v2 as imageio

import os

class Dataset(Dataset):

    def __init__(self, path,transform=None):
        self.filename_img = []
        self.transform = transform
        self.path = path
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

if __name__ == '__main__':
    folder,json_path,output = sys.argv[1],sys.argv[2],sys.argv[3]

    model, preprocess = clip.load("ViT-B/32")
    model.cuda().eval()
    input_resolution = model.visual.input_resolution
    context_length = model.context_length
    vocab_size = model.vocab_size
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dataset = Dataset(path=folder,transform=preprocess)
    dataloader = DataLoader(dataset, batch_size=64,
                            shuffle=True, num_workers=2,pin_memory=True)
    images ,f_name,idx= [],[],[]
    for image,name in tqdm(dataloader):
        for i in range(len(name)):
            images.append(image[i])
            f_name.append(name[i])
            idx.append(name[i].split('_')[0])
    # print(idx)
    names = []
    with open(json_path) as f:
        data = json.load(f)

        for i in range(50):
            names.append(data["%d" %i])
    # print(names)
    image_inputs = torch.cat([i.unsqueeze(0) for i in images]).to(device)
    # print(image_inputs.shape)
    text_inputs = torch.cat([clip.tokenize(f"This is a {c} image.") for c in names]).to(device)
    # print(text_inputs)
    model.eval()
    with torch.no_grad():
        ans = []
        text_features = model.encode_text(text_inputs)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        for image_input in tqdm(image_inputs):
            # print(image_input.shape)
            image_input = image_input.unsqueeze(0)
            image_features = model.encode_image(image_input)


            image_features /= image_features.norm(dim=-1, keepdim=True)

            similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
            _, indices = similarity[0].topk(1)
            ans.append(indices.item())

        if not isdir(dirname(output)):
            mkdir(dirname(output))
        with open(output, 'w+', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['image_name', 'label'])
            for name,i in zip(f_name,ans):
                writer.writerow([name, i])





