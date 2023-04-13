import json
import random
from sys import argv

import torch
import torchvision.transforms.functional as TF
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import imageio.v2 as imageio
import numpy as np
from tokenizers import Tokenizer
from tqdm import tqdm
from configuration import Config
from models import caption

import os
#time python output.py ../hw3_data/p2_data/images/val/ ./out.json

device = 'cuda' if torch.cuda.is_available() else 'cpu'
tokenizer = Tokenizer.from_file('../hw3_data/caption_tokenizer.json')
# print(tokenizer.encode().pad())
config = Config()
img_size=299
transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((img_size, img_size)),
            # transforms.Lambda(under_max),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
            # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

class dataset(Dataset):

    def __init__(self, path,transform):
        self.path = path
        self.data = None
        self.transform = transform
        self.filename=sorted([file for file in os.listdir(self.path)
                                        if file.endswith('.jpg')])
    def __len__(self):
        return len(self.filename)
        # return 100
    def __getitem__(self, idx):

        image = imageio.imread(os.path.join(self.path, self.filename[idx]))
        if len(image.shape) !=3:
            image = torch.from_numpy(np.array([image,image,image]))
        if self.transform:
            image = self.transform(image)
        return image, self.filename[idx].split(".")[0]


def plot_attention(img, result, attention_plot):
    # untransform
    img[0] = img[0] * 0.229
    img[1] = img[1] * 0.224
    img[2] = img[2] * 0.225
    img[0] += 0.485
    img[1] += 0.456
    img[2] += 0.406

    img = img.cpu().numpy()
    # print(img.shape)
    img = img.transpose((1, 2, 0))

    temp_image = img
    fig = plt.figure(figsize=(15, 15))

    len_result = len(result)
    # print(len_result,len(attention_plot))
    for l in range(len_result):
        # print(attention_plot[l+1].shape)
        temp_att = attention_plot[l].reshape(19, 19)

        ax = fig.add_subplot(len_result // 2, len_result // 2, l + 1)
        ax.set_title(result[l])
        img = ax.imshow(temp_image)
        ax.imshow(temp_att.cpu(), cmap='jet', alpha=0.7, extent=img.get_extent())

    plt.tight_layout()
    # plt.imsave()
    plt.show()

class RandomRotation:
    def __init__(self, angles=[0, 90, 180, 270]):
        self.angles = angles

    def __call__(self, x):
        angle = random.choice(self.angles)
        return TF.rotate(x, angle, expand=True)


def create_caption_and_mask(start_token, max_length):
    caption_template = torch.zeros((1, max_length), dtype=torch.long,device=device)
    mask_template = torch.ones((1, max_length), dtype=torch.bool,device=device)

    caption_template[:, 0] = start_token
    mask_template[:, 0] = False

    return caption_template, mask_template
@torch.no_grad()
def evaluate(model,image,caption,cap_mask):
    model.eval()
    # mask_arr = []
    mask = None
    for i in range(config.max_position_embeddings - 1):#config.max_position_embeddings - 1
        predictions,mask = model(image, caption, cap_mask)
        # mask_arr.append(mask)
        # print(mask.shape)
        # print(cap_mask)
        predictions = predictions[:, i, :]
        predicted_id = torch.argmax(predictions, axis=-1)

        if predicted_id[0] == 3:
            return caption,mask

        caption[:, i+1] = predicted_id[0]
        cap_mask[:, i+1] = False

    return caption,mask
if __name__ == '__main__':
    file_path,out_path = argv[1],argv[2]
    checkpoint_path = './checkpoint_29.pth'
    out_dict = {}
    print("Checking for checkpoint.")

    model, _ = caption.build_model_out(config)
    print("Loading Checkpoint...")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    model = model.to(device)
    testset = dataset(path=file_path, transform=transform)

    BATCH_SIZE = 256
    # BATCH_SIZE = 6
    NUM_WORKER = 4

    data_loader_val = DataLoader(dataset=testset, batch_size=BATCH_SIZE, num_workers=NUM_WORKER, shuffle=True,
                                 pin_memory=True)

    start_token = tokenizer.token_to_id('[BOS]')


    for imgs, filename in tqdm(data_loader_val):
        imgs = imgs.to(device)
        for i in tqdm(range(len(filename))):
            image = imgs[i].unsqueeze(0)
            caption, cap_mask = create_caption_and_mask(start_token, 128)
            output,mask_arr  = evaluate(model,image,caption,cap_mask)
            mask_arr = mask_arr.squeeze()
            # print(mask_arr.squeeze().shape)
            # print("length",len(mask_arr))
            # print(output)
            result = tokenizer.decode(output[0].tolist(), skip_special_tokens=True)
            # print(result)
            result = result.split(" ")
            # out_dict[filename[i]] = result.capitalize().
            # print(result)
            plot_attention(imgs[i],result,mask_arr)



