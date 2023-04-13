import json
import random
from sys import argv

import torch
import torchvision.transforms.functional as TF
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
tokenizer = Tokenizer.from_file('./caption_tokenizer.json')
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
    for i in range(config.max_position_embeddings - 1):#config.max_position_embeddings - 1
        predictions = model(image, caption, cap_mask)
        # print(cap_mask)
        predictions = predictions[:, i, :]
        predicted_id = torch.argmax(predictions, axis=-1)

        if predicted_id[0] == 3:
            return caption

        caption[:, i+1] = predicted_id[0]
        cap_mask[:, i+1] = False

    return caption
if __name__ == '__main__':
    file_path,json_path = argv[1],argv[2]
    checkpoint_path = './checkpoint_25.pth'
    out_dict = {}
    model, _ = caption.build_model(config)
    print("Loading Checkpoint...")
    checkpoint = torch.load(checkpoint_path, map_location='cuda')
    model.load_state_dict(checkpoint['model'])
    model = model.to(device)
    model.mlp.layers[2].out_features = tokenizer.get_vocab_size()
    # print(model)
    testset = dataset(path=file_path, transform=transform)

    BATCH_SIZE = 128
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
            output = evaluate(model,image,caption,cap_mask)
            result = tokenizer.decode(output[0].tolist(), skip_special_tokens=True)
            out_dict[filename[i]] = result[:77].capitalize()
    if not os.path.isdir(os.path.dirname(json_path)):
        os.mkdir(json_path)
    with open(json_path, "w+") as f:
        json.dump(out_dict, f, indent=2)


