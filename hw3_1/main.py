import numpy as np
import torch
from pkg_resources import packaging
import clip
from dataset import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import json
from os import mkdir
import csv
from os.path import isdir, dirname

if __name__ == '__main__':
    print(clip.available_models())
    model, preprocess = clip.load("ViT-B/32")
    model.cuda().eval()
    input_resolution = model.visual.input_resolution
    context_length = model.context_length
    vocab_size = model.vocab_size
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(preprocess)
    print("Model parameters:", f"{np.sum([int(np.prod(p.shape)) for p in model.parameters()]):,}")
    print("Input resolution:", input_resolution)
    print("Context length:", context_length)
    print("Vocab size:", vocab_size)

    # print(clip.tokenize("Hello World!"))
    dataset = Dataset(preprocess)
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
    with open('../hw3_data/p1_data/id2label.json') as f:
        data = json.load(f)

        for i in range(50):
            names.append(data["%d" %i])
    # print(names)
    image_inputs = torch.cat([i.unsqueeze(0) for i in images]).to(device)
    # print(image_inputs.shape)
    text_inputs = torch.cat([clip.tokenize(f"No {c}, no score.") for c in names]).to(device)
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
            # print(indices.item())
        output = './output/pred.csv'
        cnt=0
        for a,b in zip(ans,idx):
            if a == int(b):
                cnt += 1
        print('accuracy',cnt/len(ans))
        # if not isdir(dirname(output)):
        #     mkdir(dirname(output))
        # with open(output, 'w+', newline='') as csvfile:
        #     writer = csv.writer(csvfile)
        #     writer.writerow(['image_name', 'label'])
        #     for name,i in zip(f_name,ans):
        #         writer.writerow([name, i])





