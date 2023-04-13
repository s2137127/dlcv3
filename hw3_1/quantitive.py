import numpy as np
import torch
import matplotlib.pyplot as plt
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

    dataset = Dataset()

    images ,f_name,origin_image= [],[],[]
    for i in range(3):
        idx = np.random.randint(0,len(dataset),1)
        print(idx.item())
        image, name = dataset[idx.item()]
        print(name)
        origin_image.append(image)
        images.append(preprocess(image))
        f_name.append(name)
    # print(image.shape)
    names = []
    with open('../hw3_data/p1_data/id2label.json') as f:
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
            top_probs, indices = similarity[0].topk(5)
            ans.append([top_probs,indices])
            # print(indices.item())

    plt.figure(figsize=(1, 6))

    for i, image in enumerate(origin_image):
        plt.subplot(1, 6, 2*i + 1)
        plt.imshow(image)
        plt.axis("off")

        plt.subplot(1, 6, 2*i + 2)
        # print(ans[i][0].shape)
        y = np.arange(ans[i][0].shape[-1])
        plt.grid()
        plt.barh(y, ans[i][0].cpu())
        plt.gca().invert_yaxis()
        plt.gca().set_axisbelow(True)
        plt.yticks(y, [names[int(index)] for index in ans[i][1].cpu().numpy()])
        plt.xlabel("probability")

    plt.subplots_adjust(wspace=0.5)
    plt.show()