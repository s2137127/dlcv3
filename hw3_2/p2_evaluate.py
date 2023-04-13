import os
import json
from collections import defaultdict
from argparse import ArgumentParser

import numpy as np
from PIL import Image
import clip
import torch
import language_evaluation
from matplotlib import pyplot as plt
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics.pairwise import cosine_similarity

def readJSON(file_path):
    try:
        with open(file_path) as f:
            data = json.load(f)
        return data
    except:
        return None


def getGTCaptions(annotations):
    img_id_to_name = {}
    for img_info in annotations["images"]:
        img_name = img_info["file_name"].replace(".jpg", "")
        img_id_to_name[img_info["id"]] = img_name

    img_name_to_gts = defaultdict(list)
    for ann_info in annotations["annotations"]:
        img_id = ann_info["image_id"]
        img_name = img_id_to_name[img_id]
        img_name_to_gts[img_name].append(ann_info["caption"])
    return img_name_to_gts


class CIDERScore:
    def __init__(self):
        self.evaluator = language_evaluation.CocoEvaluator(coco_types=["CIDEr"])

    def __call__(self, predictions, gts):
        """
        Input:
            predictions: dict of str
            gts:         dict of list of str
        Return:
            cider_score: float
        """
        # Collect predicts and answers
        predicts = []
        answers = []
        for img_name in predictions.keys():
            predicts.append(predictions[img_name])
            answers.append(gts[img_name])
        
        # Compute CIDEr score
        results = self.evaluator.run_evaluation(predicts, answers)
        return results['CIDEr']


class CLIPScore:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)
        self.model.eval()

    def __call__(self, predictions, images_root):
        """
        Input:
            predictions: dict of str
            images_root: str
        Return:
            clip_score: float
        """
        name_score=[]
        total_score = []
        cnt = 1
        for img_name, pred_caption in predictions.items():
            image_path = os.path.join(images_root, f"{img_name}.jpg")
            image = self.preprocess(Image.open(image_path).convert("RGB")).unsqueeze(0).to('cuda')
            pred_captionid = clip.tokenize(pred_caption).to('cuda')
            image_features = self.model.encode_image(image)
            text_features = self.model.encode_text(pred_captionid)
            # print(image.shape)

            score = self.getCLIPScore(image_features, text_features)
            total_score.append(score)
            name_score.append([img_name,score,pred_caption])
        return np.mean(total_score),name_score
    def getCLIPScore(self, image, caption):
        """
        This function computes CLIPScore based on the pseudocode in the slides.
        Input:
            image: PIL.Image
            caption: str
        Return:
            cilp_score: float
        """
        # image_features = self.model.encode_image(image)
        # text_features = self.model.encode_text(caption)
        return 2.5*cosine_similarity(image.cpu().detach().numpy(), caption.cpu().detach().numpy())

def visiualize_top_last(name_score,path):
    min = np.argmin(name_score[:, 1],axis=0)
    max = np.argmax(name_score[:, 1],axis=0)
    min = name_score[min]
    max = name_score[max]
    plt.figure(figsize=(16,16))
    plt.subplot(1, 2, 1)
    plt.title('top1 \n clipscore: %f \n %s ' %(max[1],max[2]))
    # plt.text(4, 1, max[2], fontsize=20, color='green')
    # plt.text(4, 3, "clipscore: %d" % max[1], fontsize=20, color='green')
    image_path = os.path.join(path, f"{max[0]}.jpg")
    plt.imshow(Image.open(image_path).convert("RGB"))


    plt.subplot(1, 2, 2)
    plt.title('last1 \n clipscore: %f \n %s ' %(min[1],min[2]))
    # plt.text(12, 3, min[2], fontsize=20, color='green')
    # plt.text(12, 5, "clipscore: %d" %min[1], fontsize=20, color='green')
    image_path = os.path.join(path, f"{min[0]}.jpg")
    plt.imshow(Image.open(image_path).convert("RGB"))
    plt.show()
def main(args):
    # Read data
    predictions = readJSON(args.pred_file)
    annotations = readJSON(args.annotation_file)

    # Preprocess annotation file
    gts = getGTCaptions(annotations)

    # Check predictions content is correct
    assert type(predictions) is dict
    assert set(predictions.keys()) == set(gts.keys())
    assert all([type(pred) is str for pred in predictions.values()])

    # CIDErScore
    cider_score = CIDERScore()(predictions, gts)
    # print(cider_score)
    # CLIPScore
    clip_score,name_score = CLIPScore()(predictions, args.images_root)
    name_score = np.array(name_score)
    # print(name_score.shape)
    print(f"CIDEr: {cider_score} | CLIPScore: {clip_score}")

    visiualize_top_last(name_score,args.images_root)

if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--pred_file", help="Prediction json file")
    parser.add_argument("--images_root", default="../hw3_data/p2_data/images/val/", help="Image root")
    parser.add_argument("--annotation_file", default="../hw3_data/p2_data/val.json", help="Annotation json file")

    args = parser.parse_args()

    main(args)
