from itertools import product
import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from mmdet.apis import init_detector, inference_detector

threshold = 0.5
config_file = '/model/model.py'
checkpoint_file = '/model/latest.pth'
model = init_detector(config_file, checkpoint_file, device='cuda:0')  # or device='cuda:0'


def tta(image, vertical_flip=False, horizontal_flip=False, rotate=False):
    if vertical_flip:
        image = cv2.flip(image, 0)
    if horizontal_flip:
        image = cv2.flip(image, 1)
    if rotate:
        image = np.rot90(image)
    return image

def main():
    submission = []
    images = os.listdir('/data/test_dataset_test')
    for i in tqdm(images):
        image = cv2.imread(f'/data/test_dataset_test/{i}')
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        preds = []
        for vert, hor, rot in product([False, True], repeat=3):
            tta_image = tta(image, vert, hor, rot)
            pred = inference_detector(model, tta_image)[0][0]
            pred = (pred[:, 4] > threshold).sum()
            preds.append(pred)
        submission.append({'img_num': i, 'number_of_houses': int(round(np.mean(pred)))})
    submission = pd.DataFrame(submission)
    submission.to_csv('/data/submission.csv', index=False)

if __name__ == '__main__':
    main()
