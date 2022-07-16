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


def main():
    images = os.listdir('/data/test_dataset_test')
    try:
        os.mkdir('/data/test_masks')
    except Exception:
        pass
    for i in tqdm(images):
        image = cv2.imread(f'/data/test_dataset_test/{i}')
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pred = inference_detector(model, image)[0][0]
        for box in pred: 
            if box[4] > threshold:
                cv2.rectangle(image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 3)
        cv2.imwrite(f'/data/test_masks/{i}', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

if __name__ == '__main__':
    main()
