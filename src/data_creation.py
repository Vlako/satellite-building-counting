import json
import os
from multiprocessing import Pool

from tqdm import tqdm
from skimage.draw import polygon
import cv2
import numpy as np


anno = json.load(open('/data/train/annotation.json'))
anno['categories'][0]['id'] = 1

# Удаление из обучения изображений из тестовой выборки
test_images = set(os.listdir('/data/test_dataset_test/'))
anno['images'] = [i for i in anno['images'] if i['file_name'] not in test_images]

image_ids = set([i['id'] for i in anno['images']])


def change_anno(i):
    i['category_id'] = 1
    i['bbox'] = [min(i['segmentation'][0][0::2]), 
                 min(i['segmentation'][0][1::2]), 
                 max(i['segmentation'][0][0::2]) - min(i['segmentation'][0][0::2]), 
                 max(i['segmentation'][0][1::2]) - min(i['segmentation'][0][1::2])]
    return i

anno['annotations'] = [change_anno(i) for i in anno['annotations'] if i['image_id'] in image_ids]


train = {
    'categories': anno['categories']
}

train['images'] = [i for i in anno['images'] if i['id'] % 10]
train['annotations'] = [i for i in anno['annotations'] if i['image_id'] % 10]


try:
    os.mkdir('/data/train/masks')
except Exception:
    pass


def make_mask(img):
    mask = np.zeros((300, 300), np.uint8)
    for an in train['annotations']:
        if an['image_id'] != img['id']:
            continue
        seg = an['segmentation']
        rr, cc = polygon(seg[0][1::2], seg[0][::2])
        mask[np.clip(rr, 0, 299), np.clip(cc, 0, 299)] = 1
    cv2.imwrite(f'/data/train/masks/{img["file_name"].replace("jpg", "png")}', mask)


pool = Pool(12)

list(tqdm(pool.imap(make_mask, train['images']), total=len(train['images'])))

json.dump(train, open('/data/train.json', 'w'))


val = {
    'categories': anno['categories']
}

val['images'] = [i for i in anno['images'] if i['id'] % 10 == 0]
val['annotations'] = [i for i in anno['annotations'] if i['image_id'] % 10 == 0]

json.dump(val, open('/data/val.json', 'w'))
