# -*- encoding: utf-8 -*-
'''
Author: Z.T. Zhu
Date  : September 6th, 2022
'''
'''
    The dataset is found on Kaggle.

    This is a toy model. We don't need 
so many images. And UNet are trained with
limited-size dataset. Lots of images are 
against, at least, the instructional purpose.

    Besides, I want to practice data augmentation.
So I chose 20 images of 20 cars randomly then
generated other 80 augmented images :)
'''

import os
import shutil
import numpy as np
import albumentations as A
from imageio import imread, imsave

NUM_CAR = 20
NUM_IMAGE_PER_CAR = 16
ROOT = './data'
IMAGE_INPUT_PATH = '/train/'
IMAGE_OUTPUT_PATH = '/images/'
MASK_INPUT_PATH = '/train_masks/'
MASK_OUTPUT_PATH = '/masks/'
IMAGE_SIZE = (512, 512)


def get_image_indices():
    indices = []
    for i in range(0, NUM_CAR * NUM_IMAGE_PER_CAR, NUM_IMAGE_PER_CAR):
        image_index = i * NUM_IMAGE_PER_CAR + np.random.randint(
            1, NUM_IMAGE_PER_CAR + 1)
        indices.append(image_index)
    return indices


def get_image_names(path):
    return os.listdir(ROOT + path)


def batch_move(names, indices, input_path, output_path):
    for index in indices:
        name = names[index]
        shutil.move(ROOT + input_path + name, ROOT + output_path + name)


def batch_transform():
    image_names = get_image_names(IMAGE_OUTPUT_PATH)
    mask_names = get_image_names(MASK_OUTPUT_PATH)
    transformer = build_transformer()
    for i, m in zip(image_names, mask_names):
        transform(transformer, i, m)

def transform(transformer, image_name, mask_name):
    image = imread(ROOT + IMAGE_OUTPUT_PATH + image_name)
    mask = imread(ROOT + MASK_OUTPUT_PATH + mask_name)

    for i in range(1, 5):
        transformed = transformer(image=image, mask=mask)
        transformed_image = transformed['image']
        transformed_mask = transformed['mask']
        imsave(
            ROOT + IMAGE_OUTPUT_PATH + image_name.replace('.jpg', '') + f'{i}.jpg',
            transformed_image
        )
        imsave(
            ROOT + MASK_OUTPUT_PATH + mask_name.replace('.gif', '') + f'{i}.gif',
            transformed_mask
            )

def build_transformer():
    transformer = A.Compose(
        [A.HorizontalFlip(p=0.5),
         A.Rotate(p=0.5),
         A.RGBShift(p=0.4)])
    return transformer

if __name__ == '__main__':
    # indices = get_image_indices()

    # image_names = get_image_names(IMAGE_INPUT_PATH)
    # batch_move(image_names, indices, IMAGE_INPUT_PATH, IMAGE_OUTPUT_PATH)

    # mask_names = get_image_names(MASK_INPUT_PATH)
    # batch_move(mask_names, indices, MASK_INPUT_PATH, MASK_OUTPUT_PATH)
    
    batch_transform()