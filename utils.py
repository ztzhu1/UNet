# -*- encoding: utf-8 -*-
'''
Author: Z.T. Zhu
Date  : September 6th, 2022
'''

def plot_mask(ax, image, mask, alpha=0.4):
    if len(mask.shape) == 3:
        mask = mask[:, :, 0]
    ax.imshow(image)
    ax.imshow(mask, alpha=alpha)
    ax.set_axis_off()