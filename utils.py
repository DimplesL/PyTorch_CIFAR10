# -*- coding: utf-8 -*-
"""
@author: Qiu Yurui
@contact: qiuyurui@maituan.com
@software: Pycharm
@file: utils.py
@time: 2021/8/12 4:59 下午
@desc:
"""
import PIL
import torch
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt


def show_img(img: np.ndarray, title='img'):
    from matplotlib import pyplot as plt
    color = (len(img.shape) == 3 and img.shape[-1] == 3)
    # imgs = np.expand_dims(imgs, axis=0)
    # for i, img in enumerate(imgs):
    plt.figure()
    plt.title(title)
    plt.imshow(img, cmap=None if color else 'gray')
    plt.show()
