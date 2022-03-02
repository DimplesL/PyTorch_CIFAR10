# -*- coding: utf-8 -*-
"""
@author: Qiu Yurui
@contact: qiuyurui@maituan.com
@software: Pycharm
@file: test.py
@time: 2021/8/16 5:29 下午
@desc:
"""
import os
import sys
import pathlib
import cv2
import numpy as np

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
__dir__ = pathlib.Path(os.path.abspath(__file__))
sys.path.append(str(__dir__))
sys.path.append(str(__dir__.parent.parent))

import torch
from torchvision import transforms
from cifar10_models.resnet import resnet18
from utils import *
import onnxruntime


# logger = get_logger('torch classify', log_file='test.log')


class ClassInfer:
    def __init__(self, model_path, class_num):
        self.class_to_idx = {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 4, '6': 5, '7': 6, '8': 7}
        self.classes = ['圆',
                        '慢',
                        '数',
                        '禁止',
                        '箭下',
                        '箭前',
                        '箭右',
                        '箭左',
                        '箭掉头',
                        '自行车',
                        '行人']
        if model_path.endswith('onnx'):
            self.type = 'onnx'
            self.session = onnxruntime.InferenceSession(model_path)
        else:
            self.type = 'pt'
            ckpt = torch.load(model_path, map_location='cpu')
            self.model = resnet18(class_num)
            new_state_dict = {k.split('model.')[-1]: v for k, v in ckpt['state_dict'].items()}
            self.model.load_state_dict(new_state_dict)
            self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
            self.model.to(self.device)
            self.model.eval()

        self.input_size = (32, 32)
        self.transform = transforms.Compose([
            # transforms.Resize((56, 128)),  # h, w
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.4914, 0.4822, 0.4465),
                                 std=(0.2471, 0.2435, 0.2616))
        ])

    def to_numpy(self, tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

    def normalize(self, img):
        # img: RGB 32 32 3
        img = img.astype('float32') / 255.
        img = ((img - np.array([0.4914, 0.4822, 0.4465])) / np.array([0.2471, 0.2435, 0.2616])).astype('float32')
        return img

    @torch.no_grad()
    def predict(self, img):

        img = self.pre_process(img)

        if self.type == 'pt':
            tensor = self.transform(img)
            tensor = tensor.unsqueeze(dim=0)
            tensor = tensor.to(self.device)
            with torch.no_grad():
                output = self.model(tensor)[0]
            probabilities = output.sigmoid().detach().numpy()
        elif self.type == 'onnx':
            # 归一化
            # tensor = self.to_numpy(tensor)
            img = self.normalize(img)
            img = np.expand_dims(img.transpose((2, 0, 1)), axis=0)
            output = self.session.run([self.session.get_outputs()[0].name], {self.session.get_inputs()[0].name: img})[0]
            t1 = time.time()
            probabilities = self.sigmoid(output)
            t2 = time.time()
            # output = torch.tensor(output)
            # probabilities = output.sigmoid().detach().numpy()
            # t3 = time.time()
            # print(t2 - t1, t3 - t2)
        return probabilities

    def pre_process(self, image: np.ndarray) -> np.ndarray:
        """
        对cv2读取的单张BGR图像进行图像等比例伸缩，空余部分pad 0
        :param image: cv2读取的bgr格式图像， (h, w, 3)
        :return: 等比例伸缩后的图像， (h, w, 3)
        """
        h, w = image.shape[:2]
        target_h, target_w = self.input_size[0], self.input_size[1]
        (new_h, new_w), (left, right, top, bottom) = self.get_rescale_size(h, w, target_h, target_w)

        # 等比例缩放
        image = cv2.resize(image, (new_w, new_h))
        # padding
        image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # image = image.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        image = np.ascontiguousarray(image)
        # cv2.imshow(f'show', image)
        # cv2.waitKey(0)
        return image

    @staticmethod
    def get_rescale_size(src_h: int, src_w: int, target_h: int, target_w: int) -> \
            ((int, int), (int, int, int, int)):
        """
        按长边等比例缩放，短边pad 0
        :param src_h: 源尺寸高
        :param src_w: 源尺寸宽
        :param target_h: 目标尺寸高
        :param target_w: 目标尺寸宽
        :return: （缩放后高，缩放后宽），（左边需要pad的宽度，右边需要pad的宽度，上边需要pad的宽度，下边需要pad的宽度）
        """
        # 等比例缩放
        scale = max(src_h / target_h, src_w / target_w)
        new_h, new_w = int(src_h / scale), int(src_w / scale)
        # padding
        left_more_pad, top_more_pad = 0, 0
        if new_w % 2 != 0:
            left_more_pad = 1
        if new_h % 2 != 0:
            top_more_pad = 1
        left = right = (target_w - new_w) // 2
        top = bottom = (target_h - new_h) // 2
        left += left_more_pad
        top += top_more_pad
        return (new_h, new_w), (left, right, top, bottom)

    def crop(self, img, points):
        xmin, ymin, xmax, ymax = float('inf'), float('inf'), 0, 0

        for point in points:
            xmin = min(xmin, point[0])
            xmax = max(xmax, point[0])
            ymin = min(ymin, point[1])
            ymax = max(ymax, point[1])
        return img[ymin:ymax, xmin:xmax]

    def sigmoid(self, x):
        s = 1 / (1 + np.exp(-x))
        return s


if __name__ == "__main__":
    import time
    import numpy as np

    model_path = '/Users/qiuyurui/Desktop/models_file/res18focalfixepoch=49-step=1849.onnx'  # '/Users/qiuyurui/Desktop/models_file/res18focalfixepoch=49-step=1849.pt'  #res18focal-epoch=92-step=4370.pt'  # epoch=31-step=1503.pt'  # res18epoch=83-step=3947.pt'
    root = '/Users/qiuyurui/Projects/datas/new_crop/test'
    # img_path = '/Users/qiuyurui/Desktop/Text-Detect-Data/cutData2/2_6_101649580_3.jpg'
    classify = ClassInfer(model_path, 11)
    folders = os.listdir(root)
    folders.sort()
    for folder in folders[::-1]:
        folder_path = os.path.join(root, folder)
        # if not os.path.isdir(folder_path):
        #     continue
        if folder.startswith('.') or not folder_path.endswith('g'):  # os.path.isfile(folder_path)
            continue
        # imgs = os.listdir(folder_path)
        # for img_path in imgs:
        #     if img_path.startswith('.'):
        #         continue
        #     if os.path.isdir(os.path.join(folder_path, img_path)):
        #         continue
        img = cv2.imread(folder_path)
        # h, w = img.shape[:2]
        # s = h * w
        # if h > w:
        #     img = np.rot90(img, -1)
        # img1 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # img1 = cv2.resize(img1, (128, 56))

        t1 = time.time()
        pred = classify.predict(img)
        # if s < 700 or score < 0.7:
        #     pred, score = 0, 0
        t2 = time.time()
        print(np.argmax(pred))
        print(pred, f"time use: {t2 - t1}")
        # cv2.imshow(f'show', img)
        # cv2.waitKey(0)
        # cv2.destroyWindow(f'{pred}')
