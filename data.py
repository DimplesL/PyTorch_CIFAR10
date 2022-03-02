import os
import zipfile

import pytorch_lightning as pl
import requests
from torch.utils.data import DataLoader
from torchvision import transforms as T
from torchvision.datasets import CIFAR10
from torchvision.datasets import VisionDataset
from tqdm import tqdm
from PIL import Image, ImageOps
import numpy as np
import json
from typing import Any, Callable, Optional, Tuple
import imagesize
from utils import *


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def has_file_allowed_extension(filename, extensions):
    """Checks if a file is an allowed extension.

    Args:
        filename (string): path to a file
        extensions (tuple of strings): extensions to consider (lowercase)

    Returns:
        bool: True if the filename ends with one of given extensions
    """
    return filename.lower().endswith(extensions)


def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)


def areaS(path):
    # img = Image.open(path)
    # w, h = img.size
    w, h = imagesize.get(path)
    s = h * w
    return s, h


IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png')


def make_dataset(file_list, class_to_idx, extensions=None, is_valid_file=None):
    instances = []
    file_list = os.path.expanduser(file_list)
    both_none = extensions is None and is_valid_file is None
    both_something = extensions is not None and is_valid_file is not None
    if both_none or both_something:
        raise ValueError("Both extensions and is_valid_file cannot be None or not None at the same time")
    if extensions is not None:
        def is_valid_file(x):
            return has_file_allowed_extension(x, extensions)
    with open(file_list, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip('\n')
            if line == '':
                continue
            img_path, target_class = line.split('\t')
            if is_valid_file(img_path):
                s, h = areaS(img_path)
                class_index = class_to_idx[target_class]
                if s > 200 and h > 10:  # or class_index in [5, 6, 7]:
                    item = img_path, class_index
                # elif s < 500:
                #     item = path, 0
                else:
                    continue
                instances.append(item)
    return instances


def make_dataset_ori_label(file_list, extensions=None, is_valid_file=None):
    instances = []
    file_list = os.path.expanduser(file_list)
    both_none = extensions is None and is_valid_file is None
    both_something = extensions is not None and is_valid_file is not None
    if both_none or both_something:
        raise ValueError("Both extensions and is_valid_file cannot be None or not None at the same time")
    if extensions is not None:
        def is_valid_file(x):
            return has_file_allowed_extension(x, extensions)
    print('init the dataset')
    cache_file = file_list.replace('txt', 'txtcache')
    if os.path.exists(cache_file):
        print(f'reload the dataset cache: {cache_file}')
        return json.load(open(cache_file, 'r'))
    with open(file_list, 'r') as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            #             print(f"{i} / {len(lines)}")
            line = line.strip('\n')
            if line == '':
                continue
            img_path, target_class = line.split('\t')

            target_class = target_class.split('-')[-1]
            if target_class == '圆形':
                target_class = '圆'
            if target_class == '数字':
                target_class = '数'
            if target_class in ['多颜色组合', '颜色不可分', '无效', '红', '绿', '黄', '多颜色组合']:
                continue
            img_path = img_path.replace('/home/hadoop-mtcv/cephfs/data', '/workdir')
            if is_valid_file(img_path):
                s, h = areaS(img_path)
                if s > 200 and h > 10:  # or class_index in [5, 6, 7]:
                    item = img_path, target_class
                # elif s < 500:
                #     item = path, 0
                else:
                    continue
                instances.append(item)
    json.dump(instances, open(cache_file, 'w'))
    print('init complete')
    return instances


class SingleLabelDataset(VisionDataset):
    def __init__(
            self,
            root: str,
            train: bool = True,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            is_valid_file: Optional[Callable] = None,
    ) -> None:
        super(SingleLabelDataset, self).__init__(root, transform=transform,
                                                 target_transform=target_transform)

        self.train = train  # training set or test set
        extensions = IMG_EXTENSIONS if is_valid_file is None else None
        class_to_idx = {'圆': 0,  # single label
                        }
        idx_to_class = {v: k for k, v in class_to_idx.items()}
        self.samples = make_dataset(root, class_to_idx, extensions, is_valid_file)
        if len(self.samples) == 0:
            raise (RuntimeError("Found 0 files in subfolders of: " + root +
                                "\n""Supported extensions are: " + ",".join(extensions)))
        self.loader = default_loader
        self.extensions = extensions

        self.idx_to_class = idx_to_class
        self.class_to_idx = class_to_idx
        # self.targets = [s[1] for s in self.samples]

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img_path, target = self.samples[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.open(img_path)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.samples)


class MultiLabelDataset(VisionDataset):
    def __init__(
            self,
            root: str,
            train: bool = True,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            is_valid_file: Optional[Callable] = None,
    ) -> None:
        super(MultiLabelDataset, self).__init__(root, transform=transform,
                                                target_transform=target_transform)

        self.train = train  # training set or test set
        extensions = IMG_EXTENSIONS if is_valid_file is None else None

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
        self.samples = make_dataset_ori_label(root, extensions, is_valid_file)
        if len(self.samples) == 0:
            raise (RuntimeError("Found 0 files in file_list of: " + root +
                                "\n""Supported extensions are: " + ",".join(extensions)))
        self.loader = default_loader
        self.extensions = extensions

        self.targets = [self.convert_label(s[1]) for s in self.samples]

    def convert_label(self, label):
        label_items = label.split('+')
        vector = [cls in label_items for cls in self.classes]
        return torch.FloatTensor(np.array(vector, dtype=float))

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img_path, label = self.samples[index]
        target = self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.open(img_path)
        w, h = img.size
        if w > h:
            w_s = 32
            h_s = int((32 + 1) / w * h)
            border = (0, (32 - h_s) // 2, 0, (32 - h_s) // 2)
        elif w == h:
            w_s = 32
            h_s = 32
            border = (0, 0, 0, 0)
        else:
            h_s = 32
            w_s = int((32 + 1) / h * w)
            border = ((32 - w_s) // 2, 0, (32 - w_s) // 2, 0)
        img = img.resize((w_s, h_s), Image.BILINEAR)  # BILINEAR  ANTIALIAS
        img = ImageOps.expand(img, border=border, fill=0)  # left,top,right,bottom

        # img = img.crop(border)
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.samples)


class CIFAR10Data(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.hparams = args
        self.mean = (0.4914, 0.4822, 0.4465)
        self.std = (0.2471, 0.2435, 0.2616)

    def download_weights(self):
        url = (
            "https://rutgers.box.com/shared/static/gkw08ecs797j2et1ksmbg1w5t3idf5r5.zip"
        )

        # Streaming, so we can iterate over the response.
        r = requests.get(url, stream=True)

        # Total size in Mebibyte
        total_size = int(r.headers.get("content-length", 0))
        block_size = 2 ** 20  # Mebibyte
        t = tqdm(total=total_size, unit="MiB", unit_scale=True)

        with open("state_dicts.zip", "wb") as f:
            for data in r.iter_content(block_size):
                t.update(len(data))
                f.write(data)
        t.close()

        if total_size != 0 and t.n != total_size:
            raise Exception("Error, something went wrong")

        print("Download successful. Unzipping file...")
        path_to_zip_file = os.path.join(os.getcwd(), "state_dicts.zip")
        directory_to_extract_to = os.path.join(os.getcwd(), "cifar10_models")
        with zipfile.ZipFile(path_to_zip_file, "r") as zip_ref:
            zip_ref.extractall(directory_to_extract_to)
            print("Unzip file successful!")

    def train_dataloader(self):
        transform = T.Compose(
            [
                T.RandomCrop(32, padding=4),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize(self.mean, self.std),
            ]
        )
        dataset = CIFAR10(root=self.hparams.data_dir, train=True, transform=transform)
        dataloader = DataLoader(
            dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=True,
            drop_last=True,
            pin_memory=True,
        )
        return dataloader

    def val_dataloader(self):
        transform = T.Compose(
            [
                T.ToTensor(),
                T.Normalize(self.mean, self.std),
            ]
        )
        dataset = CIFAR10(root=self.hparams.data_dir, train=False, transform=transform)
        dataloader = DataLoader(
            dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            drop_last=True,
            pin_memory=True,
        )
        return dataloader

    def test_dataloader(self):
        return self.val_dataloader()


class MyData(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.mean = (0.4914, 0.4822, 0.4465)
        self.std = (0.2471, 0.2435, 0.2616)
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

    def train_dataloader(self):
        transform = T.Compose(
            [
                T.RandomCrop(32, padding=4),
                # T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize(self.mean, self.std),
            ]
        )
        dataset = MultiLabelDataset(root=self.args.train, train=True, transform=transform)
        self.classes = dataset.classes
        dataloader = DataLoader(
            dataset,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            shuffle=True,
            drop_last=True,
            pin_memory=True,
        )
        return dataloader

    def val_dataloader(self):
        transform = T.Compose(
            [
                T.ToTensor(),
                T.Normalize(self.mean, self.std),
            ]
        )
        dataset = MultiLabelDataset(root=self.args.test, train=False, transform=transform)
        dataloader = DataLoader(
            dataset,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            drop_last=True,
            pin_memory=True,
        )
        return dataloader

    def test_dataloader(self):
        return self.val_dataloader()


if __name__ == '__main__':
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2471, 0.2435, 0.2616)
    transform = T.Compose(
        [
            T.ToTensor(),
            T.Normalize(mean, std),
        ]
    )
    dataset = MultiLabelDataset(root='/Users/qiuyurui/Projects/PycharmProjects/PyTorch_CIFAR10/test_data.txt',
                                train=False, transform=transform)
    test_loader = DataLoader(dataset=dataset, batch_size=1, shuffle=True, num_workers=0)
    print(dataset.__len__())
    for i, data in enumerate(tqdm(test_loader)):
        sample, target = data
        show_img(sample[0].numpy().transpose(1, 2, 0), title='img')
        print(target)
