import os
import random

import pandas as pd
import skimage.io as sio
import torch
import torch.utils.data
import torchvision.transforms.functional as TF
from skimage.color import gray2rgb
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
import numpy as np
from PIL import Image
from scipy.ndimage import convolve1d
from scipy.ndimage import gaussian_filter1d
from scipy.signal.windows import triang

import config
from align import FaceAligner

AGE_MEAN, AGE_STD = 38.7001, 12.8755
AGE_MAX = 81

TRAIN_TRANSFORMS = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.Resize(config.RES),
        transforms.RandomHorizontalFlip(0.5),
        transforms.ToTensor(),
        transforms.RandomErasing(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

EVAL_TRANSFORMS = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.Resize(config.RES),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def normalize(age, method="z"):
    if method == "z":
        return (age - AGE_MEAN) / AGE_STD
    else:
        return age / AGE_MAX


def denormalize(age):
    return age * AGE_STD + AGE_MEAN


def no_align(x):
    return x


class GTADataset(Dataset):
    def _no_align(self, x):
        return x

    def __init__(
        self, csv_path, img_path, transform=None, align=False, return_paths=False
    ):
        super().__init__()

        if type(csv_path) == str:
            self.df = pd.read_csv(csv_path, header=None)
        else:
            self.df = csv_path

        if align:
            self.aligner = FaceAligner(256)
        else:
            self.aligner = no_align

        self.df.columns = ["image", "label"]
        add_img_dir = lambda x: os.path.join(img_path, x)
        self.df["image"] = self.df["image"].apply(add_img_dir)

        if transform is None:
            # Preprocesing
            self.preprocess = TRAIN_TRANSFORMS
        else:
            self.preprocess = transform

        self.return_paths = return_paths

    def __getitem__(self, index: int):
        img_path, label = self.df.iloc[index].to_list()
        input_image = sio.imread(img_path)
        if len(input_image.shape) == 2:
            input_image = gray2rgb(input_image)
        input_image = self.aligner(input_image)
        input_tensor = self.preprocess(input_image)
        if self.return_paths:
            return input_tensor, torch.tensor(label).long(), img_path
        else:
            return input_tensor, torch.tensor(label).long()

    def __len__(self):
        return self.df.shape[0]


def train_test_split(dataset, train_ratio=0.8):
    train_size = int(train_ratio * len(dataset))
    test_size = len(dataset) - train_size
    trainset, testset = random_split(dataset, [train_size, test_size])
    return trainset, testset


def train_test_split_loaders(
    dataset, train_ratio=0.8, batch_size=128, num_workers=8, drop_last=True
):

    trainset, testset = train_test_split(dataset, train_ratio)

    trainloader = DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=drop_last,
    )

    testloader = DataLoader(
        testset, batch_size=batch_size, num_workers=num_workers, drop_last=drop_last
    )

    return trainloader, testloader


class SelfSupervisedDataset(GTADataset):
    def __init__(self, csv_path, img_path) -> None:
        super().__init__(csv_path, img_path)
        self.rotations = [0, 90, 180, 270]

    def __getitem__(self, index: int):
        img_path, _ = self.df.iloc[index].to_list()
        input_image = sio.imread(img_path)
        if len(input_image.shape) == 2:
            input_image = gray2rgb(input_image)

        rotation = torch.randint(0, 4)
        input_tensor = TF.to_pil_image(input_image)
        input_tensor = TF.rotate(input_tensor, self.rotations[rotation])

        if self.return_paths:
            return input_tensor, torch.tensor(rotation).long(), img_path
        else:
            return input_tensor, torch.tensor(rotation).long()


class BalancedBatchSampler(torch.utils.data.sampler.Sampler):
    def __init__(self, dataset, labels=None):
        self.labels = labels
        self.dataset = dict()
        self.balanced_max = 0
        # Save all the indices for all the classes
        for idx in range(0, len(dataset)):
            label = self._get_label(dataset, idx)
            if label not in self.dataset:
                self.dataset[label] = list()
            self.dataset[label].append(idx)
            self.balanced_max = (
                len(self.dataset[label])
                if len(self.dataset[label]) > self.balanced_max
                else self.balanced_max
            )

        # Oversample the classes with fewer elements than the max
        for label in self.dataset:
            while len(self.dataset[label]) < self.balanced_max:
                self.dataset[label].append(random.choice(self.dataset[label]))
        self.keys = list(self.dataset.keys())
        self.currentkey = 0
        self.indices = [-1] * len(self.keys)

    def __iter__(self):
        while self.indices[self.currentkey] < self.balanced_max - 1:
            self.indices[self.currentkey] += 1
            yield self.dataset[self.keys[self.currentkey]][
                self.indices[self.currentkey]
            ]
            self.currentkey = (self.currentkey + 1) % len(self.keys)
        self.indices = [-1] * len(self.keys)

    def _get_label(self, dataset, idx):
        if self.labels is not None:
            return self.labels[idx].item()
        else:
            try:
                return dataset[idx][1]
            except:
                raise Exception(
                    "You should pass the tensor of labels to the "
                    + "constructor as second argument"
                )

    def __len__(self):
        return self.balanced_max * len(self.keys)
    
class WeightedGTADataset(Dataset):
    def __init__(self, csv_path, img_path, transform=None) -> None:
        super().__init__()
        if type(csv_path) == str:
            self.df = pd.read_csv(csv_path, header=None)
        else:
            self.df = csv_path
            
        self.df.columns = ["image", "label"]
        add_img_dir = lambda x: os.path.join(img_path, x)
        self.df["image"] = self.df["image"].apply(add_img_dir)
        self.weights = self._prepare_weights()

        if transform is None:
            # Preprocesing
            self.preprocess = transforms.Compose(
                [
                    transforms.Resize(224),
                    transforms.CenterCrop(224),
                    transforms.RandomHorizontalFlip(0.5),
                    transforms.ToTensor(),
                    transforms.RandomErasing(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )

        else:
            self.preprocess = transform

    def __getitem__(self, index: int):
        img_path, label = self.df.iloc[index].to_list()
        input_image = Image.open(img_path).convert("RGB")
        input_tensor = self.preprocess(input_image)
        weight = np.asarray([self.weights[index]]).astype('float32') if self.weights is not None else np.asarray([np.float32(1.)])
        return input_tensor, torch.tensor(label).long(), weight

    def __len__(self):
        return self.df.shape[0]
    
    def _prepare_weights(self):
        max_target=81
        lds_ks=5
        lds_sigma=2
        lds_kernel='gaussian'
        value_dict = {x: 0 for x in range(max_target)}
        labels = self.df['label'].values
        for label in labels:
            value_dict[max(0, int(label)-1)] += 1
        value_dict = {k: np.sqrt(v) for k, v in value_dict.items()}
        num_per_label = [value_dict[max(0, int(label)-1)] for label in labels]

        lds_kernel_window = get_lds_kernel_window(lds_kernel, lds_ks, lds_sigma)
        smoothed_value = convolve1d(
            np.asarray([v for _, v in value_dict.items()]), weights=lds_kernel_window, mode='constant')
        num_per_label = [smoothed_value[max(0, int(label)-1)] for label in labels]

        weights = [np.float32(1 / x) for x in num_per_label]
        scaling = len(weights) / np.sum(weights)
        weights = [scaling * x for x in weights]
        return weights
    
def get_lds_kernel_window(kernel, ks, sigma):
    assert kernel in ['gaussian', 'triang', 'laplace']
    half_ks = (ks - 1) // 2
    if kernel == 'gaussian':
        base_kernel = [0.] * half_ks + [1.] + [0.] * half_ks
        kernel_window = gaussian_filter1d(base_kernel, sigma=sigma) / max(gaussian_filter1d(base_kernel, sigma=sigma))
    elif kernel == 'triang':
        kernel_window = triang(ks)
    else:
        laplace = lambda x: np.exp(-abs(x) / sigma) / (2. * sigma)
        kernel_window = list(map(laplace, np.arange(-half_ks, half_ks + 1))) / max(map(laplace, np.arange(-half_ks, half_ks + 1)))

    return kernel_window