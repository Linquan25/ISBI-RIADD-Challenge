from face_alignment.utils import transform
from numpy.core.fromnumeric import size
from scipy.ndimage.measurements import label
from torch.utils.data import DataLoader, Dataset, random_split
import pandas as pd
import numpy as np
import skimage.io as sio
from skimage.color import gray2rgb
from torchvision import transforms
from scipy.stats import stats
import torch
import os
from torchvision.transforms.functional import scale

from torchvision.transforms.transforms import RandomResizedCrop, RandomRotation
import config

torch.manual_seed(63)

def contrast_strech(img):
    imgori=img.copy()
    img=img.astype(np.float32)

    imgs = img.flatten()

    z = np.abs(stats.zscore(imgs))
    threshold = 2.5

    imgs = imgs[np.where(z <= threshold)]
    norm_v=(np.max(imgs) - np.min(imgs))
    if norm_v>0:
        imgnew = (img - np.min(imgs)) / norm_v
        #print (np.min(imgnew),np.max(imgnew))
        imgnew[imgnew <=0] = 0
        imgnew[imgnew >= 1] = 1
        imgnew=imgnew * 255
    else:
        imgnew=imgori
    imgnew=np.asarray(imgnew,dtype=np.uint8)
    return imgnew

class ISBIDataset(Dataset):
    def __init__(self, csv_path, img_path, weight_csv, testing=False, input_size=244 ) -> None:
        super().__init__()
        TRAIN_TRANSFORMS = transforms.Compose(
            [
                transforms.Resize((input_size,input_size)),
                #transforms.Resize((250,250)),
                transforms.RandomHorizontalFlip(0.5),
                transforms.RandomVerticalFlip(0.5),
                transforms.RandomRotation(degrees=(0,15)),
                #transforms.RandomResizedCrop(size=(224,224),scale=(0.8,1.2), ratio=(0.999,1.001)),
                #transforms.Resize((224,224)),
                transforms.ToTensor(),
                #transforms.ColorJitter(brightness=0.2, contrast=0.1, saturation=0.1, hue=0.1), 
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),    
            ]
        )

        EVALUATION_TRANSFORMS = transforms.Compose(
            [
                transforms.Resize((input_size,input_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),    
            ]
        )
        self.df = pd.read_csv(csv_path, header=0)
        self.img_path = img_path
        self.preprocess = EVALUATION_TRANSFORMS if testing else TRAIN_TRANSFORMS
        self.testing = testing
        self.weight = pd.read_csv(weight_csv, header=0) if weight_csv else 1
    
    def __getitem__(self, index):
        img_id = self.df.iloc[index][0]
        path = os.path.join(self.img_path, str(img_id) + ".png")
        input_image = sio.imread(path)   
        #input_image = contrast_strech(input_image)
        if input_image.shape[1] == 4288:
            input_image = transforms.ToPILImage()(input_image)
            input_image = transforms.functional.affine(input_image, angle=.0, scale=1,shear=0,translate = [175,0])
            input_image = transforms.CenterCrop((2848, 3423))(input_image)
            input_tensor = self.preprocess(input_image)
        elif input_image.shape[1] == 2144:
            input_image = transforms.ToPILImage()(input_image)
            input_image = transforms.CenterCrop(1424)(input_image)
            input_tensor = self.preprocess(input_image)
        else:
            input_image = transforms.ToPILImage()(input_image)
            input_image = transforms.CenterCrop(1536)(input_image)
            input_tensor = self.preprocess(input_image)
        
        label = self.df.iloc[index][1:].to_list()
        label = torch.tensor(label)
        if len(label)>29:
            label = torch.cat((label[0:28],torch.tensor([1])),0) if label[28:].sum()>0 else torch.cat((label[0:28],torch.tensor([0])),0)
        if self.testing:
            return input_tensor, label, 1
        else:
            weight = self.getWeight(label)
            return input_tensor, label, weight
    
    def __len__(self):
        return len(self.df)
    
    def getWeight(self, label):
        weight = 0
        for i, n in enumerate(label):
            if n==1 and i!=0:
                weight += self.weight.iloc[i][0]
        if weight ==0:
            return self.weight.iloc[0][0]
        else:
            return weight
        