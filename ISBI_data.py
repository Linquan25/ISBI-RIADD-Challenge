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

TRAIN_TRANSFORMS = transforms.Compose(
    [
        transforms.Resize((244,244)),
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
        transforms.Resize((244,244)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),    
    ]
)

TRAIN_TRANSFORMS_EFF = transforms.Compose(
    [
        transforms.Resize((528,528)),  #B0->224 B1->240 B2->260 B3->300 B4->380 B5->456 B6->528 B7->600
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomVerticalFlip(0.5),
        #resize(250,250)->scale(0.8-1.2)->crop(224,224)
        #contrast norm
        #rotation, blur(smoothing), scale(0.8-1.2)
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),    
    ]
)

EVALUATION_TRANSFORMS_EFF = transforms.Compose(
    [
        transforms.Resize((528,528)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),    
    ]
)

class ISBIDataset(Dataset):
    def __init__(self, csv_path, img_path, testing=False, reweight=True) -> None:
        super().__init__()

        self.df = pd.read_csv(csv_path, header=0)
        self.img_path = img_path
        self.preprocess = EVALUATION_TRANSFORMS if testing else TRAIN_TRANSFORMS
        self.testing = testing
        self.reweight = reweight
        self.weight = self.weightCalculation()
    
    def __getitem__(self, index):
        img_id = self.df.iloc[index][0]
        path = os.path.join(self.img_path, str(img_id) + ".png")
        input_image = sio.imread(path)   
        #input_image = contrast_strech(input_image)
        if input_image.shape[1] == 4288:
            input_image = transforms.ToPILImage()(input_image)
            input_image = transforms.functional.affine(input_image, angle=.0, scale=1,shear=0,translate = [175,0])
            input_image = transforms.CenterCrop((2848, 3423))(input_image) #(3423)
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
        label = torch.tensor(label).long()
        if len(label)>29:
            label = torch.cat((label[0:28],torch.tensor([1])),0) if label[28:].sum()>0 else torch.cat((label[0:28],torch.tensor([0])),0)
        ### inverse ###
        #label[0] = 1-label[0]
        if self.reweight: 
            if self.testing:
                return input_tensor, label, 1
            else:
                return input_tensor, label, self.getWeight(label)
        else:
            return input_tensor, label
    
    def __len__(self):
        return len(self.df)
            
    def weightCalculation(self):
        data = self.df.values[:,1:]
        c = np.zeros((data.shape[1],))
        for i in data[:]:
            for j in range(i.shape[0]):
                if i[j]==1:
                    c[j]+=1
        c[0] = data.shape[0] - c[0]
        w = np.zeros_like(c)
        for i in range(w.shape[0]):
            w[i] = np.sum(c)/c[i]
        w = w/np.min(w)
        return w
    
    def getWeight(self, label):
        weight = 0
        count = 0
        for i, n in enumerate(label):
            if n==1:
                weight += self.weight[i]
                count += 1
        if count ==0:
            return self.weight[0]
        else:
            return weight/count
        