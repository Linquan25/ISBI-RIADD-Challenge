from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import torch

torch.manual_seed(63)

class ensembleDataset(Dataset):
    def __init__(self, label_csv, prediction_csv) -> None:
        super().__init__()
        self.label_df = pd.read_csv(label_csv, header=0)
        self.prediction_df = pd.read_csv(prediction_csv, header=0)
    
    def __getitem__(self, index):
        input = self.prediction_df.iloc[index].to_list()
        input = torch.tensor(input).float()
        label = self.label_df.iloc[index].to_list()
        label = torch.tensor(label).float()
        return input, label
    
    def __len__(self):
        return len(self.label_df)
        