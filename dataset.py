import numpy as np
import os
import pandas as pd
import torch
import cv2
from torch.utils.data import Dataset

class FacesDataset(Dataset):
    def __init__(self, csv_file):
        self.annotations=pd.read_csv(csv_file)
    
    def __len__(self):
        return len(self.annotations)   #number of data items
    
    def __getitem__(self, index):
        pxs = self.annotations.iloc[index, 4]
        pxs = pxs.split(' ')
        pxs = [int(i) for i in pxs]
#         print(type(pxs))
#         print(pxs)
        img = np.reshape(pxs, (48, 48))
#         cv2.imwrite("test" + str(index) + ".png", img)
        
        y_label = torch.tensor([float(self.annotations.iloc[index, 0]), float(self.annotations.iloc[index, 1]), float(self.annotations.iloc[index, 2])])
#         y_label = torch.tensor([self.annotations.iloc[index, 0], self.annotations.iloc[index, 1], self.annotations.iloc[index, 2]], dtype=torch.int32)
        
        
        return (img, y_label)