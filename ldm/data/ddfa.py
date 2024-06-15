import os
import numpy as np
from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
import torchvision.transforms as transforms
import albumentations
from ldm.data.base import DDFAInFerPaths

class DDFAInferBase(Dataset):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.data = None
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]
    



class DDFAInfer(DDFAInferBase):
    def __init__(self, size, images_list_file, lmk_dir):
        super().__init__()
        with open(images_list_file, "r") as f:
            paths = f.read().splitlines()

        self.data = DDFAInFerPaths(paths=paths, size=size, lmk_dir=lmk_dir, random_crop=False)
    

