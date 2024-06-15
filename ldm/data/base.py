from abc import abstractmethod
from torch.utils.data import Dataset, ConcatDataset, ChainDataset, IterableDataset
import albumentations
from PIL import Image
import torch
import numpy as np
from third_party import Landmark68_API
from .preprocess_func import (extract_lm5_from_lm68, POS, resize_crop_img)
from scipy.io import loadmat
import os
import cv2
from skimage import transform as trans
import random
import math
from copy import deepcopy

def np2pillow(img, src_range=255.):
    coef = 255. / src_range
    return Image.fromarray(np.squeeze(np.clip(np.round(img * coef), 0, 255).astype(np.uint8)))


def pillow2np(img, dst_range=255.):
    coef = dst_range / 255.
    return np.asarray(img, np.float32) * coef

def randomErasing(img, sl=0.02, sh=0.1, r1=0.3, dst=255.):
    area = img.shape[0] * img.shape[1]
 
    target_area = random.uniform(sl, sh) * area
    aspect_ratio = random.uniform(r1, 1 / r1)
 
    h = int(round(math.sqrt(target_area * aspect_ratio)))
    w = int(round(math.sqrt(target_area / aspect_ratio)))
    if w < img.shape[1] and h < img.shape[0]:
        x1 = random.randint(0, img.shape[0] - h)
        y1 = random.randint(0, img.shape[1] - w)
        img[x1:x1+h, y1:y1+w, :] = np.random.rand(h, w, 3)*dst
    return img

def randomRotate(img, scale = np.pi/15):
    theta = (random.uniform(0, 1) * 2 - 1) * scale
    M = np.zeros([2, 3])
    M[0, 0] = np.cos(theta)
    M[0, 1] = -np.sin(theta)
    M[1, 0] = np.sin(theta)
    M[1, 1] = np.cos(theta)
    img = cv2.warpAffine(img, M, (img.shape[1],img.shape[0]))
    return img, M[:2, :2]
    
def estimate_norm(lm_68p, H):
    lm = extract_lm5_from_lm68(lm_68p)
    lm[:, -1] = H - 1 - lm[:, -1]
    tform = trans.SimilarityTransform()
    src = np.array(
    [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
     [41.5493, 92.3655], [70.7299, 92.2041]],
    dtype=np.float32)
    tform.estimate(lm, src)
    M = tform.params
    if np.linalg.det(M) == 0:
        M = np.eye(3)

    return M[0:2, :]



class DDFAInFerPaths(Dataset):
    def __init__(self, paths, lmk_dir ,size=None, random_crop=False, labels=None, 
                 lm_detector_path='third_party/lm_model/68lm_detector.pb', 
                 mtcnn_path='third_party/mtcnn_model/mtcnn_model.pb',
                 lm68_3d_path='BFM/similarity_Lm3D_all.mat',
                 random_noise=True,
                 random_rotate=True,
                 rescale_factor=102.):
        self.size = size
        self.rescale_factor = rescale_factor
        self.random_crop = random_crop
        self.lm68_model = Landmark68_API(lm_detector_path=lm_detector_path, mtcnn_path=mtcnn_path)
        self.lm68_3d = loadmat(lm68_3d_path)['lm']
        self.labels = dict() if labels is None else labels
        self.labels["file_path_"] = paths
        self._length = len(paths)
        self.lmk_dir = lmk_dir
        self.random_noise = random_noise
        self.random_rotate = random_rotate

    def __len__(self):
        return self._length

    def preprocess_image(self, image_path):
        image_ori = Image.open(image_path)
        if not image_ori.mode == "RGB":
            image_ori = image_ori.convert("RGB")
        image = np.array(image_ori).astype(np.float32)
        
        lm68_2d = self.lm68_model(image)
        basename = os.path.basename(image_path)[:-4]

        if lm68_2d is None:
            return None, None, None, None, None, None
        lm68_2d = lm68_2d.astype(np.float32)

        image = np2pillow(image)
        w0, h0 = image.size
        
        lm5_2d = extract_lm5_from_lm68(lm68_2d)
        lm5_3d = extract_lm5_from_lm68(self.lm68_3d)
        
        t, s = POS(lm5_2d, lm5_3d)
        s = self.rescale_factor / s
        trans_params = np.array([w0, h0, s, t[0], t[1], self.size]) 
        image, lm = resize_crop_img(image, lm68_2d, trans_params)
        trans_params = np.array([w0, h0, s, t[0].item(), t[1].item(), self.size])
        image = pillow2np(image)
        image_ori = pillow2np(image_ori)
        image = (image/127.5 - 1.0).astype(np.float32)   
        image_ori = (image_ori/127.5 - 1.0).astype(np.float32)    
        return image, image_ori, trans_params, basename

    def __getitem__(self, i):
        image, image_ori ,trans_params, basename = self.preprocess_image(self.labels["file_path_"][i])
        return image, image_ori, trans_params, basename
      

