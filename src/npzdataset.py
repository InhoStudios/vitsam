import numpy as np
import matplotlib.pyplot as plt
import os
from os.path import join
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
import monai
from segment_anything import SamPredictor, sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide
from utils.SurfaceDice import compute_dice_coefficient

class NpzDataset(Dataset):
    def __init__(self, data_root):
        self.data_root = data_root
        self.npz_files = sorted(os.listdir(self.data_root))
        self.npz_data = [np.load(join(data_root, f)) for f in self.npz_files]

        self.ori_gts = np.vstack([d['gts'] for d in self.npz_data])
        self.img_embeddings = np.vstack([d['img_embeddings'] for d in self.npz_data])
    
    def __len__(self):
        return self.ori_gts.shape[0]
    
    def __getitem__(self, index):
        img_embed = self.img_embeddings[index]
        gt2D = self.ori_gts[index]

        y_indices, x_indices = np.where(gt2D > 0)
        x_min, x_max = np.min(x_indices), np.max(x_indices)
        y_min, y_max = np.min(y_indices), np.max(y_indices)

        H, W = gt2D.shape

        x_min = max(0, x_min - np.random.randint(0, 20))
        x_max = min(W, x_max + np.random.randint(0, 20))
        y_min = max(0, y_min - np.random.randint(0, 20))
        y_max = min(H, y_max + np.random.randint(0, 20))
        bboxes = np.array([x_min, y_min, x_max, y_max])

        return torch.tensor(img_embed).float(), \
            torch.tensor(gt2D[None, :, :]).long(), \
            torch.tensor(bboxes).float()

class MultispaceDataset(Dataset):
    def __init__(self, data_root, is_hsv=False):
        self.data_root = data_root
        self.npz_files = sorted(os.listdir(self.data_root))
        if (is_hsv):
            self.npz_data = [np.load(join(data_root, f)) for f in self.npz_files if f.startswith("hsv_")]
        else:
            self.npz_data = [np.load(join(data_root, f)) for f in self.npz_files if not f.startswith("hsv_")]

        self.ori_gts = np.vstack([d['gts'] for d in self.npz_data])
        self.img_embeddings = np.vstack([d['img_embeddings'] for d in self.npz_data])
    
    def __len__(self):
        return self.ori_gts.shape[0]
    
    def __getitem__(self, index):
        img_embed = self.img_embeddings[index]
        gt2D = self.ori_gts[index]

        y_indices, x_indices = np.where(gt2D > 0)
        x_min, x_max = np.min(x_indices), np.max(x_indices)
        y_min, y_max = np.min(y_indices), np.max(y_indices)

        H, W = gt2D.shape

        x_min = max(0, x_min - np.random.randint(0, 20))
        x_max = min(W, x_max + np.random.randint(0, 20))
        y_min = max(0, y_min - np.random.randint(0, 20))
        y_max = min(H, y_max + np.random.randint(0, 20))
        bboxes = np.array([x_min, y_min, x_max, y_max])

        return torch.tensor(img_embed).float(), \
            torch.tensor(gt2D[None, :, :]).long(), \
            torch.tensor(bboxes).float()
        
