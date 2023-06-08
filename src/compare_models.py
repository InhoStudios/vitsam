# imports
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
from npzdataset import NpzDataset

from skimage import io


torch.manual_seed(2023)
np.random.seed(2023)

# test dataset class and dataloader
test_npz = "../data/processed_npz_vit_b"
test_dataset = NpzDataset(test_npz)
test_dataloader = DataLoader(test_dataset, batch_size=8, shuffle=True)

for img_embed, gt2D, bboxes in test_dataloader:
    # img_embed: (B, 256, 64, 64), gt2D: (B, 1, 256, 256), bboxes: (B, 4)
    print(f"{img_embed.shape=}, {gt2D.shape=}, {bboxes.shape=}")
    break


# set up model for fine-tuning
npz_tr_path = "../data/processed_npz_vit_b"
work_dir = "../model"
task_name = "vitsam"
# prepare SAM model
model_type = "vit_b"
checkpoint = "../data/sam_vit_b_01ec64.pth"
device = "cuda:0"
model_save_path = join(work_dir, task_name)
os.makedirs(model_save_path, exist_ok=True)

ori_sam_model = sam_model_registry[model_type](checkpoint=checkpoint).to(device)
ori_sam_predictor = SamPredictor(ori_sam_model)

ts_img_path = "../data/training_images/test/images"
ts_gt_path = "../data/training_images/test/labels"
test_names = sorted(os.listdir(ts_img_path))

img_idx = np.random.randint(len(test_names))
image_data = io.imread(join(ts_img_path, test_names[img_idx]))
if image_data.shape[-1] > 3 and len(image_data.shape) == 3:
    image_data = image_data[:, :, :3]
if len(image_data.shape) == 2:
    image_data = np.repeat(image_data[:, :, None], 3, axis=-1)

def get_bbox_from_mask(mask):
    y_indices, x_indices = np.where(mask > 0)
    x_min, x_max = np.min(x_indices), np.max(x_indices)
    y_min, y_max = np.min(y_indices), np.max(y_indices)

    H, W = mask.shape
    x_min = max(0, x_min - np.random.randint(0, 20))
    x_max = min(W, x_max + np.random.randint(0, 20))
    y_min = max(0, y_min - np.random.randint(0, 20))
    y_max = min(H, y_max + np.random.randint(0, 20))

    return np.array([x_min, y_min, x_max, y_max])

gt_data = io.imread(join(ts_gt_path, test_names(img_idx)))
bbox_raw = get_bbox_from_mask(gt_data)

lower_bound, upper_bound = np.percentile(image_data, 0.5), np.percentile(image_data, 99.5)
image_data_pre = np.clip(image_data, lower_bound, upper_bound)
image_data_pre = (image_data_pre - np.min(image_data_pre))/(np.max(image_data_pre)-np.min(image_data_pre))*255.0
image_data_pre[image_data==0] = 0
image_data_pre = np.uint8(image_data_pre)
H, W, _ = image_data_pre.shape

# predict the segmentation mask using the original SAM model
ori_sam_predictor.set_image(image_data_pre)
ori_sam_seg, _, _ = ori_sam_predictor.predict(point_coords=None, box=bbox_raw, multimask_output=False)

sam_transform = ResizeLongestSide(sam_model.image_encoder.img_size)