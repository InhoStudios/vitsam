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

from skimage import io, transform

torch.manual_seed(2023)
np.random.seed(2023)

# GET DICE LOSS COEFF
seg_loss = monai.losses.DiceLoss(sigmoid=True, squared_pred=True, reduction='mean')

# CONSTANTS
model_type = "vit_b"
checkpoint = "../data/sam_vit_b_01ec64.pth"
finetuned_checkpoint="../model/vitsam/sam_model_best.pth"
device = "cuda"

ts_img_path ="../data/training_images/test/images"
ts_gt_path = "../data/training_images/test/labels"

image_name_suffix = ".jpg"
label_name_suffix = ".png"

image_size = 256 
label_id = 255

default_sam = sam_model_registry[model_type](checkpoint=checkpoint)
default_sam_predictor = SamPredictor(default_sam)
finetuned_sam = sam_model_registry[model_type](checkpoint=finetuned_checkpoint)

def create_annotation_mask(anns):
    if len(anns) == 0:
        return
    
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0

def get_bbox_from_mask(mask):
    '''Returns a bounding box from a mask'''
    y_indices, x_indices = np.where(mask > 0)
    x_min, x_max = np.min(x_indices), np.max(x_indices)
    y_min, y_max = np.min(y_indices), np.max(y_indices)
    # add perturbation to bounding box coordinates
    H, W = mask.shape
    x_min = max(0, x_min - np.random.randint(0, 20))
    x_max = min(W, x_max + np.random.randint(0, 20))
    y_min = max(0, y_min - np.random.randint(0, 20))
    y_max = min(H, y_max + np.random.randint(0, 20))

    return np.array([x_min, y_min, x_max, y_max])

def get_clean_image_gt_pair(gt_name: str, image_name: str):
    if (image_name == None):
        image_name = gt_name.split(".")[0] + image_name_suffix
    
    # ground truth masks
    gt_data = io.imread(join(ts_gt_path, gt_name))

    if len(gt_data.shape) == 3:
        gt_data = gt_data[:, :, 0]
    assert len(gt_data.shape) == 2, "ground truth should be 2D"

    gt_data = transform.resize(
        gt_data == label_id,
        (image_size, image_size),
        order=0,
        preserve_range=True,
        mode="constant"
    )
    gt_data = np.uint8(gt_data)

    # images
    im_data = io.imread(join(ts_img_path, image_name))

    if (im_data.shape[-1] > 3 and len(im_data.shape) == 3):
        im_data = im_data[:, :, :3]
    
    if (len(im_data.shape) == 2):
        im_data = np.repeat(im_data[:, :, None], 3, axis=-1)
    
    im_data = transform.resize(
        im_data,
        (image_size, image_size),
        order=3,
        preserve_range=True,
        mode="constant",
        anti_aliasing=True
    )
    im_data = np.uint8(im_data)
    
    return im_data, gt_data

def predict_image(image, mask):
    default_sam_predictor.set_image(image)
    bbox_raw = get_bbox_from_mask(mask)
    default_seg, _, _ = default_sam_predictor.predict(point_coords = None, box=bbox_raw, multimask_output=False)

    sam_transform = ResizeLongestSide(finetuned_sam.image_encoder.img_size)
    resize_im = sam_transform.apply_image(image)
    resize_im_tensor = torch.as_tensor(resize_im.transpose(2, 0, 1)).to(device)
    input_im = finetuned_sam.preprocess(resize_im_tensor[None,:,:,:])

    H, W, _ = image.shape

    with torch.no_grad():
        ts_img_embedding = finetuned_sam.image_encoder(input_im)
        bbox = sam_transform.apply_boxes(bbox_raw, (H, W))

        box_torch = torch.as_tensor(bbox, dtype=torch.float, device=device)

        if (len(box_torch.shape == 2)):
            box_torch = box_torch[:, None, :]

        sparse_embeddings, dense_embeddings = finetuned_sam.prompt_encoder(
            points=None,
            boxes=box_torch,
            masks=None
        )

        medsam_seg_prob, _ = finetuned_sam.mask_decoder(
            image_embeddings=ts_img_embedding.to(device),
            image_pe=finetuned_sam.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
        )

        medsam_seg_prob = torch.sigmoid(medsam_seg_prob)

        medsam_seg_prob = medsam_seg_prob.cpu().numpy().squeeze()
        medsam_seg = (medsam_seg_prob > 0.5).astype(np.uint8)
    
    return default_seg, medsam_seg

def get_dice_score(pred_mask, ground_truth):
    seg_loss = monai.losses.DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')
    loss = seg_loss(pred_mask, torch.tensor(ground_truth[None, :, :]).long())
    return loss.item()

def get_jaccard_score(pred_mask, ground_truth):
    seg_loss = monai.losses.DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')
    loss = seg_loss(pred_mask, torch.tensor(ground_truth[None, :, :]).long(), jaccard=True)
    return loss.item()

if (__name__ == "__main__"):
    gt_names = sorted(os.listdir(ts_gt_path))
    finetuned_dice = 0
    finetuned_jaccard = 0
    def_dice = 0
    def_jaccard = 0
    for gt_path in gt_names:
        im, gt = get_clean_image_gt_pair(gt_path, None)
        def_seg, ft_seg = predict_image(im, gt)
        finetuned_dice += get_dice_score(ft_seg, gt)
        finetuned_jaccard += get_jaccard_score(ft_seg, gt)
        def_dice += get_dice_score(def_seg, gt)
        def_jaccard += get_jaccard_score(def_jaccard, gt)
    
    print(f"For all test and train images: \n",
        "Default ViT-B Model: {def_jaccard} (Jaccard), {def_dice} (Dice)\n",
        "Finetuned: {finetuned_jaccard} (Jaccard), {finetuned_dice} (Dice)")
