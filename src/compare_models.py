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

# # test dataset class and dataloader
# test_npz = "../data/processed_npz_vit_b"
# test_dataset = NpzDataset(test_npz)
# test_dataloader = DataLoader(test_dataset, batch_size=8, shuffle=True)

# for img_embed, gt2D, bboxes in test_dataloader:
#     # img_embed: (B, 256, 64, 64), gt2D: (B, 1, 256, 256), bboxes: (B, 4)
#     print(f"{img_embed.shape=}, {gt2D.shape=}, {bboxes.shape=}")
#     break

# # set up model for fine-tuning
# npz_tr_path = "../data/processed_npz_vit_b"
# work_dir = "../model"
# task_name = "vitsam"
# prepare SAM model
model_type = "vit_b"
checkpoint = "../data/sam_vit_b_01ec64.pth"
finetuned_checkpoint="../model/vitsam/sam_model_best.pth"
device = "cuda"
# model_save_path = join(work_dir, task_name)
# os.makedirs(model_save_path, exist_ok=True)

sam_model = sam_model_registry[model_type](checkpoint=finetuned_checkpoint).to(device)

ori_sam_model = sam_model_registry[model_type](checkpoint=checkpoint).to(device)
ori_sam_predictor = SamPredictor(ori_sam_model)

ts_img_path ="../data/training_images/test/images"
ts_gt_path = "../data/training_images/test/labels"

test_names = sorted(os.listdir(ts_img_path))
gt_names = sorted(os.listdir(ts_gt_path))

img_idx = np.random.randint(len(test_names))
image_data = io.imread(join(ts_img_path, test_names[img_idx]))
gt_data = io.imread(join(ts_gt_path, gt_names[img_idx]))

def clean_image_channels(input_image):
    if (input_image.shape[-1] > 3 and len(input_image.shape) == 3):
        # remove alpha channe
        input_image = input_image[:,:,:3]

    if (len(input_image.shape) == 2):
        input_image = np.repeat(input_image[:, :, None], 3, axis=-1)
    
    return input_image

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

image_data = clean_image_channels(image_data)
bbox_raw = get_bbox_from_mask(gt_data)

gt_data = clean_image_channels(gt_data)

# preprocess
lower_bound, upper_bound = np.percentile(image_data, 0.5), np.percentile(image_data, 99.5)
image_data_pre = np.clip(image_data, lower_bound, upper_bound)
image_data_pre = (image_data_pre - np.min(image_data_pre)) / (np.max(image_data_pre) - np.min(image_data_pre)) * 255.0
image_data_pre[image_data==0] = 0
image_data_pre = np.uint8(image_data_pre)
H_img, W_img, _ = image_data_pre.shape

gt_data_pre = np.uint8(gt_data)
H_gt, W_gt, _ = gt_data_pre.shape

ori_sam_predictor.set_image(image_data_pre)
ori_sam_seg, _, _ = ori_sam_predictor.predict(point_coords = None, box=bbox_raw, multimask_output=False)

sam_transform = ResizeLongestSide(sam_model.image_encoder.img_size)
resize_img = sam_transform.apply_image(image_data_pre)
resize_gt = sam_transform.apply_image(gt_data_pre)

resize_img_tensor = torch.as_tensor(resize_img.transpose(2, 0, 1)).to(device)
resize_gt_tensor = torch.as_tensor(resize_gt.transpose(2, 0, 1)).to(device)

input_image = sam_model.preprocess(resize_img_tensor[None,:,:,:])
input_gt = sam_model.preprocess(resize_gt_tensor[None,:,:,:]).cpu()

with torch.no_grad():
    # precompute image embedding
    ts_img_embedding = sam_model.image_encoder(input_image)
    bbox = sam_transform.apply_boxes(bbox_raw, (H_img, W_img))

    print(f'{bbox_raw=} -> {bbox=}')

    box_torch = torch.as_tensor(bbox, dtype=torch.float, device=device)

    if (len(box_torch.shape) == 2):
        box_torch = box_torch[:, None, :]
    
    sparse_embeddings, dense_embeddings = sam_model.prompt_encoder(
        points=None,
        boxes=box_torch,
        masks=None,
    )

    medsam_seg_prob, _ = sam_model.mask_decoder(
        image_embeddings=ts_img_embedding.to(device),
        image_pe=sam_model.prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse_embeddings,
        dense_prompt_embeddings=dense_embeddings,
        multimask_output=False,
    )

    medsam_seg_prob = torch.sigmoid(medsam_seg_prob)

    medsam_seg_prob = medsam_seg_prob.cpu().numpy().squeeze()
    medsam_seg = (medsam_seg_prob > 0.5).astype(np.uint8)

ori_sam_dsc = compute_dice_coefficient(input_gt>0, ori_sam_seg>0)
medsam_dsc = compute_dice_coefficient(input_gt>0, medsam_seg>0)
print('Original SAM DSC: {:.4f}'.format(ori_sam_dsc), 'MedSAM DSC: {:.4f}'.format(medsam_dsc))

def show_mask(mask, ax, random_color=False):
    if (random_color):
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([251/255, 252/255, 30/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='blue', facecolor=(0,0,0,0), lw=2))    

_, axs = plt.subplots(1, 3, figsize=(25, 25))
axs[0].imshow(image_data)
show_mask(gt_data>0, axs[0])
# show_box(box_np[img_id], axs[0])
# axs[0].set_title('Mask with Tuned Model', fontsize=20)
axs[0].axis('off')

axs[1].imshow(image_data)
show_mask(ori_sam_seg, axs[1])
show_box(bbox_raw, axs[1])
# add text to image to show dice score
axs[1].text(0.5, 0.5, 'SAM DSC: {:.4f}'.format(ori_sam_dsc), fontsize=30, horizontalalignment='left', verticalalignment='top', color='yellow')
# axs[1].set_title('Mask with Untuned Model', fontsize=20)
axs[1].axis('off')

axs[2].imshow(image_data)
show_mask(medsam_seg, axs[2])
show_box(bbox_raw, axs[2])
# add text to image to show dice score
axs[2].text(0.5, 0.5, 'MedSAM DSC: {:.4f}'.format(medsam_dsc), fontsize=30, horizontalalignment='left', verticalalignment='top', color='yellow')
# axs[2].set_title('Ground Truth', fontsize=20)
axs[2].axis('off')
plt.show()  
plt.subplots_adjust(wspace=0.01, hspace=0)
# save plot
# plt.savefig(join(model_save_path, test_npzs[npz_idx].split('.npz')[0] + str(img_id).zfill(3) + '.png'), bbox_inches='tight', dpi=300)
plt.close()