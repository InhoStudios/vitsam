# Import libraries
import numpy as np
import os
from glob import glob
import pandas as pd

from os.path import join
from skimage import transform, io, segmentation
from tqdm import tqdm
import torch
from segment_anything import sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide

# set up global variables
img_path = "../data/training_images/train/images"
gt_path = "../data/training_images/train/labels"

npz_path = "../data/processed_npz"
data_name = "processed_data"

image_name_suffix = ".jpg"
label_name_suffix = ".png"

image_size = 256 # TODO: CROP IMAGES?
label_id = 255
model_type = "vit_b"
checkpoint = "../data/sam_vit_b_01ec64.pth"

device = "cuda:0"
seed = 2023

imgs = []
gts = []
img_embeddings = []

sam_model = sam_model_registry[model_type](checkpoint=checkpoint).to(device)

def process(gt_name: str, image_name: str):
    if image_name == None:
        image_name = gt_name.split(".")[0] + image_name_suffix
    gt_data = io.imread(join(gt_path, gt_name))

    # if the mask is RGB, select the first channel
    if len(gt_data.shape) == 3:
        gt_data = gt_data[:, :, 0]
    assert len(gt_data.shape) == 2, "ground truth should be 2D"

    # resize ground truth image
    gt_data = transform.resize(
        gt_data == label_id,
        (image_size, image_size),
        order=0,
        preserve_range=True,
        mode="constant"
    )

    # convert image to uint8
    gt_data = np.uint8(gt_data)

    # exclude small objects
    if (np.sum(gt_data) > 100):
        # check that ground truth only has two possible values
        assert(
            np.max(gt_data) == 1 and np.unique(gt_data).shape[0] == 2
        ), "ground truth should be binary"
        # if assertion fails, rescale binary mask to 1

        # read in image file
        image_data = io.imread(join(img_path, image_name))

        # remove alpha channel
        if (image_data.shape[-1] > 3 and len(image_data.shape) == 3):
            image_data = image_data[:, :, :3]
        
        # if the image is greyscale, repeat the last channel to convert it to RGB
        if (len(image_data.shape) == 2):
            image_data = np.repeat(image_data[:, :, None], 3, axis=-1)
        
        # scale image to 50th and 99.5th percentile

        # rescale image
        image_data = transform.resize(
            image_data,
            (image_size, image_size),
            order=3,
            preserve_range=True,
            mode="constant",
            anti_aliasing=True
        )
        image_data = np.uint8(image_data)

        imgs.append(image_data)

        assert np.sum(gt_data) > 100, "ground truth should have more than 100 pixels"

        gts.append(gt_data)

        sam_transform = ResizeLongestSide(sam_model.image_encoder.img_size)
        resize_img = sam_transform.apply_image(image_data)
        resize_img_tensor = torch.as_tensor(resize_img.transpose(2, 0, 1)).to(device)

        input_image = sam_model.preprocess(
            resize_img_tensor[None, :, :, :]
        )

        assert input_image.shape == (
            1,
            3,
            sam_model.image_encoder.img_size,
            sam_model.image_encoder.img_size,
        ), "input image should be resized by 1024 * 1024"

        with torch.no_grad():
            embedding = sam_model.image_encoder(input_image)
            img_embeddings.append(embedding.cpu().numpy()[0])

def preprocess_and_save():
    names = sorted(os.listdir(gt_path))
    print("image number:", len(names))
    for gt_name in tqdm(names):
        process(gt_name, None)

    save_path = npz_path + "_" + model_type
    os.makedirs(save_path, exist_ok=True)

    # save images as one npz file
    print("num images: ", len(imgs))

    if (len(imgs) > 1):
        imgs = np.stack(imgs, axis=0)
        gts = np.stack(gts, axis=0)

        img_embeddings = np.stack(img_embeddings, axis=0)

        np.savez_compressed(
            join(save_path, data_name + ".npz"),
            imgs=imgs,
            gts=gts,
            img_embeddings=img_embeddings,
        )
        # save example image for sanity check
        idx = np.random.randint(imgs.shape[0])
        img_idx = imgs[idx, :, :, :]
        gt_idx = gts[idx, :, :]
        bd = segmentation.find_boundaries(gt_idx, mode="inner")
        img_idx[bd, :] = (255, 0, 0)
        io.imsave(save_path + ".png", img_idx, check_contrast=False)
    
    return imgs, gts, img_embeddings

preprocess_and_save()