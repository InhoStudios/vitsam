# Import libraries
import numpy as np
import os
from glob import glob
import pandas as pd
import random

from os.path import join
from skimage import transform, io, segmentation
from tqdm import tqdm
import torch
from segment_anything import sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide

imgs = []
gts = []
img_embeddings = []

# set up global variables
img_path = "../data/training_images/train/images"
gt_path = "../data/training_images/train/labels"

npz_path = "/scratch/st-tklee-1/ndsz/data/processed_npz"
data_name = "processed_data"

image_name_suffix = ".jpg"
label_name_suffix = ".png"

image_size = 256
label_id = 255
model_type = "vit_b"
checkpoint = "../data/sam_vit_b_01ec64.pth"

device = "cuda"
seed = 2023

sam_model = sam_model_registry[model_type](checkpoint=checkpoint).to(device)

def augment(image, ground_truth):
    num_img = 0
    remixes_per_iter = 6
    for i in range(remixes_per_iter):
        im_scale, gt_scale = ran_scale(image, ground_truth)
        for j in range(remixes_per_iter):
            im_flip, gt_flip = ran_flip(im_scale, gt_scale)
            for k in range(remixes_per_iter):
                im_rot, gt_rot = ran_rotate(im_flip, gt_flip)
                for l in range(remixes_per_iter):
                    if (np.count_nonzero(gt_rot) < 100):
                        print("mask too small")
                        break
                    im_crop, gt_crop = ran_crop(im_rot, gt_rot)
                    if (np.count_nonzero(gt_crop) < 100):
                        continue
                    try:
                        # ensure training images are the right size, rescale if necessary
                        im = transform.resize(
                            im_crop,
                            (image_size, image_size),
                            order=3,
                            preserve_range=True,
                            mode="constant",
                            anti_aliasing=True
                        )
                        gt = transform.resize(
                            gt_crop == label_id,
                            (image_size, image_size),
                            order=0,
                            preserve_range=True,
                            mode="constant"
                        )
                        # change data type
                        im = np.uint8(im)
                        gt = np.uint8(gt)

                        sam_transform = ResizeLongestSide(sam_model.image_encoder.img_size)
                        resize_img = sam_transform.apply_image(im)
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

                        if (np.count_nonzero(gt) > 100):
                            imgs.append(im)
                            gts.append(gt)
                            num_img += 1

                            with torch.no_grad():
                                embedding = sam_model.image_encoder(input_image)
                                img_embeddings.append(embedding.cpu().numpy()[0])
                    except Exception as e:
                        print(e)
                        continue;
                    
        print(f"Batch {i} complete, total images so far: {num_img}")

def ran_flip(image, ground_truth):
    choice = [1, -1]
    xscale = np.random.choice(choice)
    yscale = np.random.choice(choice)
    return image[::xscale, ::yscale, :], ground_truth[::xscale, ::yscale]

def ran_rotate(image, ground_truth):
    theta = random.randint(0, 360)
    ret_img = transform.rotate(
        image,
        angle=theta,
        resize=True,
        order=3,
        preserve_range=True,
        mode="constant"
    )
    ret_gt = transform.rotate(
        ground_truth,
        angle=theta,
        resize=True,
        order=0,
        preserve_range=True,
        mode="constant"
    )
    return ret_img, ret_gt

def ran_scale(image, ground_truth):
    scale = random.uniform(0.2, 1.5)
    ret_img = transform.rescale(
        image,
        scale=scale,
        order=3,
        preserve_range=True,
        mode="constant",
        anti_aliasing=True,
        channel_axis=2
    )
    ret_gt = transform.rescale(
        ground_truth,
        scale=scale,
        order=0,
        preserve_range=True,
        mode="constant",
        anti_aliasing=True
    )
    return ret_img, ret_gt

def ran_crop(image, ground_truth):
    if (image.shape[0] < image_size or image.shape[1] < image_size):
        return image, ground_truth
    y_true, x_true = np.where(ground_truth > 0)
    y_cent = np.average(y_true)
    x_cent = np.average(x_true)
    if (np.isnan(y_cent) or np.isnan(x_cent)):
        return image, ground_truth

    yy = np.random.randint(y_cent - 128, y_cent + 128)
    xx = np.random.randint(x_cent - 128, x_cent + 128)

    yy = max(0, min(image.shape[0] - image_size - 1, yy))
    xx = max(0, min(image.shape[1] - image_size - 1, xx))

    ey = yy + image_size
    ex = xx + image_size

    return image[yy:ey, xx:ex, :], ground_truth[yy:ey, xx:ex]
    
def process(gt_name: str, image_name: str):
    if image_name == None:
        image_name = gt_name.split(".")[0] + image_name_suffix
    ori_gt_data = io.imread(join(gt_path, gt_name))

    # if the mask is RGB, select the first channel
    if len(ori_gt_data.shape) == 3:
        ori_gt_data = gt_data[:, :, 0]
    assert len(ori_gt_data.shape) == 2, "ground truth should be 2D"

    # resize ground truth image
    gt_data = transform.resize(
        ori_gt_data == label_id,
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
        ori_image_data = io.imread(join(img_path, image_name))

        # remove alpha channel
        if (ori_image_data.shape[-1] > 3 and len(ori_image_data.shape) == 3):
            ori_image_data = ori_image_data[:, :, :3]
        
        # if the image is greyscale, repeat the last channel to convert it to RGB
        if (len(ori_image_data.shape) == 2):
            ori_image_data = np.repeat(ori_image_data[:, :, None], 3, axis=-1)
        
        # scale image to 50th and 99.5th percentile

        # rescale image
        image_data = transform.resize(
            ori_image_data,
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
        
        augment(ori_image_data, ori_gt_data)
    return

def preprocess_and_save(imgs, gts, img_embeddings):
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

if (__name__ == "__main__"):
    preprocess_and_save(imgs, gts, img_embeddings)