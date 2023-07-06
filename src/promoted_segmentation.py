import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from segment_anything import sam_model_registry, SamPredictor
from segment_anything.utils.onnx import SamOnnxModel
from os.path import join, exists
import os

import onnxruntime
from onnxruntime.quantization import QuantType
from onnxruntime.quantization.quantize import quantize_dynamic

def show_mask(mask, ax):
    colour = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * colour.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_points(coords, labels, ax, marker_size=50):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))

# checkpoint = "../data/sam_vit_b_01ec64.pth"
checkpoint = "../model/vitsam/minor_finetuned_sam_model_best.pth"
model_type = "vit_b"
output_dir = "/scratch/st-tklee-1/ndsz/promoted_w_onnx/"
os.makedirs(output_dir, exist_ok=True)

sam = sam_model_registry[model_type](checkpoint=checkpoint)

onnx_model_path = None

import warnings

onnx_model_path = join(output_dir, "ft_sam_onnx_example.onnx")

if not exists(onnx_model_path):
    onnx_model = SamOnnxModel(sam, return_single_mask=True)

    dynamic_axes = {
        "point_coords": {1: "num_points"},
        "point_labels": {1: "num_points"}
    }

    embed_dim = sam.prompt_encoder.embed_dim
    embed_size = sam.prompt_encoder.image_embedding_size
    mask_input_size = [4 * x for x in embed_size]
    dummy_inputs = {
        "image_embeddings": torch.randn(1, embed_dim, *embed_size, dtype=torch.float),
        "point_coords": torch.randint(low=0, high=1024, size=(1, 5, 2), dtype=torch.float),
        "point_labels": torch.randint(low=0, high=4, size=(1, 5), dtype=torch.float),
        "mask_input": torch.randn(1, 1, *mask_input_size, dtype=torch.float),
        "has_mask_input": torch.tensor([1], dtype=torch.float),
        "orig_im_size": torch.tensor([1500, 2250], dtype=torch.float),
    }
    output_names = ["masks", "iou_predictions", "low_res_masks"]

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
        warnings.filterwarnings("ignore", category=UserWarning)
        with open(onnx_model_path, "wb") as f:
            torch.onnx.export(
                onnx_model,
                tuple(dummy_inputs.values()),
                f, 
                export_params=True,
                verbose=False,
                opset_version=17,
                do_constant_folding=True,
                input_names=list(dummy_inputs.keys()),
                output_names=output_names,
                dynamic_axes=dynamic_axes
            )

im_path = "../data/training_images/test/images/test3.jpg"
im_path = "../data/IMG_2188.jpg"

image = cv2.imread(im_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# plt.figure(figsize=(10, 10))
# plt.imshow(image)
# plt.show()

# plt.close()
# import sys
# sys.exit()

fig, axs = plt.subplots(2, 2, figsize=(20,20))

axs[0, 0].imshow(image)
axs[0, 0].set_title("Original Image")
axs[0, 0].axis('off')

print("===ORIGINAL IMAGE===")

ort_session = onnxruntime.InferenceSession(onnx_model_path)

sam.to(device="cuda")
predictor = SamPredictor(sam)

predictor.set_image(image)

image_embedding = predictor.get_image_embedding().cpu().numpy()

# input_point = np.array([[500, 626], [329, 389], [306, 568], [248, 487], [197, 634], [385, 843], [440, 931], [457, 978], [331, 682], [639, 649],
#                         [540, 883], [284, 529], [348, 620], [357, 428], [390, 548]])
# input_label = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
#                         0, 0, 0, 0, 0])
# input_box = np.array([116, 292, 755, 1047])

input_point = np.array([[438, 397], [491, 146], [290, 477], [623, 580], [857, 493], [307, 258], [568, 110], [424, 99],
                        [795, 443], [872, 562], [825, 697], [215, 701], [107, 488], [609, 241], [318, 191], 
                        [256, 422], [295, 348], [306, 325], [618, 368], [668, 397], [671, 522], [712, 446], [595, 315], [577, 229], [524, 232], [377, 163], [342, 101]])
input_label = np.array([1, 1, 1, 1, 1, 1, 1, 1,
                        0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
input_box = np.array([147, 8, 942, 761])

onnx_coord = np.concatenate([input_point, np.array([[0.0, 0.0]])], axis=0)[None, :, :]
onnx_label = np.concatenate([input_label, np.array([-1])], axis=0)[None, :].astype(np.float32)

onnx_coord = predictor.transform.apply_coords(onnx_coord, image.shape[:2]).astype(np.float32)

onnx_mask_input = np.zeros((1, 1, 256, 256), dtype=np.float32)
onnx_has_mask_input = np.zeros(1, dtype=np.float32)

ort_inputs = {
    "image_embeddings": image_embedding,
    "point_coords": onnx_coord,
    "point_labels": onnx_label,
    "mask_input": onnx_mask_input,
    "has_mask_input": onnx_has_mask_input,
    "orig_im_size": np.array(image.shape[:2], dtype=np.float32)
}

print("===POINT PROMO===")

masks, _, low_res_logits = ort_session.run(None, ort_inputs)
masks = masks > predictor.model.mask_threshold

axs[0, 1].imshow(image)
show_mask(masks, axs[0, 1])
show_points(input_point, input_label, axs[0, 1])
axs[0, 1].axis('off')
axs[0, 1].set_title("Point promotion")

onnx_box_coords = input_box.reshape(2, 2)
onnx_box_labels = np.array([2, 3])

onnx_coord = np.concatenate([np.array([[0.0, 0.0]]), onnx_box_coords], axis=0)[None, :, :]
onnx_label = np.concatenate([np.array([-1]), onnx_box_labels], axis=0)[None, :].astype(np.float32)

onnx_coord = predictor.transform.apply_coords(onnx_coord, image.shape[:2]).astype(np.float32)

onnx_mask_input = np.zeros((1, 1, 256, 256), dtype=np.float32)
onnx_has_mask_input = np.zeros(1, dtype=np.float32)

ort_inputs = {
    "image_embeddings": image_embedding,
    "point_coords": onnx_coord,
    "point_labels": onnx_label,
    "mask_input": onnx_mask_input,
    "has_mask_input": onnx_has_mask_input,
    "orig_im_size": np.array(image.shape[:2], dtype=np.float32)
}

print("===BOX PROMO===")
masks, _, _ = ort_session.run(None, ort_inputs)
masks = masks > predictor.model.mask_threshold

axs[1, 0].imshow(image)
show_mask(masks, axs[1, 0])
show_box(input_box, axs[1, 0])
axs[1, 0].axis('off')
axs[1, 0].set_title("Box promotion")

onnx_coord = np.concatenate([input_point, onnx_box_coords], axis=0)[None, :, :]
onnx_label = np.concatenate([input_label, onnx_box_labels], axis=0)[None, :].astype(np.float32)

onnx_coord = predictor.transform.apply_coords(onnx_coord, image.shape[:2]).astype(np.float32)

onnx_mask_input = np.zeros((1, 1, 256, 256), dtype=np.float32)
onnx_has_mask_input = np.zeros(1, dtype=np.float32)

ort_inputs = {
    "image_embeddings": image_embedding,
    "point_coords": onnx_coord,
    "point_labels": onnx_label,
    "mask_input": onnx_mask_input,
    "has_mask_input": onnx_has_mask_input,
    "orig_im_size": np.array(image.shape[:2], dtype=np.float32)
}

print("===POINT + BOX PROMO===")
masks, _, _ = ort_session.run(None, ort_inputs)
masks = masks > predictor.model.mask_threshold

axs[1, 1].imshow(image)
show_mask(masks, axs[1, 1])
show_box(input_box, axs[1, 1])
show_points(input_point, input_label, axs[1, 1])
axs[1, 1].axis('off')
axs[1, 1].set_title("Point + Box")
fig.tight_layout()
plt.savefig(join(output_dir, "2188_finetuned.jpg"))

plt.close()
del masks

################################


checkpoint = "../data/sam_vit_b_01ec64.pth"
# checkpoint = "../model/vitsam/minor_finetuned_sam_model_best.pth"
model_type = "vit_b"
output_dir = "/scratch/st-tklee-1/ndsz/promoted_w_onnx/"
os.makedirs(output_dir, exist_ok=True)

sam = sam_model_registry[model_type](checkpoint=checkpoint)

onnx_model_path = None

import warnings

onnx_model_path = join(output_dir, "def_sam_onnx_example.onnx")

if not exists(onnx_model_path):
    onnx_model = SamOnnxModel(sam, return_single_mask=True)

    dynamic_axes = {
        "point_coords": {1: "num_points"},
        "point_labels": {1: "num_points"}
    }

    embed_dim = sam.prompt_encoder.embed_dim
    embed_size = sam.prompt_encoder.image_embedding_size
    mask_input_size = [4 * x for x in embed_size]
    dummy_inputs = {
        "image_embeddings": torch.randn(1, embed_dim, *embed_size, dtype=torch.float),
        "point_coords": torch.randint(low=0, high=1024, size=(1, 5, 2), dtype=torch.float),
        "point_labels": torch.randint(low=0, high=4, size=(1, 5), dtype=torch.float),
        "mask_input": torch.randn(1, 1, *mask_input_size, dtype=torch.float),
        "has_mask_input": torch.tensor([1], dtype=torch.float),
        "orig_im_size": torch.tensor([1500, 2250], dtype=torch.float),
    }
    output_names = ["masks", "iou_predictions", "low_res_masks"]

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
        warnings.filterwarnings("ignore", category=UserWarning)
        with open(onnx_model_path, "wb") as f:
            torch.onnx.export(
                onnx_model,
                tuple(dummy_inputs.values()),
                f, 
                export_params=True,
                verbose=False,
                opset_version=17,
                do_constant_folding=True,
                input_names=list(dummy_inputs.keys()),
                output_names=output_names,
                dynamic_axes=dynamic_axes
            )

# plt.figure(figsize=(10, 10))
# plt.imshow(image)
# plt.show()

# plt.close()

fig, axs = plt.subplots(2, 2, figsize=(20,20))

axs[0, 0].imshow(image)
axs[0, 0].set_title("Original Image")
axs[0, 0].axis('off')

print("===ORIGINAL IMAGE===")

ort_session = onnxruntime.InferenceSession(onnx_model_path)

sam.to(device="cuda")
predictor = SamPredictor(sam)

predictor.set_image(image)

image_embedding = predictor.get_image_embedding().cpu().numpy()

onnx_coord = np.concatenate([input_point, np.array([[0.0, 0.0]])], axis=0)[None, :, :]
onnx_label = np.concatenate([input_label, np.array([-1])], axis=0)[None, :].astype(np.float32)

onnx_coord = predictor.transform.apply_coords(onnx_coord, image.shape[:2]).astype(np.float32)

onnx_mask_input = np.zeros((1, 1, 256, 256), dtype=np.float32)
onnx_has_mask_input = np.zeros(1, dtype=np.float32)

ort_inputs = {
    "image_embeddings": image_embedding,
    "point_coords": onnx_coord,
    "point_labels": onnx_label,
    "mask_input": onnx_mask_input,
    "has_mask_input": onnx_has_mask_input,
    "orig_im_size": np.array(image.shape[:2], dtype=np.float32)
}

print("===POINT PROMO===")

masks, _, low_res_logits = ort_session.run(None, ort_inputs)
masks = masks > predictor.model.mask_threshold

axs[0, 1].imshow(image)
show_mask(masks, axs[0, 1])
show_points(input_point, input_label, axs[0, 1])
axs[0, 1].axis('off')
axs[0, 1].set_title("Point promotion")

onnx_box_coords = input_box.reshape(2, 2)
onnx_box_labels = np.array([2, 3])

onnx_coord = np.concatenate([np.array([[0.0, 0.0]]), onnx_box_coords], axis=0)[None, :, :]
onnx_label = np.concatenate([np.array([-1]), onnx_box_labels], axis=0)[None, :].astype(np.float32)

onnx_coord = predictor.transform.apply_coords(onnx_coord, image.shape[:2]).astype(np.float32)

onnx_mask_input = np.zeros((1, 1, 256, 256), dtype=np.float32)
onnx_has_mask_input = np.zeros(1, dtype=np.float32)

ort_inputs = {
    "image_embeddings": image_embedding,
    "point_coords": onnx_coord,
    "point_labels": onnx_label,
    "mask_input": onnx_mask_input,
    "has_mask_input": onnx_has_mask_input,
    "orig_im_size": np.array(image.shape[:2], dtype=np.float32)
}

print("===BOX PROMO===")
masks, _, _ = ort_session.run(None, ort_inputs)
masks = masks > predictor.model.mask_threshold

axs[1, 0].imshow(image)
show_mask(masks, axs[1, 0])
show_box(input_box, axs[1, 0])
axs[1, 0].axis('off')
axs[1, 0].set_title("Box promotion")

onnx_coord = np.concatenate([input_point, onnx_box_coords], axis=0)[None, :, :]
onnx_label = np.concatenate([input_label, onnx_box_labels], axis=0)[None, :].astype(np.float32)

onnx_coord = predictor.transform.apply_coords(onnx_coord, image.shape[:2]).astype(np.float32)

onnx_mask_input = np.zeros((1, 1, 256, 256), dtype=np.float32)
onnx_has_mask_input = np.zeros(1, dtype=np.float32)

ort_inputs = {
    "image_embeddings": image_embedding,
    "point_coords": onnx_coord,
    "point_labels": onnx_label,
    "mask_input": onnx_mask_input,
    "has_mask_input": onnx_has_mask_input,
    "orig_im_size": np.array(image.shape[:2], dtype=np.float32)
}

print("===POINT + BOX PROMO===")
masks, _, _ = ort_session.run(None, ort_inputs)
masks = masks > predictor.model.mask_threshold

axs[1, 1].imshow(image)
show_mask(masks, axs[1, 1])
show_box(input_box, axs[1, 1])
show_points(input_point, input_label, axs[1, 1])
axs[1, 1].axis('off')
axs[1, 1].set_title("Point + Box")
fig.tight_layout()
plt.savefig(join(output_dir, "2188_default.jpg"))