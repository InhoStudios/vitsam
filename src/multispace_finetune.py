# link to local segment_anything instead of installed segment_anything

# do all imports
import numpy as np
import matplotlib.pyplot as plt

import os
from os.path import join

from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader, random_split

import monai

from segment_anything import SamPredictor, sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide

from utils.SurfaceDice import compute_dice_coefficient
from npzdataset import NpzDataset, MultispaceDataset

# torch.manual_seed(2023)
# np.random.seed(2023)

# test dataset class and dataloader
val_npz = "../data/multispace/"

dataset_filepath = "../data/multispace_vit_b/"
rgb_dataset = MultispaceDataset(dataset_filepath, is_hsv=False)
hsv_dataset = MultispaceDataset(dataset_filepath, is_hsv=True)
rgb_train, rgb_val = random_split(rgb_dataset, [np.floor(0.75 * len(rgb_dataset)), np.ceil(0.25 * len(rgb_dataset))])
hsv_train, hsv_val = random_split(hsv_dataset, [np.floor(0.75 * len(hsv_dataset)), np.ceil(0.25 * len(hsv_dataset))])


# set up model for fine-tuning
npz_tr_path = "../data/multispace/"
work_dir = "/scratch/st-tklee-1/ndsz/model"
task_name = "vitsam"
# prepare SAM model
model_type = "vit_b"
checkpoint = "../data/sam_vit_b_01ec64.pth"
device = "cuda"


model_save_path = join(work_dir, task_name)

os.makedirs(model_save_path, exist_ok=True)

rgb_sam_model = sam_model_registry[model_type](checkpoint=checkpoint).to(device)
rgb_sam_model.train()

# set up optimizer, hyperparameter tuning will improve performance here
optimizer = torch.optim.Adam(rgb_sam_model.mask_decoder.parameters(), lr=1e-5, weight_decay=0)
seg_loss = monai.losses.DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')

# train the model
num_epochs = 300
losses = []
val_losses = []
best_loss = 1e10
relative_tolerance = 0.9
early_stopping_epochs = 30
epochs_since_opt_loss = 0

rgb_train_dataloader = DataLoader(rgb_train, batch_size=48, shuffle=True)
rgb_val_dataloader = DataLoader(rgb_val, batch_size=16, shuffle=True)

hsv_train_dataloader = DataLoader(hsv_train, batch_size=48, shuffle=True)
hsv_val_dataloader = DataLoader(hsv_val, batch_size=16, shuffle=True)

train_dataset = NpzDataset(npz_tr_path)
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# TODO: repeat training step for HSV and RGB dataset
# TODO: change val_dataloader to rgb_val_dataloader
# TODO: repeat continue from here
# TODO: plot two plots

for epoch in range(num_epochs):
    rgb_epoch_loss = 0
    rgb_val_loss = 0
    hsv_epoch_loss = 0
    hsv_val_loss = 0

    # train RGB
    for step, (image_embedding, gt2D, boxes) in enumerate(tqdm(train_dataloader)):
        with torch.no_grad():
            box_np = boxes.numpy()
            sam_trans = ResizeLongestSide(rgb_sam_model.image_encoder.img_size)
            box = sam_trans.apply_boxes(box_np, (gt2D.shape[-2], gt2D.shape[-1]))
            box_torch = torch.as_tensor(box, dtype=torch.float, device=device)
            if (len(box_torch.shape) == 2):
                box_torch = box_torch[:, None, :]
            # get prompt embeddings
            sparse_embeddings, dense_embeddings = rgb_sam_model.prompt_encoder(
                points=None,
                boxes=box_torch,
                masks=None
            )

        # predicted masks
        mask_predictions, _ = rgb_sam_model.mask_decoder(
            image_embeddings=image_embedding.to(device),
            image_pe=rgb_sam_model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
        )

        loss = seg_loss(mask_predictions, gt2D.to(device))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        rgb_epoch_loss += loss.item()
    
    # validate RGB
    for step, (val_img_embedding, val_gt2D, val_boxes) in enumerate(tqdm(val_dataloader)):
        box_np = val_boxes.numpy()
        sam_trans = ResizeLongestSide(rgb_sam_model.image_encoder.img_size)
        box = sam_trans.apply_boxes(box_np, (val_gt2D.shape[-2], val_gt2D.shape[-1]))
        box_torch = torch.as_tensor(box, dtype=torch.float, device=device)
        if (len(box_torch.shape) == 2):
            box_torch = box_torch[:, None, :]
        # get prompt embeddings
        sparse_embeddings, dense_embeddings = rgb_sam_model.prompt_encoder(
            points=None,
            boxes=box_torch,
            masks=None
        )

        print(sparse_embeddings.shape, dense_embeddings.shape)

        mask_predictions, _ = rgb_sam_model.mask_decoder(
            image_embeddings=val_img_embedding.to(device),
            image_pe=rgb_sam_model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
        )
        vloss = seg_loss(mask_predictions, val_gt2D.to(device))
        rgb_val_loss += loss.item()

    if (step != 0):
        rgb_epoch_loss /= step
        rgb_val_loss /= step
    losses.append(rgb_epoch_loss)
    val_losses.append(rgb_val_loss)
    print(f'EPOCH: {epoch}, Training Loss: {rgb_epoch_loss}, Validation Loss: {rgb_val_loss}')
    # save latest model checkpoint
    torch.save(rgb_sam_model.state_dict(), join(model_save_path, 'sam_model_latest.pth'))
    # save best model
    if rgb_val_loss < relative_tolerance * best_loss:
        best_loss = rgb_epoch_loss
        epochs_since_opt_loss = 0
        torch.save(rgb_sam_model.state_dict(), join(model_save_path, 'sam_model_best.pth'))
    else:
        epochs_since_opt_loss += 1
    
    if epochs_since_opt_loss > early_stopping_epochs:
        break


# plot the loss
plt.plot(losses, label="Training Loss")
plt.plot(val_losses, label="Validation Loss")
plt.legend()
plt.title("Dice + Cross Entropy Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()
plt.savefig(join(model_save_path, "train_loss.png"))
plt.close()
