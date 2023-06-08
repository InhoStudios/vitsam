# link to local segment_anything instead of installed segment_anything

# do all imports
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
sam_model = sam_model_registry[model_type](checkpoint=checkpoint).to(device)
sam_model.train()
# set up optimizer, hyperparameter tuning will improve performance here
optimizer = torch.optim.Adam(sam_model.mask_decoder.parameters(), lr=1e-5, weight_decay=0)
seg_loss = monai.losses.DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')


# train the model
num_epochs = 100
losses = []
best_loss = 1e10
train_dataset = NpzDataset(npz_tr_path)
train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
for epoch in range(num_epochs):
    epoch_loss = 0
    # train
    for step, (image_embedding, gt2D, boxes) in enumerate(tqdm(train_dataloader)):
        with torch.no_grad():
            box_np = boxes.numpy()
            sam_trans = ResizeLongestSide(sam_model.image_encoder.img_size)
            box = sam_trans.apply_boxes(box_np, (gt2D.shape[-2], gt2D.shape[-1]))
            box_torch = torch.as_tensor(box, dtype=torch.float, device=device)
            if (len(box_torch.shape) == 2):
                box_torch = box_torch[:, None, :]
            # get prompt embeddings
            sparse_embeddings, dense_embeddings = sam_model.prompt_encoder(
                points=None,
                boxes=box_torch,
                masks=None
            )

            print(sparse_embeddings.shape, dense_embeddings.shape)

        # predicted masks
        mask_predictions, _ = sam_model.mask_decoder(
            image_embeddings=image_embedding.to(device),
            image_pe=sam_model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
        )

        loss = seg_loss(mask_predictions, gt2D.to(device))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    if (step != 0):
        epoch_loss /= step
    losses.append(epoch_loss)
    print(f'EPOCH: {epoch}, Loss: {epoch_loss}')
    # save latest model checkpoint
    torch.save(sam_model.state_dict(), join(model_save_path, 'sam_model_latest.pth'))
    # save best model
    if epoch_loss < best_loss:
        best_loss = epoch_loss
        torch.save(sam_model.state_dict(), join(model_save_path, 'sam_model_best.pth'))

# plot the loss
plt.plot(losses)
plt.title("Dice + Cross Entropy Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()
plt.savefig(join(model_save_path, "train_loss.png"))
plt.close()

# compare segmentation results between original SAM model and fine-tuned model