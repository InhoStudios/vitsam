# VitSAM
Finetuning Facebook Research's Segment Anything Algorithm for vitiligo dermatology images

## Notes

`./data/` file structure:

- `./data/training_images` contains all training images and associated masks for fine-tuning
  - `/train/` contains the training set
  - `/test/` contains the test set
  - Inside each folder, images are in the `/images` folder, and binary masks are in the `/labels` folder
- `./data/sam_vit_b_01ec64.pth`: This is our training checkpoint before we do fine-tuning. Currently, we're using the `'vit_b'` model type.
- `./data/processed_npz_[model_type]`: This stores our final training archive as a `.npz` file. `[model_type]` refers to the checkpoint type used for fine-tuning. 
  - The final data is stored as a file named `processed_data.npz`