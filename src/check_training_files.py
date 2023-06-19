import numpy as np
from npzdataset import NpzDataset

aug_processed = "../data/other/aug_processed_data.npz"
processed = "../data/other/processed_data.npz"

npza = np.load(aug_processed)
npzb = np.load(processed)

print(npza['imgs'], npzb['imgs'])
print(npza['gts'], npzb['gts'])
print(npza['img_embeddings'], npzb['img_embeddings'])