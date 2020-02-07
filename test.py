from PIL import Image
from models import FeatureExtractionModel
import torch
import pandas as pd
import os
from params import dataset_path
import numpy as np

model = torch.load('saved_models/model_epoch_3.pt')
model.eval()

img = Image.open("crim.jpg").resize([128, 128])
df = pd.read_csv(os.path.join(dataset_path, 'list_attr_celeba.csv'))

# img_name, attr_numpy_array
all_attrs = df.columns[1:]
img = np.transpose(np.array(img), (1, 2, 0))
img = np.reshape(img, (1, 3, 128, 128)).astype(np.float32)
img_t = torch.Tensor(img).float()
pred = model.forward(img_t)
pred = pred.detach().numpy()
pred = pred.reshape(-1).round()
pred = pred.astype(np.int32)
pred = list(pred)
l = []
for a, p in zip(all_attrs, pred):
    if p == 1:
        l.append(a)
print(l)
