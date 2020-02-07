import pandas as pd
import numpy as np
from params import dataset_path
import os
from PIL import Image
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
import torch

def generate_dataset():
    df = pd.read_csv(os.path.join(dataset_path, 'list_attr_celeba.csv'))

    # img_name, attr_numpy_array
    data = []
    all_attrs = df.columns[1:]
    for index, row in df.iterrows():
        im = row['image_id']
        a = []
        for key, value in row.items():
            if value == 1:
                a.append(key)
        img_name = im
        attr_np_array = np.zeros(shape=(len(all_attrs)))
        for i in range(len(all_attrs)):
            attr_np_array[i] = 1 if all_attrs[i] in a else 0
        data.append([img_name, attr_np_array])
    return data


class TrainDataset(Dataset):

    def __init__(self):
        self.data = generate_dataset()
        print('Dataset created')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img = np.array(
            Image.open(os.path.join(dataset_path, 'img_align_celeba/img_align_celeba/{}'.format(self.data[index][0]))).resize([128, 128]))
        labels = self.data[index][1]

        img = np.transpose(img, (2, 0, 1))
        img = img.reshape((3, 128, 128)).astype(np.float32)
        labels = labels.astype(np.float32)
        img_tensor = torch.Tensor(img).float()
        return img_tensor, labels

