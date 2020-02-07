import torch.nn as nn
from torch.utils.data import DataLoader
from models import FeatureExtractionModel
from generate_dataset import TrainDataset
import params
import torch

train_dataset = TrainDataset()
train_loader = DataLoader(train_dataset, batch_size=params.batch_size)

criterion = nn.BCELoss()
net = FeatureExtractionModel()
optim = torch.optim.Adam(net.parameters(), lr=params.lr, betas=(params.beta1, 0.999))

print('dataset created')
for e in range(params.num_epochs + 1):
    epoch_loss = 0
    for bindex, data in enumerate(train_loader, 0):

        img = data[0]
        label = data[1]

        img_t = img

        pred_t = net.forward(img_t)
        loss = criterion(pred_t.float(), label)
        epoch_loss += loss.item()
        optim.zero_grad()

        # Backpropagation
        loss.backward()
        optim.step()
        if (bindex + 1) % 250 == 0:
            print("Epoch {}, loss={}, batch={}".format(e, loss, bindex))

    print("Epoch {}, loss={}".format(e, epoch_loss))
    torch.save(net, 'saved_models/model_epoch_{}.pt'.format(e))
    print("Saved model...")
