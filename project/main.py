from Data_loader import ImageDataset
from U_net import UNet
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
import numpy as np

TRAIN_IMAGES_DIRECTORY = "data/train/images"
TRAIN_ANNOTATIONS_PATH = "data/train/annotation.json" #roughly 280000 images
TRAIN_ANNOTATIONS_SMALL_PATH = "data/train/annotation-small.json" #8366 images

BATCH_SIZE = 2
EPOCHS = 10
LEARNING_RATE = 1e-4 
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def train(loader, model, optimizer, criterion):
    
    for epoch in range(EPOCHS):
        avg_loss = []
        for batch_idx, (data, mask) in enumerate(loader):
            data = data.to(DEVICE, dtype=torch.float)
            mask =  mask.float().unsqueeze(0).to(DEVICE)

            optimizer.zero_grad()
            pred = model(data)
            loss = criterion(pred, mask)
            loss.backward()

            avg_loss.append(loss.item())
            loss.step()
        print(f"epoch: {epoch}/{EPOCHS}     loss: {np.mean(avg_loss)}")


if __name__ == "__main__":
    model = UNet().to(DEVICE)
    
    train_ds = ImageDataset(TRAIN_ANNOTATIONS_SMALL_PATH, TRAIN_IMAGES_DIRECTORY)
    train_loader = DataLoader(train_ds, batch_size = BATCH_SIZE, shuffle=True)

    loss = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train(train_loader, model, optimizer, loss)