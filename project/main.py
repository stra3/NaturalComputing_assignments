from Data_loader import ImageDataset, Data_loader
from U_net import UNet
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
import numpy as np

TRAIN_IMAGES_DIRECTORY = "small_data" # data/train/images"
TRAIN_ANNOTATIONS_PATH = "data/train/annotation.json" #roughly 280000 images
TRAIN_ANNOTATIONS_SMALL_PATH = "data/train/annotation-small.json" #8366 images

BATCH_SIZE = 1
EPOCHS = 10
LEARNING_RATE = 1e-4 
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def train(train_loader, val_loader, model, optimizer, criterion):
    
    best_score = 0
    for epoch in range(EPOCHS):
        avg_loss = []
        i = 0
        for batch_idx, (data, mask) in enumerate(train_loader):
            data = data.to(DEVICE, dtype=torch.float)
            mask =  mask.float().unsqueeze(0).to(DEVICE)

            optimizer.zero_grad()
            pred = model(data)
            loss = criterion(pred, mask)
            loss.backward()

            avg_loss.append(loss.item())
            optimizer.step()
            #if i % 5 == 0:
            #    print(f"{i/len(loader)*100}%")
            i+=1

            dice_score = eval_model(val_loader, model)
        if dice_score > best_score:
            best_score = dice_score
            print("saving model...")
            save("model", model)
            print("saved succesfully")

        print(f"epoch: {epoch}/{EPOCHS}     loss: {np.mean(avg_loss)}")

def save(path, model):
    torch.save(model.state_dict(), path)

def eval_model(loader, model):
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(DEVICE, dtype=torch.float)
            y = y.float().to(DEVICE).unsqueeze(1)
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
            num_correct += (preds == y).sum()
            num_pixels += torch.numel(preds)
            dice_score += (2 * (preds * y).sum()) / (
                (preds + y).sum() + 1e-8
            )

    print(f"Got {num_correct}/{num_pixels} with acc {num_correct/num_pixels*100:.2f}")
    print(f"Dice score: {dice_score/len(loader)}")
    model.train()
    return dice_score

if __name__ == "__main__":
    
    model = UNet().to(DEVICE)
    
    dataset = Data_loader(TRAIN_ANNOTATIONS_SMALL_PATH, TRAIN_IMAGES_DIRECTORY, 0.2, BATCH_SIZE, transform=None, shuffle=True, seed=42 )
    train_loader, val_loader = dataset.get_loaders()

    loss = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train(train_loader, val_loader, model, optimizer, loss)