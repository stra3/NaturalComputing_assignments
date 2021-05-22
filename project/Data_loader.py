from torch.utils.data import Dataset, DataLoader
from pycocotools.coco import COCO
from pycocotools import mask as cocomask
import numpy as np
import skimage.io as io
import os
from cv2 import resize
import numpy as np 
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data.dataloader import DataLoader

class Data_loader():
    
    def __init__(self, annotation_path, image_path, val_size, batch_size, transform, shuffle = True, seed = 42):
        self.annotation_path = annotation_path
        self.image_path = image_path
        self.val_size = val_size
        self.batch_size = batch_size 
        self.transform = transform
        self.shuffle = shuffle
        self.seed = seed
     

    def get_loaders(self):
        data_set = ImageDataset(self.annotation_path, self.image_path)
        data_size = len(data_set)
        indices = list(range(data_size))
        split = int(np.floor(self.val_size * data_size))
        if self.shuffle:
            np.random.seed(self.seed)
            np.random.shuffle(indices)
        train_indices, val_indices = indices[split:], indices[:split]

        train_sampler = SubsetRandomSampler(train_indices)
        val_sampler = SubsetRandomSampler(val_indices)
        
        train_loader = DataLoader(data_set, batch_size=self.batch_size, sampler=train_sampler)
        val_loader = DataLoader(data_set, batch_size=self.batch_size, sampler=val_sampler)

        return train_loader, val_loader

class ImageDataset(Dataset):
    def __init__(self, annotaions_dir, images_dir, transform=None):
        self.annotaions_dir = annotaions_dir
        self.images_dir = images_dir
        self.transform = transform
        self.coco = COCO(annotaions_dir)
        self.category_ids = self.coco.loadCats(self.coco.getCatIds())
        self.image_ids = self.coco.getImgIds(catIds=self.coco.getCatIds())

        
    def __len__(self):
        return len(self.image_ids)
    
    def  __getitem__(self, idx):
        idx = self.image_ids[idx]
        img = self.coco.loadImgs(idx)[0]
        
        image_path = os.path.join(self.images_dir, img["file_name"])
        I = io.imread(image_path)
        I = I.transpose(-1, 0, 1) #w * h * c to c * w * h
        
        annotation_ids = self.coco.getAnnIds(imgIds = img['id'])
        annotations = self.coco.loadAnns(annotation_ids)
        
        mask = np.zeros((img['height'], img['width']))
        for annotation in annotations:
            rle = cocomask.frPyObjects(annotation['segmentation'], img['height'], img['width'])
            m = cocomask.decode(rle)
            m = m.reshape((img['height'], img['width']))
            mask = mask + m
        
        mask[mask > 0] = 1

        if self.transform is not None:
            augmentations = self.transform(image=I, mask=mask)
            I = augmentations["image"]
            mask = augmentations["mask"]
        return I, mask


