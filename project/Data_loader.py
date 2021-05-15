from torch.utils.data import Dataset, DataLoader
from pycocotools.coco import COCO
from pycocotools import mask as cocomask
import numpy as np
import skimage.io as io
import os

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
        return I, mask