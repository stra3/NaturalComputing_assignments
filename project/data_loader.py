from pycocotools.coco import COCO
from pycocotools import mask as cocomask
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import matplotlib
import pylab
import random
import os
pylab.rcParams['figure.figsize'] = (8.0, 10.0)
matplotlib.use('TKAgg')

data_directory = "data/"
annotation_file_template = "{}/{}/annotation{}.json"

TRAIN_IMAGES_DIRECTORY = "data/train/images"
TRAIN_ANNOTATIONS_PATH = "data/train/annotation.json"
TRAIN_ANNOTATIONS_SMALL_PATH = "data/train/annotation-small.json"
coco = COCO(TRAIN_ANNOTATIONS_SMALL_PATH)

category_ids = coco.loadCats(coco.getCatIds())
print(category_ids)

# This generates a list of all `image_ids` available in the dataset
image_ids = coco.getImgIds(catIds=coco.getCatIds())
# For this demonstration, we will randomly choose an image_id
random_image_id = random.choice(image_ids)

# Now that we have an image_id, we can load its corresponding object by doing :
img = coco.loadImgs(random_image_id)[0]

image_path = os.path.join(TRAIN_IMAGES_DIRECTORY, img["file_name"])
I = io.imread(image_path)
plt.imshow(I)
plt.show()