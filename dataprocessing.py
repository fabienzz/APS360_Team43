import numpy as np
import os
import math
from skimage.io import imread
import cv2

def loadData(path,width,height,size,scale=255):
    data = np.zeros([height,width,size])

    for i in range(size):
        img = imread(os.path.join(path,str(i)+'.png'),as_grey=True)
        cur_height,cur_width = img.shape
        if cur_height / cur_width > height / width:
            new_width = cur_height * width / height
            pad = (new_width - cur_width) / 2
            img_padded= cv2.copyMakeBorder(img, 0, 0, math.ceil(pad),math.floor(pad), cv2.BORDER_CONSTANT, value=255)
        elif cur_height / cur_width < height / width:
            new_height = cur_width * height / width
            pad = (new_height - cur_height) / 2
            img_padded= cv2.copyMakeBorder(img, math.ceil(pad),math.floor(pad), 0, 0, cv2.BORDER_CONSTANT, value=255)
    
    img_rescaled = img_padded / (scale / 2) - 1 # Rescale to [-1, 1]
    img_resize = cv2.resize(img_rescaled, (width, height)) # Resize image
    data[:,:,i] = img_resize

    return data

def normalizeData(test_images, train_images, val_images):
  # Find mean and std
  #all_data = np.concatenate([train_images, test_images, val_images], axis = 2)
  mean = np.mean(train_images)
  std = np.std(train_images)

  # Normalize
  test_images = (test_images - mean) / std
  train_images = (train_images - mean) / std
  val_images = (val_images - mean) / std

  return test_images, train_images, val_images