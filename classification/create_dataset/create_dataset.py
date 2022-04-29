import cv2
import os
import numpy as np


def onehot_labels(labels):
    return np.eye(24)[labels]  

def add_padding(im):
  old_size = im.shape[:2] 

  ## Scale larger images and then padding them if it is necessary
  if (old_size[0]>128 or old_size[1]>32):
      scaling = max(old_size[0]/128 , old_size[1]/32)
      new_size_1 = int(round(old_size[1] // scaling))
      new_size_0 = int(round(old_size[0] // scaling))

      im = cv2.resize(im, (new_size_1, new_size_0))
      delta_w = 32  - int(round(new_size_1))
      delta_h = 128 - int(round(new_size_0))

      top, bottom = int(delta_h//2), int(delta_h-(delta_h//2))
      left, right = int(delta_w//2), int(delta_w-(delta_w//2))

      color = [0, 0, 0]
      new_im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT,
                              value=color)
  ## Padding smaller images to the size of 128*32
  else:
      delta_w = 32  - old_size[1]
      delta_h = 128 - old_size[0]
      top, bottom = delta_h//2, delta_h-(delta_h//2)
      left, right = delta_w//2, delta_w-(delta_w//2)
      
      color = [0, 0, 0]
      new_im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT,
                              value=color)

  return new_im

## Create  dataset
imgs_dir = 'images/'
txts_dir = ''
num_cls = 24

X_train = []
y_train = []
X_area_train   = []
X_length_train = []
## Creat train data
for name in os.listdir(imgs_dir):#for i in range(len(all_txts_train)):
    # img = all_imgs_train[i]
    img = cv2.imread(os.path.join(imgs_dir, name), cv2.IMREAD_GRAYSCALE)
    img = np.transpose(add_padding(img))
    lbl = int(os.path.splitext(name)[0].split('-')[-2].strip())
    
    text_file = open(os.path.join(txts_dir, name.split('.')[0]+'.txt'),'r')
    area = float(text_file.readline())
    length = float(text_file.readline())
    
    X_train.append(img)
    y_train.append(lbl)
    X_area_train.append(area)
    X_length_train.append(length)
    
X_train = np.array(X_train)
X_area_train = np.array(X_area_train)
X_length_train = np.array(X_length_train)
X_train = X_train/255.0
y_train = np.array(y_train).reshape(-1,1)

y_train = onehot_labels(y_train)
y_train = y_train.reshape((X_train.shape[0], num_cls))
