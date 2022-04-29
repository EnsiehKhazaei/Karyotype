import cv2
import os

def add_padding(im):
  old_size = im.shape[:2] # old_size is in (height, width) format

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
Data_dir = 'images/'
train_data = []
test_data  = []
validation_data = []
length_data2   = []
area_data2     = []
IMG_SIZE_1, IMG_SIZE_2 = 32, 128
X_train = []
Xp_train = []
y_train = []
X_area_train   = []
X_length_train = []
## Creat train data
for i in range(len(all_txts_train)):
    img = all_imgs_train[i]
    img_array = cv2.imread(os.path.join(Data_dir,img), cv2.IMREAD_GRAYSCALE)
    new_array = add_padding(img_array)
    lbl = int(os.path.splitext(img)[0].split('-')[-2].strip())
    
X_train = np.array(Xp_train).reshape(-1, IMG_SIZE_1, IMG_SIZE_2, 1)
X_area_train = np.array(X_area_train)
X_length_train = np.array(X_length_train)
X_train = X_train/255.0
y_train = np.array(y_train).reshape(-1,1)

from keras.models import Sequential , Model
from keras.layers import Dense, Dropout, Flatten , GlobalAveragePooling2D , Input, Conv2D
from keras.applications.mobilenet import MobileNet
from keras.applications.resnet50 import ResNet50
import tensorflow as tf 

## Define onehot_label function
def onehot_labels(labels):
    return np.eye(24)[labels]  

num_cls = 24
y_train = onehot_labels(y_train)
y_valid = onehot_labels(y_valid)
y_test  = onehot_labels(y_test)
print(y_train.shape)
print(y_test.shape)
y_train = y_train.reshape((X_train.shape[0], num_cls))
    text_file = open(all_txts_train[i],'r')
    area = (float(text_file.readline()))
    length = (float(text_file.readline()))
    train_data.append([new_array, area, length, lbl])
