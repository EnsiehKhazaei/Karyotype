# Prepare datasets

After segmentation of original karyotype images, you have the images of chromosomes seperately and the following instructions explain how to provide the dataset for training neural networks. 
1. Provide a directory with the following structure and put all of your chromosome images in the `DATADIR/JPEGImages`.
```
$DATADIR/
|---- JPEGImages/
|---- FeatureTxts/
|     |---- VOC2007
|     |     |---- Annotations
|     |     |---- JPEGImages
|     |     |---- ImageSets
|     |     |     |---- Main
|     |     |     |     |---- test.txt
|     |     |     |     |---- trainval.txt

```
2. Run the [extract_features.py](https://github.com/EnsiehKhazaei/Karyotype/blob/main/classification/create_dataset/extract_features.py) on all chromosome images and save the results as a text file with the same name at `DATADIR/FeatureTxts`.
3. Run the [extract_features.py](https://github.com/EnsiehKhazaei/Karyotype/blob/main/classification/create_dataset/extract_features.py
```
line 46: image_sets=[('2007', 'trainval')]
line 104: image_sets=[('2007', 'test')]
```
Using the following code, divide the destined dataset to train and test sets.
```
import os
import random

trainval_percent = 0.1
train_percent = 0.9
xmlfilepath = './datasets/VOCdevkit/VOC2007/Annotations'
txtsavepath = './datasets/VOCdevkit/VOC2007/ImageSets'
total_xml = os.listdir(xmlfilepath)

num = len(total_xml)
list = range(num)
tv = int(num * trainval_percent)
tr = int(tv * train_percent)
trainval = random.sample(list, tv)
train = random.sample(trainval, tr)

ftest = open('./datasets/VOCdevkit/VOC2007/ImageSets/test.txt', 'w')
ftrain = open('./datasets/VOCdevkit/VOC2007/ImageSets/trainval.txt', 'w')

for i in list:
    name = total_xml[i][:-4] + '\n'
    if i in trainval:
        ftest.write(name)
    else:
        ftrain.write(name)

ftrain.close()
ftest.close()
```
