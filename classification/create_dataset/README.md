# Prepare datasets

After segmentation of original karyotype images, you have the images of chromosomes seperately and the following instructions explain how to provide the dataset for training neural networks. 
1. Provide a directory with the following structure and put all of your chromosome images in the `DATADIR/JPEGImages`.
```
$DATADIR/
|---- JPEGImages/
|---- FeatureTxts/

```

2. Run the [extract_features.py](https://github.com/EnsiehKhazaei/Karyotype/blob/main/classification/create_dataset/extract_features.py) on all chromosome images and save the results as a text file with the same name at `DATADIR/FeatureTxts`.

3. Run the [create_dataset.py](https://github.com/EnsiehKhazaei/Karyotype/blob/main/classification/create_dataset/create_dataset.py) to creat the training set in order to train your deep neural network.
