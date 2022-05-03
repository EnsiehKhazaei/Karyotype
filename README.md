# Analysis of Karyotype Images
## Introduction
Karyotype is a genetic test that is used for detection of chromosomal defects. In a karyotype test, an image is captured from chromosomes during the cell division. The captured images are then analyzed by cytogeneticists in order to detect possible chromosomal defects. In this research, we have proposed an automated pipeline for analysis of karyotype images. There are three main steps for karyotype image analysis: image enhancement, image segmentation and chromosome classification. In this research, we have proposed a novel chromosome segmentation algorithm to decompose overlapped chromosomes. We have also proposed a CNN-based classifier which outperforms all the existing classifiers. 

## Guideline
<details>
<summary>Chromosome resolving</summary>
 
 First, run the [main_resolving.py](https://github.com/EnsiehKhazaei/Karyotype/blob/main/chromosome_resolving/main_resolving.py) file.
 
 There are two functions at the end of [overlap_resolving.py](https://github.com/EnsiehKhazaei/Karyotype/blob/main/chromosome_resolving/overlap_resolving.py) file:

 1. “plot_overlap_org_img”: returning the overlapped points for the original image 
 2. “plot_overlap_contour”: returning the contour image for the original image 
  
 Results are saved in [output_resolving](https://github.com/EnsiehKhazaei/Karyotype/tree/main/chromosome_resolving/output_resolving) folder. You can find samples for the 1 and 3 images in the [output_resolving](https://github.com/EnsiehKhazaei/Karyotype/tree/main/chromosome_resolving/output_resolving) folder.

</details>

<details>
<summary>Create dataset</summary>
 
After segmentation of original karyotype images, you have the images of chromosomes seperately and the following instructions explain how to provide the dataset for training neural networks. 
1. Provide a directory with the following structure and put all of your chromosome images in the `DATADIR/JPEGImages`.
```shell
$DATADIR/
|---- JPEGImages/
|---- FeatureTxts/
```
 
2. Run the [extract_features.py](https://github.com/EnsiehKhazaei/Karyotype/blob/main/classification/create_dataset/extract_features.py) on all chromosome images in order to calculate the length and area features for each chromosome image. You shoud change the file paths in this code.
 ```shell
 img_path = 'DATADIR/JPEGImages' # directory of input images
 txt_path = 'DATADIR/FeatureTxts' # directory for saving text files
 ```
 
 [extract_features.py](https://github.com/EnsiehKhazaei/Karyotype/blob/main/classification/create_dataset/extract_features.py) saves the results (area and length features) as a text file with the same name at `DATADIR/FeatureTxts`.

3. Run the [create_dataset.py](https://github.com/EnsiehKhazaei/Karyotype/blob/main/classification/create_dataset/create_dataset.py) to creat the training set in order to train your deep neural network. The input of [create_dataset.py](https://github.com/EnsiehKhazaei/Karyotype/blob/main/classification/create_dataset/create_dataset.py) is the images and text files path and the outputs of this code are `X_train`, `X_area_train`, `X_length_train`, and `y_train`. 
</details>

<details>
<summary>Classification</summary>
 
 All codes of deep neural networks which are used in this research exist at [classification](https://github.com/EnsiehKhazaei/Karyotype/blob/main/classification) directory. In addition, we put the confusion matrixes of all neural networks at [confusion_matrixes](https://github.com/EnsiehKhazaei/Karyotype/blob/main/classification/confusion_matrixes) folder. In order to train the deep neural networks, you should create your dataset using the instructions in the previous part and then run the code of your desired neural network at [classification](https://github.com/EnsiehKhazaei/Karyotype/blob/main/classification) directory. The outputs of these codes are the weight of trained model, plot of accuracy and loss during training, and confusion matrix of the model.

 After training our models, we perform the trained model on the segmented chromosomes of a sample and then the output of the model is the chromosome lables which are from 1 to 24. Finally, according the results of the classifier, it is possible to detect the abnormal chromosomes.
</details>

## Results
Our classifier is trained by a dataset of about 162,000 human chromosome images and the success rate of our segmentation algorithm is 95%. In addition, our experimental
results show that the accuracy of our classifier for human chromosomes is 92.63% and our novel post-processing algorithm increases the classification results to 94%.
We put some output of our chromosome resolving algorithm and our proposed pipeline in the follwing.
### overlap resolving
Four samples of overlapped chromosomes and the results of our algorithm are shown in the follwing image. 

<div align="center"><img src="/chromosome_resolving/output_resolving/4_sample_overlap_resolving.png" width="700"></div>


The two first sample show two overlapped chromosomes while the third and forth samples show the success of our algorithm on three overlapped chromosomes. Furthermore, in the forth sample, the overlapped chromosomes created a loop, however our chromosome resolving algorithm successfully separates them.

## Contact
For questions about our paper or code, please contact [Ensieh Khazaei](mailto:khazaei1394@gmail.com) and [Ala Emrany](mailto:emranyala@gmail.com).


