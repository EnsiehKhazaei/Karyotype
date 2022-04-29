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
 
2. Run the [extract_features.py](https://github.com/EnsiehKhazaei/Karyotype/blob/main/classification/create_dataset/extract_features.py) on all chromosome images and save the results as a text file with the same name at `DATADIR/FeatureTxts`.

3. Run the [create_dataset.py](https://github.com/EnsiehKhazaei/Karyotype/blob/main/classification/create_dataset/create_dataset.py) to creat the training set in order to train your deep neural network.
</details>

<details>
<summary>Classification</summary>
 
 All codes of deep neural networks which are used in this research exist at [classification](https://github.com/EnsiehKhazaei/Karyotype/blob/main/classification) directory. In addition, we put the confusion matrixes of all neural networks at [confusion_matrixes](https://github.com/EnsiehKhazaei/Karyotype/blob/main/classification/confusion_matrixes) folder. 

</details>

# Results
Our classifier is trained by a dataset of about 162,000 human chromosome images. The success rate of our segmentation algorithm is 95%. In addition, our experimental
results show that the accuracy of our classifier for human chromosomes is 92.63% and our novel post-processing algorithm increases the classification results to 94%.

# Contact
For questions about our paper or code, please contact [Ensieh Khazaei](mailto:khazaei1394@gmail.com) and [Ala Emrany](mailto:emranyala@gmail.com).


