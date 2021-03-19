# SegSalad - A Hands On Review of Semantic Segmentation Techniques for Weed/Crop Datasets
Source code is in [this Jupyter Notebook](SegSalad.ipynb) which is also avaiable on Google Colab [here](https://colab.research.google.com/drive/1Xmzz54j1JgksESurdbqAeGQFGnHiCQU9?usp=sharing).
## Abstract
In this project, I explore a few common semantic segmentation techniques that have been used on weed/crop datasets for precision agriculture tasks. I use image patching as a data augmentation technique, train with transfer learning for three different semantic segmentation neural network architecures, and compare these models qualitatively and with the Intersection Over Union metric. I discuss the challenges and takeaways of the project. For a video summary of this project, please click [here](https://youtu.be/FyGz-Pb-K2k).


## Problem and Motivation
A very exciting application of computer vision to me is in precision agriculture. Precision agriculture is an industry aimed at increasing yield of crops while decreasing use of land, water, chemicals, and labor. By doing so, precision agriculture aims to make farming more sustainable and tolerant to climate change. A few of the usages of computer vision in precision agriculture are identification of weeds and crops for targeted pesticide application, automated harvesting, and plant health monitoring. For this project, I will focus on semantic segmentation of weeds and crops, which is most applicable to the first example usage mentioned.

## Dataset and Data Augmentation
The dataset that I used is the CWFID Carrot/Weed image dataset (1). This dataset is available [here](https://github.com/bhimar/cwfid-dataset) in a repo that I forked from the original. This dataset contains 60 high resolution images of carrot plants and weeds from an organic carrot farm, and it intended to be used as a benchmarking dataset for precision agriculture vision tasks. For each image, the dataset has a corresponding annotation in which the pixels in the red channel correspond to the weeds, and the pixels in the green channel correspond to the crops. The background of the annotation was originally black, 0 in all RGB channels, but I modified them so that the background is labeled in the blue channel. I have visualized a few images from the dataset below:
![alt text](writeup/image_annotation_comparison.JPG?raw=true)

These images are very high resolution at 996 x 1296 pixels which would hinder the efficiency of training models. Because of the small size of the dataset and the high resolution, I split the images for training into patches of size 224 x 224 as shown below. This data augmentation method was first employed by Brilhador and colleagues (2). Notice, there is a slight cropping involved to take exact patches from the original images. I have visualized a patched image and annotation below. Each of the patches is fed as a single example to the models for training.

![alt text](writeup/slices.JPG?raw=true)

I split the original 60 images into the training and test sets using a random 80/20 split. There were 48 images in the training set and 12 images in the testing set. I used the same training and testing sets for all models. The images were normalized according to the following: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225] for compatability with the PyTorch pretrained models.

## Semantic Segmentation Models
I trained 3 semantic segmentation models for this project: Fully Convolutional Network (FCN), Google DeepLabv3, and UNet. Each of these models had a ResNet50 backbone. I chose these models because Khan and collegues had compared similar models on several weed/crop datasets including the CWFID dataset (3). Each of these models was trained on Google Colab GPU for 10 epochs, with the same learning rate configuation of 0.001 for the first 5 epochs and 0.0001 for the last 5 epochs. I used the Adam optimizer for training. I adapted the training  and checkpoint code from a PyTorch tutorial from my Computer Vision class, delivered by Joe Redmon.

I used transfer learning to train each of these models. I used the FCN and DeepLabv3 models from [torchvision](https://pytorch.org/vision/stable/models.html), which were pretrained on the COCO 2017 dataset. I used the Unet model from [Segmentation Models Pytorch](https://smp.readthedocs.io/en/latest/), which was pretrained on ImageNet. Because of the slight differences between implementation for the torchvision and Segmentation Models Pytorch models, there is a separate training and evaluation function for Unet which are very similar to the functions used for the FCN and DeepLabv3 models.

Because I was able to represent the annotations as a 3 channel RGB image with each channel corresponding to a class, I used Mean Squared Error loss while training. The output of the networks was a 3 channel image, where the maximum value in the channel corresponds to the pixel class prediction.

## Model Evaluation
To evalute the models, I use Intersection Over Union as a metric. The results from evaluating on the testing set are:

| Model                | Crop IOU | Weed IOU | Soil IOU |
| -------------------- | -------- | -------- | -------- |
| FCN                  | 42.79%   | 66.78%   | 97.66%   |
| DeepLabv3            | 20.32%   | 67.55%   | 97.86%   |
| Unet                 | 49.45%   | 75.21%   | 98.38%   |

We see that the Unet performs better on all classes than the FCN and DeepLabv3 models, which is expected from the results found by Khan and collegues (3). However, the crop IOU was much lower for all models, and the weed IOU was much higher for all models than the results reported by Khan and collegues. One explanation for this might be that 10 epochs was too much fine tuning on the models, which resulted in overfitting to the training set. Because there are a lot more weed labeled pixels than crop labeled pixels, the models might be more sensitive to detecting weeds.

Below I have visualized an example from the test set and the predictions from each of the networks.

![alt text](writeup/model_comparison.JPG?raw=true)

This is only a single example, so we cannot make any definitive conclusions, but we do see that the DeepLabv3 model is not detecting the crops well, and the Unet and FCN models are reporting some weed and soil pixels as crops. All three models are quite good at detecting vegitation, but it seems that differentiating between the weeds and crops remains a difficult task (even for the untrained naked eye). Notice that there is some cropping for the Unet because I had to patch the testing set as well to feed into the network for evaluation, then reconstruct the prediction to compare to the annotation.

## Takeaways and Future Work
Throughout this project I learned about common semantic segmentation models, gained experience working with image data and training neural networks in PyTorch, and explored applications of computer vision in precision agriculture. Future work on this project may include exploring other network architectures, implementing and training networks from scratch, and exploring non-deep learning based semantic segmentation techniques. I may also train models on some other image datasets such as [sugarbeets](https://www.ipb.uni-bonn.de/data/sugarbeets2016/) and [rice seedlings](https://figshare.com/articles/dataset/rice_seedlings_and_weeds/7488830). When using one of these larger datasets, I would also include a validation set to measure generalization performance while training to avoid overfitting. I really enjoyed this project, and I am excited to keep working on it and/or explore other applications of computer vision!

## References
1. Haug S., Ostermann J. (2015) A Crop/Weed Field Image Dataset for the Evaluation of Computer Vision Based Precision Agriculture Tasks. In: Agapito L., Bronstein M., Rother C. (eds) Computer Vision - ECCV 2014 Workshops. ECCV 2014. Lecture Notes in Computer Science, vol 8928. Springer, Cham. https://doi.org/10.1007/978-3-319-16220-1_8
2. A. Brilhador, M. Gutoski, L. T. Hattori, A. de Souza Inácio, A. E. Lazzaretti and H. S. Lopes, "Classification of Weeds and Crops at the Pixel-Level Using Convolutional Neural Networks and Data Augmentation," 2019 IEEE Latin American Conference on Computational Intelligence (LA-CCI), Guayaquil, Ecuador, 2019, pp. 1-6, doi: 10.1109/LA-CCI47412.2019.9037044.
3. Khan A, Ilyas T, Umraiz M, Mannan ZI, Kim H. CED-Net: Crops and Weeds Segmentation for Smart Farming Using a Small Cascaded Encoder-Decoder Architecture. Electronics. 2020; 9(10):1602. https://doi.org/10.3390/electronics9101602
