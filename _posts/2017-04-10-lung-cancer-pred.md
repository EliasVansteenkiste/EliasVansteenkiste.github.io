---
layout: post
title: Predicting lung cancer
description: "Description of our solution for Kaggle's third Data Science Bowl. "
modified: 2016-12-02
tags: [lung cancer, transfer learning, convolutional networks, imagenet, Kaggle, Data Science Bowl]
categories: [machine learning]
image:
    feature: feature.jpg
    credit: dargadgetz
    creditlink: http://www.dargadgetz.com/ios-8-abstract-wallpaper-pack-for-iphone-5s-5c-and-ipod-touch-retina/
---

The National Data Science Bowl is an annual data science competition hosted by Kaggle. In this year's edition the goal was to predict lung cancer based on CT scans of the chest a year before diagnosis.
We formed a team with 
The competition just finished and our team finished ??! In this post, we explain our approach.

The Deep Breath team consists of Jonas Degrave, Matthias Freiberger, Fréderic Godin, Ira Korshunova, Lionel Pigou, Elias Vansteenkiste and Andreas Verleysen (alphabetical order). We are all PhD students and post-docs at Ghent University. 


<figure class="center threequart">
<a href="http://www.cancerresearchuk.org/health-professional/cancer-statistics/worldwide-cancer/mortality#heading-One">
<img src="{{ site.url }}/images/common_cause_cancer_death.jpg" alt=""></a>
<figcaption>The 10 Most Common Causes of Cancer Death (Credit: Cancer Research UK) </figcaption>
</figure>


# Introduction
Lung cancer is the most common cause of cancer death worldwide. Second to breast cancer, it is also the most common form of cancer. To prevent lung cancer deaths, high risk individuals are being screened with low-dose CT scans, because early detection [doubles the survival rate of lung cancer patients](https://www.cancer.org/cancer/small-cell-lung-cancer/detection-diagnosis-staging/survival-rates.html). So it makes a lot of sense to try to predict lung cancer for high risk patients. Additionally analyzing CT-scans is a time consuming job for radiologists and automating the process can help to make it more affordable.

To predict lung cancer starting from a CT scan of the chest, the overall strategy was to reducing the high dimensional CT scan to a few regions of interest. Starting from these regions of interest we tried to predict lung cancer. In what follows we will explain how we trained several networks to extract the region of interests and to make final prediction starting from the regions of interest.
This post is pretty long, so here is a clickable overview of different sections if you want to skip ahead:

* [The Needle in The Haystack](#the-needle-in-the-haystack)
* [Nodule Detection](#nodule-detection)
   1. [Nodule Segmentation](#nodule-segmentation)
   2. [Lung Segmentation](#lung-segmentation)
   3. [Blob Detection](#blob-detection)
* [False Positive Reduction](#false-positive-reduction)
* [Malignancy Prediction](#malignancy-prediction)
* [Lung Cancer Prediction](#lung-cancer-prediction)
   1. [Transfer learning](#transfer-learning)
   2. [Aggregating Nodule Predictions](#aggregating-nodule-predictions)
   3. [Ensembling](#ensembling)


# The Needle in The Haystack
{::comment}
Explain why it is a needle in the haystack problem
{:/comment}
To determine if someone will develop lung cancer, we have to look for early stages of malignant pulmonary nodules. Finding early stage malignant nodule in the CT scan of a lung is like finding a needle in the haystack. To support this statement, let's take a look at a few examples of malignant nodules in the LIDC/IDRI data set from the [LUng Node Analysis Grand Challenge (LUNA)](https://luna16.grand-challenge.org/) dataset. We used this dataset extensively in our approach, because it contains detailed annotations from radiologists.

//figuur van wat nodules

The radius of the average malicious nodule in the LUNA dataset is 4.8 mm and the average lung photo captures a volume of 400mm x 400mm x 400mm. So we are looking for a feature that is almost a million times smaller than the input volume and this feature determines the classification for the whole input volume. This makes analyzing CT scans an enormous burden for radiologists and a difficult task for conventional classification algorithms using convolutional networks. 

This problem is even worse in our case because we have to try to predict lung cancer starting from a CT scan a year before diagnosis. In the LUNA dataset the patients are already diagnosed with lung cancer. In our case the patients may not yet have developed a malignant nodule. So it is reasonable to assume that training directly on the data and labels from the competition wouldn't work, but we tried it anyway and observed that the network doesn't learn more than the bias in the training data.


# Nodule Detection
## Nodule Segmentation
To reduce the amount of information in the scans, we first tried to detect pulmonary nodules. 
We built a network for segmenting the nodules in the input scan. The LUNA dataset contains annotations for each nodule in a patient. These annotations contain the location and diameter of the nodule. We used this information to train our segmentation network. 

64x64x64 mm patches are cut out of the CT scan and fed to the input of the segmentation network. The ground truth are 32x32x32 mm binary masks. For each voxel they indicate if the voxel is inside the nodule. The masks are constructed by using the diameters in the nodule annotations. 

{% highlight python %} 
intersection = T.sum(y_true * y_pred, axis=(1, 2, 3))
dice = (2. * intersection + epsilon) / 
        (T.sum(y_true, axis=(1, 2, 3)) + T.sum(y_pred, axis=(1, 2, 3)) + epsilon)
{% endhighlight %}

As an objective we choose to directly optimize the Dice coefficient, commonly used for image segmentation. It handles the imbalance between the voxels in- and outside the nodule well. This is particularly important for the smaller nodules, which are also interesting in case of the early stage cancer detection.

The downside of using the Dice coefficient is that the 
We also apply translation and rotation augmentations


We applied a translation and rotation augmentation. 

We start at the center of the nodule and applied a random translation and rotation
 a random translation between -12 and 12 mm in each dimension.   and a random augmentation.  For each nodule wWe cut out  is trained on
The segmentation network architecture is based on U-Net, but with less pooling. (TODO REASON?) 

<figure>
<a href="{{ site.url }}/images/nodule_segnet.jpg">
<img src="{{ site.url }}/images/nodule_segnet.jpg" alt=""></a>
<figcaption>A schematic of the segmentation network architecture, the tensor shapes and network operations</figcaption>
</figure>


## Blob Detection

## Lung Segmentation
The nodule segmentation network generates a lot of candidate nodules that will be filtered by the false positive reduction (FPR) network. However a number of those candidates are lying outside the lung itself and therefore should be filtered out before they are send through the FPR network. 

At first, we used a similar strategy as proposed in the Kaggle Tutorial using a number of morphological operations such as erosion and dilation. After visual inspection of the segmentations, we noticed that the quality of the segmentation was too dependent on the size of the morphological operations. On top of that, morphological operations with a large diameter are very slow. A second observation we made was that 2D segmentation only works well on regular slice of the lung. Whenever there are more than 2 cavities, it’s not clear anymore if that cavity is part of the lung which is actually the main reason we apply this step.

Our final approach was a fast 3D approach which focused on removing non-lung cavities that could contain nodules. Hence we opted for a simple convex hull of the lungs based on the center slice of the CT scan. Because we work in 3D, lung regions are always connected. Finally, we removed all other cavities that are part of the convex hull but that are not connected to the lung. 




# False Positive Reduction
False positive Reduction of the 
## Architecture
based on inception resnet v2 architecture
some

# Malignancy Prediction
It was only in the final 2 weeks of the competition that we discovered the existence of malignancy labels for the nodules in the LUNA dataset. The discussions on the forum mainly focussed on the LUNA dataset but it was only when we trained a model to predict the malignancy of the individual nodules/patches that we were able to get close to the top scores on the LB.

The network we used was exactly the same as the FPR network, namely a resnet-v2 architecture. We rescaled the malignancy labels between 0 and 1 to create a probability label. We constructed a training set by sampling an equal amount of candidate nodules that did not have a malignancy label in the LUNA dataset.

As the objective function, we used the Mean Squared Error (MSE) loss which showed to work better than a binary cross-entropy objective function. 

# Lung Cancer Prediction
After we ranked the candidate nodules with the false positive reduction network and trained a malignancy prediction network, we are finally able to train a network for lung cancer prediction on the Kaggle dataset.


## Transfer learning 
After training a number of different architectures from scratch, we realized that we needed better ways of inferring good features. Although we reduced the full CT scan to a number of regions of interest, the number of patients is still low so the number of malignant nodules is still low. Therefore, we focussed on initializing the networks with pre-trained weights. 

The transfer learning idea is quite popular in image classification tasks with RGB images where the majority of the transfer learning approaches use a network trained on the ImageNet dataset to as the convolutional layers of their network. Hence, good features are learned on a big dataset and are then reused (transferred) as part of another neural network/another classification task. However, for CT scans we did not have access to such a pretrained network so we needed to train one ourselves.

At first, we used the the fpr network which already gave some improvements. Subsequently, we trained a network to predict the size of the nodule because that was also part of the LUNA dataset. In both cases, our main strategy was to reuse the convolutional layers but to randomly initialize the dense layers. 

In the final weeks, we used the full malignancy network to start from and only added an aggregation layer on top of it. However, we retrained all layers anyway. Somehow logical, this was the best solution.

## Aggregating Nodule Predictions
We tried several approaches to combine the malignancy predictions of the nodules. We highlight the 2 most successful aggregation strategies:
* **P_patient_cancer = 1 - M P_nodule_benign**: The idea behind this aggregation is that the probability of having cancer is equal to 1 if all of the selected nodules are benign. If one nodule is classified as malignant, P_patient_cancer will be one.
The problem with this approach is that it doesn't behave well when the malignancy prediction network is convinced one of the nodules is malignant. Once the network is correctly predicting that the network one of the nodules is malignant, the learning stops. 
* **Log Mean Exponent**: The idea behind this aggregation strategy is that the cancer probability is determined by the most malignant/the least benign nodule. The LME aggregation works as soft version of a max operator.  As the name suggest, it exponential blows up the predictions of the individual nodule predictions, hence focussing on the largest(s) probability(s). Compared to a simple max function, this function also allows backpropagating through the networks of the other predictions. 


## Ensembling
Andreas schrijft een stukje over ensembling

# Final Thoughts
Weird leaderboard

