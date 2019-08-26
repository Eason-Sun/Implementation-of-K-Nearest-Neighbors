# Implementation-of-K-Nearest-Neighbors

## Objective:
This project implements the K Nearest Neighbors algorithm from scratch (without using any existing machine learning libraries e.g. sklearn) for optical recognition of handwritten digits dataset.   

The hyperparameter k (the number of nearest neighbours) is fine-tuned using 10-Fold Cross Validataion which is also implemented from scratch.

Note: sklearn packages are used in this project for verification and comparision purposes.

## Dataset:
Link: http://archive.ics.uci.edu/ml/datasets/Optical+Recognition+of+Handwritten+Digits  

The data used for this project is a modified version of the Optical Recognition of Handwritten Digits Dataset from the UCI repository. 
It contains pre-processed black and white images of the digits 5 and 6. Each attribute indicates how many pixels are black in a patch of 4 x 4 pixels.

### Format: 
There is one row per image and one column per attribute. The class labels are 5 and 6. The training set is already divided into 10 subsets for 10-fold cross validation.

## Classification Accuracy Results from 10-Fold Cross Validation:

K increases from 1 to 30.

![Capture](https://user-images.githubusercontent.com/29167705/63726919-a989b000-c82c-11e9-8037-73c6dc9d203c.JPG)


## Visualization:

![Capture](https://user-images.githubusercontent.com/29167705/63727621-e656a680-c82e-11e9-8d88-7cd10e86294a.JPG)
