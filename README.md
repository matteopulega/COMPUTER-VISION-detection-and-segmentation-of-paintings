# PAINTING DETECTION AND SEGMENTATION
A Computer Vision problem: detection and segmentation of paintings, without Convolutional Neural Networks

## INTRODUCTION
The aim of our work is to detect paintings inside frames captured in a museum. Since it’s an indoor situation, the luminance condition is not variable. But images presented a large illumination gradient with which we have to deal.
We tried to remove the distortion caused by the cameras in order to obtain straight lines even at the borders of the image.
Once found the components inside the image, we isolate one component at time and then we analyse the characteristics of the portion of the image to be able to tell paintings and other objects apart. In particular, since we know that all paintings are rectangular, we exploited some geometrical properties that helped us to crop them away of the frame.

## RELATED WORKS
In order to be able to detect the paintings inside the image, we exploited some algorithms that were useful considering our approach.
Here we will present a list and then we will explain them, leaving for the approach section the explanation on how we use them:
•	Hough Transform-based Radial Distortion Correction (HTRDC) [1]
•	Adaptive Threshold
•	Optimized Block-based Connected Components Labeling with Decision Trees [2]
•	Canny Edge Detection
•	Probabilistic Hough Transform for straight lines detection [3] [4]
•	Totally Arbitrary 3D Texture Mapping [5]


### HOUGH TRANSFORM-BASED RADIAL DISTORTION CORRECTION
Radial distortion is a very big problem for our approach because it can make straight lines appear as curves. This is very bad since we use straight lines detection to find the 4 sides of the picture.
To correct the radial distortion we implemented the HTRDC algorithm [1] which uses the Hough Transform for straight lines in order to find the best distortion coefficient k that will be used to correct the image.
The resulting image will be cropped since not all original coordinates will be mapped in the undistorted image (Figure 1).
This is an iterative method which will stop when the range of k in which to search is lower than a fixed threshold. Since we have some prior knowledge, we narrow the range from 0 to 1*10-4, causing the algorithm to converge faster.
![Figure 1](https://github.com/matteopulega/Paintings-Detection-and-Segmentation/blob/master/otherImages/01.jpg) ![Figure 1](https://github.com/matteopulega/Paintings-Detection-and-Segmentation/blob/master/otherImages/02.jpg)
