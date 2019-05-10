# Paintings Detection and Segmentation
A Computer Vision problem: detection and segmentation of paintings, without Convolutional Neural Networks

## Introduction
The aim of our work is to detect paintings inside frames captured in a museum. Since it’s an indoor situation, the luminance condition is not variable. But images presented a large illumination gradient with which we have to deal.
We tried to remove the distortion caused by the cameras in order to obtain straight lines even at the borders of the image.
Once found the components inside the image, we isolate one component at time and then we analyse the characteristics of the portion of the image to be able to tell paintings and other objects apart. In particular, since we know that all paintings are rectangular, we exploited some geometrical properties that helped us to crop them away of the frame.

## Related works
In order to be able to detect the paintings inside the image, we exploited some algorithms that were useful considering our approach.
Here we will present a list and then we will explain them, leaving for the approach section the explanation on how we use them:
•	Hough Transform-based Radial Distortion Correction (HTRDC) [1]
•	Adaptive Threshold
•	Optimized Block-based Connected Components Labeling with Decision Trees [2]
•	Canny Edge Detection
•	Probabilistic Hough Transform for straight lines detection [3] [4]
•	Totally Arbitrary 3D Texture Mapping [5]
