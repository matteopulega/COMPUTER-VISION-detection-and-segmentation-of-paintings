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

### OPTIMIZED BLOCK-BASED CONNECTED COMPONENTS LABELING WITH DECISION TREES
In order to obtain the connected components in the image, we used the Optimized Block-based Connected Components Labeling with Decision Trees [2]. We made this choice because this algorithm is faster compared to the Haralick algorithm and perform better in identifying the components. This algorithm is implemented inside the opencv library, so we used this implementation.
The reason why this algorithm is faster is because it models the neighbourhood exploration with a Decision Table that will be converted into Optimal Decision Trees which allow to generate the code.

### PROBABILISTIC HOUGH TRANSFORM FOR STRAIGHT LINES DETECTION
Probabilistic Hough Transform [3] [4] is an optimization of the Hough Transform. It takes into account only a random subset of the points. The only thing we must be careful of is to lower the threshold since it uses fewer points.

### TOTALLY ARBITRARY 3D TEXTURE MAPPING
Once found the rectangle enclosing the picture, we need to rectify it. Due to the perspective, this sometimes leads to picture that have a dimension that is too tiny, therefore we need a method the adjust the size of the enclosing rectangle to obtain a good approximation of the size, based on the perspective.
This algorithm [5] tries to find a good approximation of the size of the rectangle through recursion.
First of all we need to find the 4 vertices of the rectangle, and we will name them from A to D starting from top left corner and moving anticlockwise. Through the intersection of lines AC and BD we find the center of the rectangle O. After that we have to compute the two vanishing points of the rectangle, we will call them V1 and V2. V1 can be computed as the intersection between lines BC and AD, while V2 can be computed as the intersection between AB and CD. Now we need to find the center of each edge of the rectangle as it would appear in 3D. This requires again lines intersection and we will name this points from i1 to i4:
•	i1 is the intersection between lines AB and OV1;
•	i2 is the intersection between lines BC and OV2;
•	i3 is the intersection between lines CD and OV1;
•	i4 is the intersection between lines AD and OV2.
With those new points we can construct 4 small rectangles, we will name them from R1 to R4:
•	R1 is (A, i1, O, i4);
•	R2 is (i1, B, i2, O);
•	R3 is (O, i2, C, i3);
•	R4 is (i4, O, i3, D).
Now we select one rectangle and we recursively apply the same procedure. When we find a rectangle small enough and then we multiply 2N times each measure of the rectangle to obtain the real measures of the rectangle in 3D (Figure 2).

