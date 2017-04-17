# Vehicle Detection Project

The goals / steps of this project are the following:

1. Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images.
1. Apply a color transform and append binned color features, as well as histograms of color, to the HOG feature vector.
1. Train a SVM classifier.
1. Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
1. Run the pipeline on a video stream and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
1. Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[example]: ./output_images/example_image.png
[converted]: ./output_images/converted_image.png
[window0]: ./output_images/window_0.png
[window1]: ./output_images/window_1.png
[window2]: ./output_images/window_2.png
[window3]: ./output_images/window_3.png
[window4]: ./output_images/window_4.png
[window5]: ./output_images/window_5.png
[image4]: ./examples/sliding_window.jpg
[image5]: ./examples/bboxes_and_heat.png
[image6]: ./examples/labels_map.png
[image7]: ./examples/output_bboxes.png
[video1]: ./project_video.mp4

## Histogram of Oriented Gradients (HOG) feature extraction

The code for this step is contained in lines 42 through 72 of the file called `feature_extractor.py`.

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![Example image][example]

I then explored different color spaces and different `skimage.feature.hog()` parameters.  I grabbed images from each of the two classes and displayed them to get a feel for what the `skimage.feature.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:

![Image conversion][converted]

I tried various combinations of parameters and chose following convination:

- Color space: YCrCb
- Image channels used for the feature: All channels
- The number of orientations: 9
- The number of pixels per cell: 8
- The number of cells per block: 2

For the color space, I tried `RGB`, `HSV`, `HLS` and `LUV` other than `YCrCb`.  These color spaces achieved lower test accuracy of the classifier and got more false positives on vehicle detection compared with `YCrCb`.  Regarding the channel of the image, using single channel increased the number of false positives.  As for the remaining parameters, I chosed to reduce false positives on the classification.

## Binned color feature and histogram feature extraction

The code for this step is contained in lines 11 through 40 of the file called `feature_extractor.py`.

In addition to HOG feature, I decided to use color features, which are binned color feature and histogram feature, for my SVM classifier.  I converted each image to `LUV` color space in advance. For binned color feature, I converted each image to a `32 x 32` pixel image and used it as a feature.  Regarding the histogram feature, I calculated histogram of each color channnel with `32` bins, and use them for feature.  I tried different values and combinations of parameter. As a result, I found that these parameter could detect all the cars shown in the video.

## SVM classifier

The code for this step is contained in lines 84 through 157 of the file called `find_car.py`.

Before I train SVM, I extracted all the images into three features introduced above, and concatenated as a vector.  To prevent some feature from dominating in the following step, I normalized a feature vector usinge `sklearn.preprocessing.StandardScaler`.  After normalization, I randomly splited all the images into 80 percent for training set and 20 percent for test set.

Then I trained a `sklearn.svm.LinearSVC` using features conputed above.  As a result of training, the test accuracy was 98.2 percent.

## Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I decided to search the image with windows with 6 different scalse as shown below:

| Scale: 1.0           | Scale: 1.6           | Scale: 2.2           |
|----------------------|----------------------|----------------------|
| ![Window-0][window0] | ![Window-1][window1] | ![Window-2][window2] |

| Scale: 2.8           | Scale: 3.4           | Scale: 4.0           |
|----------------------|----------------------|----------------------|
| ![Window-3][window3] | ![Window-4][window4] | ![Window-5][window5] |

####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on two scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][image4]
---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video.mp4)


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps:

![alt text][image5]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![alt text][image6]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image7]



---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.
