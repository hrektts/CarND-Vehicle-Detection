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
[bbox1]: ./output_images/test1_bbox.png
[bbox2]: ./output_images/test2_bbox.png
[bbox3]: ./output_images/test3_bbox.png
[bbox4]: ./output_images/test4_bbox.png
[heat1]: ./output_images/test1_heat.png
[heat2]: ./output_images/test2_heat.png
[heat3]: ./output_images/test3_heat.png
[heat4]: ./output_images/test4_heat.png
[bbox501]: ./output_images/flame_501.png
[bbox502]: ./output_images/flame_502.png
[bbox503]: ./output_images/flame_503.png
[bbox504]: ./output_images/flame_504.png
[heat501]: ./output_images/flame_501_heat.png
[heat502]: ./output_images/flame_502_heat.png
[heat503]: ./output_images/flame_503_heat.png
[heat504]: ./output_images/flame_504_heat.png


## Histogram of Oriented Gradients (HOG) feature extraction

The code for this step is contained in the fourth code cell of the IPython notebook.

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

For the color space, I tried `RGB`, `HSV`, `HLS` and `LUV` other than `YCrCb`.  These color spaces achieved lower test accuracy on the classifier and got more false positives on vehicle detection compared with `YCrCb`.  Regarding the channel of the image, using single channel increased the number of false positives.  As for the remaining parameters, I chosed to reduce false positives on the classification.

## Binned color feature and histogram feature extraction

The code for this step is contained in the fourth code cell of the IPython notebook.

In addition to HOG feature, I decided to use color features, which are binned color feature and histogram feature, for my SVM classifier.  I converted each image to `YCrCb` color space in advance. For binned color feature, I converted each image to a `32 x 32` pixel image and used it as a feature.  Regarding the histogram feature, I calculated histogram of each color channnel with `32` bins, and use them for feature.  Though I tried different values and combinations of parameter, I found that these parameter could detect all the cars shown in the video.

## SVM classifier

The code for this step is contained in the fifth code cell of the IPython notebook.

Before I train SVM, I extracted all the images into three features introduced above, and concatenated as a vector.  To prevent some feature from dominating in the following step, I normalized a feature vector usinge `sklearn.preprocessing.StandardScaler`.  After normalization, I randomly splited all the images into 90 percent for training set and 10 percent for test set.

Then I trained a `sklearn.svm.LinearSVC` using features conputed above.  I tried several `C` values to avoid overfitting and found that small `C` values increase false positives. Consequently, I use `1.0` for `C` of the `LinearSVC`. As a result of training, the test accuracy was 98.4 percent.

## Sliding Window Search

The code for this step is contained in the sixth code cell of the IPython notebook.

I decided to search the image with windows as shown below:

![Window-0][window0]

Each window was overlapped by 75 percent.  I decided the scale of the window and search scope as above to cover the area where cars would be present.  The sliding window search provided some false positives, and I ignored these by the number of overlapes.  Concretely, I ignored places where the overlap is `2` or less.

As a result, I got images as below:

| Bounding box      | Heatmap           |
|-------------------|-------------------|
| ![BBox-1][bbox1] | ![H-map-1][heat1] |
| ![BBox-2][bbox2] | ![H-map-2][heat2] |
| ![BBox-3][bbox3] | ![H-map-1][heat3] |
| ![BBox-4][bbox4] | ![H-map-1][heat4] |

---

## Video Implementation

Here's a [link to my video result](./output_images/project_video.mp4)

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  In addition to these, I applied exponential smoothing to ignore false positives.  This is based on the feature that false positives are less likely to be detected in consecutive frames.

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes:

| Flame # | Bounding box             | Heatmap               |
|---------|--------------------------|-----------------------|
| 501     | ![BBox-501][bbox501]     | ![H-map-501][heat501] |
| 502     | ![BBox-502][bbox502]     | ![H-map-502][heat502] |
| 503     | ![BBox-503][bbox503]     | ![H-map-503][heat503] |
| 504     | ![BBox-504][bbox504]     | ![H-map-504][heat504] |

---

## Discussion

My sliding window search with SVM classifier did not work well without frame by frame compensation.  Compensation using several video frames worked well, however, this aproach provided delay because the compensation works as low pass filter.  I think this have to be improved if accute detections are needed.
