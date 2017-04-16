#!/usr/bin/env python
""" Feature extractor
"""
import numpy as np
import cv2
from skimage.feature import hog

from logging import getLogger
logger = getLogger(__name__)

def bin_spatial(image, size=(32, 32)):
    """ Compute binned color features

    Args:
        image: A image to be processed
        size: Size of converted image

    Return:
        The color features
    """
    return cv2.resize(image, size).ravel()

def color_hist(image, nbins=32, bins_range=(0, 256)):
    """ Compute color histogram features

    Args:
        image: A image to be processed
        nbins: The number of bins
        bins_range: The range of the bins

    Return:
        The histogram features
    """
    channel1_hist = np.histogram(image[:, :, 0], bins=nbins, range=bins_range)
    channel2_hist = np.histogram(image[:, :, 1], bins=nbins, range=bins_range)
    channel3_hist = np.histogram(image[:, :, 2], bins=nbins, range=bins_range)
    hist_features = np.concatenate((channel1_hist[0],
                                    channel2_hist[0],
                                    channel3_hist[0]))
    return hist_features

def get_hog_features(image, orient, pix_per_cell, cell_per_block,
                     vis=False, feature_vec=True):
    """ Return HOG features

    Args:
        image: A image to be processed
        orient: The number of orientation bins
        pix_per_cell: The cell size over which each gradient histogram is computed
        cell_per_block: The size of local area
        vis: Flag to output a HOG image
        feature_vec: A type of feature returned; True: 1D-array, False: 2D-array

    Return:
        The HOG features
    """
    if vis == True:
        features, hog_image = hog(image, orientations=orient,
                                  pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block),
                                  block_norm='L2-Hys',
                                  transform_sqrt=True,
                                  visualise=vis, feature_vector=feature_vec)
        return features, hog_image
    else:
        features = hog(image, orientations=orient,
                       pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block),
                       block_norm='L2-Hys',
                       transform_sqrt=True,
                       visualise=vis, feature_vector=feature_vec)
        return features

def convert_color_space(image, cspace='RGB'):
    """ Convert color space of a image from RGB to specified one

    Args:
        image: A image to be processed
        cspace: A color space for output image

    Return:
        A image converted
    """
    if cspace != 'RGB':
        if cspace == 'HSV':
            return cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        elif cspace == 'LUV':
            return cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
        elif cspace == 'HLS':
            return cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
        elif cspace == 'YUV':
            return cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
        elif cspace == 'YCrCb':
            return cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
    else:
        return np.copy(image)

def extract_features(images,
                     use_spatial=True, use_hist=True, use_hog=True,
                     cspace_color='YCrCb',
                     spatial_size=(32, 32),
                     hist_bins=32, hist_range=(0, 256),
                     cspace_hog='YCrCb',
                     orient=9, pix_per_cell=8,
                     cell_per_block=2, hog_channel='ALL'):
    """ Extract features from a list of images

    Args:
        images: A list of images to be processed
        use_spatial: A flag for extracting spatial features
        use_hist: A flag for extracting histogram features
        use_hog: A flag for extracting HOG features
        cspace_color: A color space into which the images are converted for color features
        spatial_size: Size of converted image
        hist_bins: The number of bins
        hist_range: The range of the bins
        cspace_hog: A color space into which the images are converted for hog features
        orient: The number of orientation bins
        pix_per_cell: The cell size over which each gradient histogram is computed
        cell_per_block: The size of local area
        hog_channel: The channel of image which is used to compute HOG features

    Returns:
        Conbined features
    """
    features = []
    for image in images:
        feature = []

        feature_image = convert_color_space(image, cspace=cspace_color)
        if use_spatial:
            spatial_features = bin_spatial(feature_image, size=spatial_size)
            feature.append(spatial_features)

        if use_hist:
            hist_features = color_hist(feature_image,
                                       nbins=hist_bins,
                                       bins_range=hist_range)
            feature.append(hist_features)

        if use_hog:
            feature_image = convert_color_space(image, cspace=cspace_hog)
            if hog_channel == 0 or hog_channel == 1 or hog_channel == 2:
                hog_features = get_hog_features(feature_image[:, :, hog_channel],
                                                orient=orient,
                                                pix_per_cell=pix_per_cell,
                                                cell_per_block=cell_per_block)
            else:
                hog_features = \
                [get_hog_features(feature_image[:, :, channel],
                                  orient=orient,
                                  pix_per_cell=pix_per_cell,
                                  cell_per_block=cell_per_block)
                 for channel in range(feature_image.shape[2])]
                hog_features = np.ravel(hog_features)

            feature.append(hog_features)

        features.append(np.concatenate(feature))

    return features

def main():
    """ Train SVM using color features
    """
    import time
    from sklearn.svm import LinearSVC
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    import setting
    import util


    fnames = util.read_filename('../data/vehicles/', ['png'])
    cars = util.read_image(fnames)
    util.check_image_size(cars)
    util.check_image_data_type(cars)

    fnames = util.read_filename('../data/non-vehicles/', ['png'])
    notcars = util.read_image(fnames)
    util.check_image_size(notcars)
    util.check_image_data_type(notcars)

    car_features = extract_features(cars)
    notcar_features = extract_features(notcars)

    assert len(car_features[0]) == len(notcar_features[0])
    logger.debug('%s features extracted', len(car_features[0]))

    x = np.vstack((car_features, notcar_features)).astype(np.float64)
    x_scaler = StandardScaler().fit(x)
    scaled_x = x_scaler.transform(x)

    y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

    rand_state = np.random.randint(0, 100)
    x_train, x_test, y_train, y_test = train_test_split(
        scaled_x, y, test_size=0.2, random_state=rand_state)

    clf = LinearSVC()

    logger.debug('train SVC...')
    t = time.time()
    clf.fit(x_train, y_train)
    t2 = time.time()
    logger.debug('takes %d seconds to finish training', round(t2 - t))
    logger.debug('test accuracy of SVC = %.4lf', round(clf.score(x_test, y_test), 4))

if __name__ == '__main__':
    main()
