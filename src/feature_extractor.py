#!/usr/bin/env python
""" Image classifier using color features
"""
import numpy as np
import cv2

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

def extract_features(images, cspace='RGB', spatial_size=(32, 32),
                     hist_bins=32, hist_range=(0, 256)):
    """ Extract features from a list of images

    Args:
        images: A list of images to be processed
        cspace: A color space into which the images are converted prior to processing
        spatial_size: Size of converted image
        hist_bins: The number of bins
        hist_range: The range of the bins

    """
    features = []
    for image in images:
        if cspace != 'RGB':
            if cspace == 'HSV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            elif cspace == 'LUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
            elif cspace == 'HLS':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
            elif cspace == 'YUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
        else:
            feature_image = np.copy(image)

        spatial_features = bin_spatial(feature_image, size=spatial_size)
        hist_features = color_hist(feature_image, nbins=hist_bins, bins_range=hist_range)
        features.append(np.concatenate((spatial_features, hist_features)))

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

    car_features = extract_features(cars, cspace='RGB', spatial_size=(32, 32),
                                    hist_bins=32, hist_range=(0, 256))
    notcar_features = extract_features(notcars, cspace='RGB', spatial_size=(32, 32),
                                       hist_bins=32, hist_range=(0, 256))

    x = np.vstack((car_features, notcar_features)).astype(np.float64)
    x_scaler = StandardScaler().fit(x)
    scaled_x = x_scaler.transform(x)

    y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

    rand_state = np.random.randint(0, 100)
    x_train, x_test, y_train, y_test = train_test_split(
        scaled_x, y, test_size=0.2, random_state=rand_state)

    cls = LinearSVC()

    logger.debug('train SVC...')
    t = time.time()
    cls.fit(x_train, y_train)
    t2 = time.time()
    logger.debug('takes %d seconds to finish training', round(t2 - t))
    logger.debug('test accuracy of SVC = %.4lf', round(cls.score(x_test, y_test), 4))

if __name__ == '__main__':
    main()
