#!/usr/bin/env python

import glob
import os
from functools import reduce
import matplotlib.image as mpimg

from logging import getLogger
logger = getLogger(__name__)

def read_filename(directory, exts):
    """ Read files which have the extension from the directory

    Args:
        directory: A directory where files are read from
        exts: A list of extensions that the file has

    Return:
        A list of filenames
    """
    names = []
    for ext in exts:
        for name in glob.glob(directory + '/**/*.' + ext):
            names.append(name)

    if 0 == len(names):
        logger.debug('found no files')

    return names

def read_image(filenames):
    """ Read images

    Args:
        filenames: A list of the name of the images

    Return:
        A list of images
    """
    logger.debug('reading images...')
    images = []
    for name in filenames:
        image = mpimg.imread(name)
        images.append(image)
    logger.debug('read %d images', len(images))
    return images

def check_image_size(images):
    """ Return the most common image size

    Args:
        images: A list of images

    Return:
        The most common image size
    """
    shapes = {}
    for image in images:
        if image.shape in shapes:
            shapes[image.shape] += 1
        else:
            shapes[image.shape] = 1

    if len(shapes) > 0:
        msg = reduce(lambda acc, v: acc + ',' + str(v), shapes.keys())
        logger.debug('detected shapes: %s', msg)

    return max(shapes, key=lambda k: shapes[k])

def check_image_data_type(images):
    """ Return the most common data type of images

    Args:
        images: A list of images

    Return:
        The most common data type
    """
    dtypes = {}
    for image in images:
        if image.dtype in dtypes:
            dtypes[image.dtype] += 1
        else:
            dtypes[image.dtype] = 1

    if len(dtypes) > 0:
        msg = reduce(lambda acc, v: acc + ',' + str(v), dtypes.keys())
        logger.debug('detected dtypes: %s', msg)

    return max(dtypes, key=lambda k: dtypes[k])
