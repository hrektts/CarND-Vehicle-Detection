#!/usr/bin/env python
""" Vehicle Detection Project
"""
import getopt
import glob
import os
import pickle
import sys
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from moviepy.editor import VideoFileClip
import feature_extractor as fe
import util

from logging import getLogger
logger = getLogger(__name__)

class ImageProcessor:
    """ A image processor
    """
    def __init__(self, pos_data_dir, neg_data_dir, data_dir, out_img_dir):
        """ The initializer

        Args:
            pos_data_dir: A path to a directory which contains positive images
            neg_data_dir: A path to a directory which contains negative images
            data_dir: A path to a directory which contains data files
            out_img_dir: A directory where processed images are written
        """
        self.pos_data_dir = pos_data_dir
        if not os.path.exists(pos_data_dir):
            os.mkdir(pos_data_dir)

        self.neg_data_dir = neg_data_dir
        if not os.path.exists(neg_data_dir):
            os.mkdir(neg_data_dir)

        self.data_dir = data_dir
        if not os.path.exists(data_dir):
            os.mkdir(data_dir)

        self.out_img_dir = out_img_dir
        if not os.path.exists(out_img_dir):
            os.mkdir(out_img_dir)

        self.pos_data_exts = ['png']
        self.neg_data_exts = ['png']
        self.data_file = 'clf.p'

        self.scales = None
        self.ystarts = None
        self.ystops = None

        self.params = {}
        self.params['use_spatial'] = True
        self.params['use_hist'] = True
        self.params['use_hog'] = True
        self.params['cspace_color'] = 'YCrCb'
        self.params['spatial_size'] = (32, 32)
        self.params['hist_bins'] = 32
        self.params['hist_range'] = (0, 256)
        self.params['cspace_hog'] = 'YCrCb'
        self.params['orient'] = 9
        self.params['pix_per_cell'] = 8
        self.params['cell_per_block'] = 2
        self.params['hog_channel'] = 'ALL'

    def train_classifier(self):
        """ Train SVC
        """
        fnames = util.read_filename(self.pos_data_dir, self.pos_data_exts)
        posimgs = util.read_image(fnames)
        fnames = util.read_filename(self.neg_data_dir, self.neg_data_exts)
        negimgs = util.read_image(fnames)

        pos_features = fe.extract_features(
            posimgs, scale=1.0,
            use_spatial=self.params['use_spatial'],
            use_hist=self.params['use_hist'],
            use_hog=self.params['use_hog'],
            cspace_color=self.params['cspace_color'],
            spatial_size=self.params['spatial_size'],
            hist_bins=self.params['hist_bins'],
            hist_range=self.params['hist_range'],
            cspace_hog=self.params['cspace_hog'],
            orient=self.params['orient'],
            pix_per_cell=self.params['pix_per_cell'],
            cell_per_block=self.params['cell_per_block'],
            hog_channel=self.params['hog_channel'])

        neg_features = fe.extract_features(
            negimgs, scale=1.0,
            use_spatial=self.params['use_spatial'],
            use_hist=self.params['use_hist'],
            use_hog=self.params['use_hog'],
            cspace_color=self.params['cspace_color'],
            spatial_size=self.params['spatial_size'],
            hist_bins=self.params['hist_bins'],
            hist_range=self.params['hist_range'],
            cspace_hog=self.params['cspace_hog'],
            orient=self.params['orient'],
            pix_per_cell=self.params['pix_per_cell'],
            cell_per_block=self.params['cell_per_block'],
            hog_channel=self.params['hog_channel'])

        assert len(pos_features[0]) == len(neg_features[0])
        logger.debug('%s features extracted', len(pos_features[0]))
        x = np.vstack((pos_features, neg_features)).astype(np.float64)
        x_scaler = StandardScaler().fit(x)
        scaled_x = x_scaler.transform(x)

        y = np.hstack((np.ones(len(pos_features)), np.zeros(len(neg_features))))

        x_train, x_test, y_train, y_test = train_test_split(
            scaled_x, y, test_size=0.2, random_state=np.random.randint(0, 100))

        clf = LinearSVC()

        logger.debug('train SVC...')
        clf.fit(x_train, y_train)
        accuracy = round(clf.score(x_test, y_test), 4)
        logger.debug('test accuracy of SVC = %.4lf', accuracy)

        self.params['num_pos_samples'] = len(posimgs)
        self.params['pos_img_size'] = util.check_image_size(posimgs)
        self.params['pos_image_data_type'] = util.check_image_data_type(posimgs)
        self.params['pos_img_example'] = posimgs[0]

        self.params['num_neg_samples'] = len(negimgs)
        self.params['neg_img_size'] = util.check_image_size(negimgs)
        self.params['neg_image_data_type'] = util.check_image_data_type(negimgs)
        self.params['neg_img_example'] = negimgs[0]

        self.params['clf'] = clf
        self.params['scaler'] = x_scaler
        self.params['test_accuracy'] = accuracy

        fname = self.data_dir + '/' + self.data_file
        with open(fname, mode='wb') as f:
            pickle.dump(self.params, f)
            logger.debug('classifier data is saved as %s', fname)

    def process_test_imgs(self):
        """ TODO: Add docstring
        """
        clf_param_file = self.data_dir + '/' + self.data_file
        if os.path.exists(clf_param_file):
            with open(clf_param_file, mode='rb') as f:
                self.params = pickle.load(f)
                logger.debug('classifier parameters are read from a file')
        else:
            self.params = self.train_classifier()

    def draw_window(self, image):
        tmp_scales = self.scales
        tmp_ystarts = self.ystarts
        tmp_ystops = self.ystops

        for scale, ystart, ystop in zip(tmp_scales, tmp_ystarts, tmp_ystops):
            self.scales = [scale]
            self.ystarts = [ystart]
            self.ystops = [ystop]

            out = self.find_object(image, test=True)
            fig = plt.figure()
            plt.imshow(out)
            plt.show()

        self.scales = tmp_scales
        self.ystarts = tmp_ystarts
        self.ystops = tmp_ystops

    def find_object(self, image, test=False):
        """ Find objects from a image using classifier trained in advance

        Args:
            image: A image to be processed

        Return:
            A image processed
        """
        cspace_hog = self.params['cspace_hog']
        cspace_color = self.params['cspace_color']
        orient = self.params['orient']
        pix_per_cell = self.params['pix_per_cell']
        cell_per_block = self.params['cell_per_block']
        window = self.params['pos_img_size'][0]
        spatial_size = self.params['spatial_size']
        hist_bins = self.params['hist_bins']
        hist_range = self.params['hist_range']
        hog_channel = self.params['hog_channel']
        scaler = self.params['scaler']
        clf = self.params['clf']
        use_spatial = self.params['use_spatial']
        use_hist = self.params['use_hist']
        use_hog = self.params['use_hog']

        draw_image = np.copy(image)
        # convert jpeg value to png value
        image = image.astype(np.float32)/255
        for scale, ystart, ystop in zip(self.scales, self.ystarts, self.ystops):
            img_tosearch = image[ystart:ystop, :, :]
            if scale != 1:
                imshape = img_tosearch.shape
                img_tosearch = cv2.resize(
                    img_tosearch,
                    (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))
            ctrans_for_hog = fe.convert_color_space(img_tosearch,
                                                    cspace=cspace_hog)
            ctrans_for_color = fe.convert_color_space(img_tosearch,
                                                      cspace=cspace_color)

            '''
            hog_features = fe.get_hog_features(
                ctrans_for_hog, orient=orient, pix_per_cell=pix_per_cell,
                cell_per_block=cell_per_block, hog_channel=hog_channel,
                feature_vec=False)
            print(hog_features.shape)
            '''

            nxblocks = (img_tosearch.shape[1] // pix_per_cell) - 1
            nyblocks = (img_tosearch.shape[0] // pix_per_cell) - 1

            nblocks_per_window = (window // pix_per_cell) - 1
            cells_per_step = 2
            nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
            nysteps = (nyblocks - nblocks_per_window) // cells_per_step

            hog_feature_image0 = fe.get_hog_features(
                ctrans_for_hog, orient, pix_per_cell,
                cell_per_block, 0, feature_vec=False)
            hog_feature_image1 = fe.get_hog_features(
                ctrans_for_hog, orient, pix_per_cell,
                cell_per_block, 1, feature_vec=False)
            hog_feature_image2 = fe.get_hog_features(
                ctrans_for_hog, orient, pix_per_cell,
                cell_per_block, 2, feature_vec=False)

            for xb in range(nxsteps):
                for yb in range(nysteps):
                    ypos = yb*cells_per_step
                    xpos = xb*cells_per_step
                    # Extract HOG for this patch
                    hog_feat0 = hog_feature_image0[ypos:ypos+nblocks_per_window,
                                                   xpos:xpos+nblocks_per_window].ravel()
                    hog_feat1 = hog_feature_image1[ypos:ypos+nblocks_per_window,
                                                   xpos:xpos+nblocks_per_window].ravel()
                    hog_feat2 = hog_feature_image2[ypos:ypos+nblocks_per_window,
                                                   xpos:xpos+nblocks_per_window].ravel()
                    hog_features = np.hstack((hog_feat0, hog_feat1, hog_feat2))

                    xleft = xpos*pix_per_cell
                    ytop = ypos*pix_per_cell
                    subimg = cv2.resize(
                        ctrans_for_color[ytop:ytop+window, xleft:xleft+window],
                        (64,64))

                    spatial_features = fe.bin_spatial(subimg, size=spatial_size)
                    hist_features = fe.color_hist(subimg, nbins=hist_bins,
                                                  bins_range=hist_range)

                    feature = []
                    if use_spatial:
                        feature.append(spatial_features)
                    if use_hist:
                        feature.append(hist_features)
                    if use_hog:
                        feature.append(hog_features)

                    #feature = fe.extract_features([subimg])

                    features_scaled = scaler.transform(np.concatenate(feature).reshape(1, -1))
                    pred = clf.predict(features_scaled)

                    if test or pred == 1:
                        xbox_left = np.int(xleft*scale)
                        ytop_draw = np.int(ytop*scale)
                        win_draw = np.int(window*scale)
                        if test:
                            if xb == 0 and yb == 0:
                                color = (255, 0, 0)
                                line = 6
                            else:
                                color = (0, 0, 255)
                                line = 2
                        else:
                            color = (0, 0, 255)
                            line = 2

                        cv2.rectangle(draw_image, (xbox_left, ytop_draw+ystart),
                                      (xbox_left+win_draw, ytop_draw+win_draw+ystart),
                                      color, line)

            return draw_image

'''
def train_classifier(pos_data_dir, pos_data_exts,
                     neg_data_dir, neg_data_exts,
                     data_dir, data_file='clf.p'):
    """ Train SVC

    Args:
        pos_data_dir: A path to a directory which contains positive images
        pos_data_exts: A list of extensions which positive images have
        neg_data_dir: A path to a directory which contains negative images
        neg_data_exts: A list of extensions which negative images have
        data_dir: A path to a directory which contains data files
        data_file: A name of the data file

    Returns:
        A dictionary of parameters
    """
    fnames = util.read_filename(pos_data_dir, pos_data_exts)
    posimgs = util.read_image(fnames)
    fnames = util.read_filename(neg_data_dir, neg_data_exts)
    negimgs = util.read_image(fnames)

    pos_features = fe.extract_features(posimgs)
    neg_features = fe.extract_features(negimgs)

    assert len(pos_features[0]) == len(neg_features[0])
    logger.debug('%s features extracted', len(pos_features[0]))
    x = np.vstack((pos_features, neg_features)).astype(np.float64)
    x_scaler = StandardScaler().fit(x)
    scaled_x = x_scaler.transform(x)

    y = np.hstack((np.ones(len(pos_features)), np.zeros(len(neg_features))))

    rand_state = np.random.randint(0, 100)
    x_train, x_test, y_train, y_test = train_test_split(
        scaled_x, y, test_size=0.2, random_state=rand_state)

    clf = LinearSVC()

    logger.debug('train SVC...')
    clf.fit(x_train, y_train)
    logger.debug('test accuracy of SVC = %.4lf', round(clf.score(x_test, y_test), 4))

    params = {}

    params['num_pos_samples'] = len(posimgs)
    params['pos_img_size'] = util.check_image_size(posimgs)
    params['pos_image_data_type'] = util.check_image_data_type(posimgs)
    params['pos_img_example'] = posimgs[0]

    params['num_neg_samples'] = len(negimgs)
    params['neg_img_size'] = util.check_image_size(negimgs)
    params['neg_image_data_type'] = util.check_image_data_type(negimgs)
    params['neg_img_example'] = negimgs[0]

    params['clf'] = clf
    params['scaler'] = x_scaler

    fname = data_dir + '/' + data_file
    with open(fname, mode='wb') as f:
        pickle.dump(params, f)
        logger.debug('classifier data is saved as %s', fname)

    return params
'''
def main():
    import setting

    opts, _ = getopt.getopt(sys.argv[1:], 'ft', [])
    force = False
    test = False
    for opt, _ in opts:
        if opt == '-f':
            force = True
        if opt == '-t':
            test = True

    ooo = ImageProcessor('../data/vehicles/',
                         '../data/non-vehicles/',
                         '../data/',
                         '../output_images')

    if force:
        ooo.train_classifier()
    else:
        ooo.process_test_imgs()

    if test:
        _fig = plt.figure()
        plt.subplot(121)
        plt.imshow(ooo.params['pos_img_example'])
        plt.title('Example Car Image')
        plt.subplot(122)
        plt.imshow(ooo.params['neg_img_example'])
        plt.title('Example Not-car Image')
        plt.savefig(ooo.out_img_dir + '/' + 'example_image.png', bbox_inches='tight')

    ooo.scales = np.linspace(1, 3.5, 10)
    ooo.ystarts = np.linspace(380, 350, 10, dtype=np.int)
    ooo.ystops = np.linspace(600, 800, 10, dtype=np.int)

    #import window_search as ws

    name = '../test_images/test1.jpg'
    #name = '../data/vehicles/GTI_Far/image0000.png'
    img = mpimg.imread(name)

    out = ooo.find_object(img)
    #out = ooo.draw_window(img)
    '''
    import window_search as ws
    out = ws.window_search(img, 400, 700, 1, ooo.params['clf'], ooo.params['scaler'],
                           pix_per_cell=8)
    '''
    fig = plt.figure()
    plt.imshow(out)
    plt.show()







    '''
    fig = plt.figure()
    plt.imshow(out)
    plt.show()
    '''
    '''
    def f(img):
        return ws.window_search(img, 400, 656, scale, params['clf'], params['scaler'],
                         pix_per_cell=8)
    '''
    '''
    clip = VideoFileClip('../test_video.mp4')
    output = '../output_images/test_video.mp4'
    logger.debug('start')
    clip.fl_image(ooo.find_object).write_videofile(output, audio=False)
    '''

if __name__ == '__main__':
    main()
