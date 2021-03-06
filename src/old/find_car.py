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
from scipy.ndimage.measurements import label
from moviepy.editor import VideoFileClip
import feature_extractor as fe
import bounding_box as bb
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
        self.params['cspace_color'] = 'LUV'
        self.params['spatial_size'] = (32, 32)
        self.params['hist_bins'] = 32
        self.params['hist_range'] = (0, 256)
        self.params['cspace_hog'] = 'YCrCb'
        self.params['orient'] = 9
        self.params['pix_per_cell'] = 8
        self.params['cell_per_block'] = 2
        self.params['hog_channel'] = 'ALL'

        self.p_heat = None
        self.idx = 0

    def reset(self):
        """ Reset image processor
        """
        self.idx = 0
        self.p_heat = None

    def train_classifier(self):
        """ Train SVC
        """
        fnames = util.read_filename(self.pos_data_dir, self.pos_data_exts)
        posimgs = util.read_image(fnames)
        fnames = util.read_filename(self.neg_data_dir, self.neg_data_exts)
        negimgs = util.read_image(fnames)

        pos_features = fe.extract_features(
            posimgs,
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
            negimgs,
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

        clf = LinearSVC(C=0.01)

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
        """ Search car on test images
        """
        clf_param_file = self.data_dir + '/' + self.data_file
        if os.path.exists(clf_param_file):
            with open(clf_param_file, mode='rb') as f:
                self.params = pickle.load(f)
                logger.debug('classifier parameters are read from a file')
        else:
            self.params = self.train_classifier()

        fnames = glob.glob('../test_images/*.jpg')
        for name in fnames:
            img = mpimg.imread(name)
            self.reset()
            out = self.find_object(img)
            ftitle, fext = os.path.splitext(os.path.basename(name))

            path = self.out_img_dir + '/' + ftitle + '_detected'+ fext
            mpimg.imsave(path, out)

    def draw_sample(self):
        """ Draw sample images of feature extraction
        """
        clf_param_file = self.data_dir + '/' + self.data_file
        if os.path.exists(clf_param_file):
            with open(clf_param_file, mode='rb') as f:
                self.params = pickle.load(f)
                logger.debug('classifier parameters are read from a file')
        else:
            self.params = self.train_classifier()

        fig = plt.figure(figsize=(14, 6))
        idx = 1
        img = self.params['pos_img_example']
        for img in range(2):
            if img == 0:
                img = self.params['pos_img_example']
                fig.add_subplot(2, 7, idx)
                plt.imshow(img)
                plt.title('Car')
            else:
                img = self.params['neg_img_example']
                fig.add_subplot(2, 7, idx)
                plt.imshow(img)
                plt.title('Not car')
            idx += 1

            cimg = fe.convert_color_space(img, self.params['cspace_color'])
            for channel in range(3):
                fig.add_subplot(2, 7, idx)
                plt.imshow(cimg[:, :, channel], cmap='gray')
                plt.title('Ch-{0}'.format(channel+1))
                idx += 1

            himg = fe.convert_color_space(img, self.params['cspace_hog'])
            for channel in range(3):
                _, hogimg = fe.get_hog_features(
                    himg[:, :, channel],
                    orient=self.params['orient'],
                    pix_per_cell=self.params['pix_per_cell'],
                    cell_per_block=self.params['cell_per_block'],
                    vis=True, feature_vec=False)

                fig.add_subplot(2, 7, idx)
                plt.imshow(hogimg, cmap='gray')
                plt.title('Hog Ch-{0}'.format(channel+1))
                idx += 1
        plt.subplots_adjust(left=0.02, bottom=0.02, right=0.98, top=0.98,
                            wspace=0.5, hspace=0.)
        plt.savefig(self.out_img_dir + '/' + 'converted_image.png', bbox_inches='tight')

    def draw_window(self, image=None):
        """ Draw search area on a image

        Args:
            image: A image to be process
        """
        clf_param_file = self.data_dir + '/' + self.data_file
        if os.path.exists(clf_param_file):
            with open(clf_param_file, mode='rb') as f:
                self.params = pickle.load(f)
                logger.debug('classifier parameters are read from a file')
        else:
            self.params = self.train_classifier()

        tmp_scales = self.scales
        tmp_ystarts = self.ystarts
        tmp_ystops = self.ystops

        if not image:
            image = mpimg.imread('../test_images/test1.jpg')

        i = 0
        for scale, ystart, ystop in zip(tmp_scales, tmp_ystarts, tmp_ystops):
            self.scales = [scale]
            self.ystarts = [ystart]
            self.ystops = [ystop]

            out = self.find_object(image, test=True)
            path = self.out_img_dir + '/' + 'window_' + str(i) + '.png'
            mpimg.imsave(path, out)
            i += 1

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
        test_image = np.copy(image)
        # convert jpeg value to png value
        image = image.astype(np.float32)/255
        bboxes = []
        for scale, ystart, ystop in zip(self.scales, self.ystarts, self.ystops):
            img_tosearch = image[ystart:ystop, :, :]
            if scale != 1:
                imshape = img_tosearch.shape
                img_tosearch = cv2.resize(
                    img_tosearch,
                    (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))

            nxblocks = (img_tosearch.shape[1] // pix_per_cell) - 1
            nyblocks = (img_tosearch.shape[0] // pix_per_cell) - 1

            nblocks_per_window = (window // pix_per_cell) - 1
            cells_per_step = 2
            pics_per_step = pix_per_cell * cells_per_step
            nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
            nysteps = (nyblocks - nblocks_per_window) // cells_per_step

            for xb in range(nxsteps+1):
                for yb in range(nysteps):
                    if xb == (nxsteps+1):
                        xleft = img_tosearch.shape[1] - nblocks_per_window
                    else:
                        xleft = xb*pics_per_step
                    ytop = yb*pics_per_step

                    subimg = cv2.resize(
                        img_tosearch[ytop:ytop+window, xleft:xleft+window],
                        (window, window))

                    feature = fe.extract_features(
                        [subimg],
                        use_spatial=use_spatial,
                        use_hist=use_hist,
                        use_hog=use_hog,
                        cspace_color=cspace_color,
                        spatial_size=spatial_size,
                        hist_bins=hist_bins,
                        hist_range=hist_range,
                        cspace_hog=cspace_hog,
                        orient=orient,
                        pix_per_cell=pix_per_cell,
                        cell_per_block=cell_per_block,
                        hog_channel=hog_channel)

                    features_scaled = scaler.transform(feature[0].reshape(1, -1))
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

                        if test:
                            cv2.rectangle(test_image, (xbox_left, ytop_draw+ystart),
                                          (xbox_left+win_draw,
                                           ytop_draw+win_draw+ystart),
                                          color, line)

                        bboxes.append(((xbox_left, ytop_draw+ystart),
                                       (xbox_left+win_draw, ytop_draw+win_draw+ystart)))

        heat = np.zeros_like(image[:, :, 0]).astype(np.float)
        heat = bb.add_heat(heat, bboxes)
        if self.p_heat is not None:
            heat = heat * 0.3 + self.p_heat * 0.7
        heat = bb.apply_threshold(heat, 8)
        heatmap = np.clip(heat, 0, 255)
        labels = label(heatmap)
        draw_image = bb.draw_labeled_bboxes(draw_image, labels)

        self.p_heat = heat

        if True or self.idx == 400 or self.idx == 500 or self.idx == 600 or self.idx == 700:
            path = self.out_img_dir + '/' + 'flame_' + str(self.idx) + '_heat.png'
            mpimg.imsave(path, heat, cmap='hot')
            path = self.out_img_dir + '/' + 'flame_' + str(self.idx) + '.png'
            mpimg.imsave(path, draw_image)

        self.idx += 1

        if test:
            return test_image
        else:
            return draw_image


def main():
    import setting

    opts, _ = getopt.getopt(sys.argv[1:], 'ftc', [])
    force = False
    test = False
    check = False
    for opt, _ in opts:
        if opt == '-f':
            force = True
        if opt == '-t':
            test = True
        if opt == '-c':
            check = True

    ooo = ImageProcessor('../data/vehicles/',
                         '../data/non-vehicles/',
                         '../data/',
                         '../output_images/')
    ooo.scales = np.linspace(1.5, 3.5, 6)
    ooo.ystarts = np.linspace(400, 350, 6, dtype=np.int)
    ooo.ystops = np.linspace(656, 800, 6, dtype=np.int)

    if force:
        ooo.train_classifier()
    else:
        ooo.draw_sample()

    if test:
        logger.debug('plot windows')
        ooo.draw_window()

        _fig = plt.figure()
        plt.subplot(121)
        plt.imshow(ooo.params['pos_img_example'])
        plt.title('Example Car Image')
        plt.subplot(122)
        plt.imshow(ooo.params['neg_img_example'])
        plt.title('Example Not-car Image')
        plt.savefig(ooo.out_img_dir + '/' + 'example_image.png', bbox_inches='tight')

        logger.debug('process test images')
        ooo.process_test_imgs()

    if check:
        name = '../tmp/1.201591.png'
        img = mpimg.imread(name)
        img = img[:, :, 0:3]
        print(img.shape)
        #img = img.astype(np.float32)/255
        features = fe.extract_features(
            [img],
            use_spatial=ooo.params['use_spatial'],
            use_hist=ooo.params['use_hist'],
            use_hog=ooo.params['use_hog'],
            cspace_color=ooo.params['cspace_color'],
            spatial_size=ooo.params['spatial_size'],
            hist_bins=ooo.params['hist_bins'],
            hist_range=ooo.params['hist_range'],
            cspace_hog=ooo.params['cspace_hog'],
            orient=ooo.params['orient'],
            pix_per_cell=ooo.params['pix_per_cell'],
            cell_per_block=ooo.params['cell_per_block'],
            hog_channel=ooo.params['hog_channel'])
        clf = ooo.params['clf']
        scaler = ooo.params['scaler']
        features_scaled = scaler.transform(features)
        pred = clf.predict(features_scaled)
        print(pred)

    if not test and not check:
        ooo.reset()
        #clip = VideoFileClip('../test_video.mp4')
        #output = '../output_images/test_video.mp4'
        clip = VideoFileClip('../project_video.mp4')
        output = '../output_images/project_video.mp4'
        clip.fl_image(ooo.find_object).write_videofile(output, audio=False)

if __name__ == '__main__':
    main()
