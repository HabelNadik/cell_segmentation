# coding=utf-8

### copy me to the plugins dir with: cp nucleus_mrcnn.py /mnt/c/Users/bhaladik/AppData/Roaming/plugins

"""
Neuroblastoma MRCNN Cluster
==========

**Neuroblastoma MRCNN** identifies nuclei.

Instructions:

This module works with a custom setup of matterprot's Mask R CNN implementation
|

============ ============ ===============
Supports 2D? Supports 3D? Respects masks?
============ ============ ===============
YES          NO           YES
============ ============ ===============
"""
print("This is a test print to see how often the module is started")
import os
import os.path
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import numpy as np

import numpy
import skimage
import tensorflow as tf
import scipy


import sys
from os.path import isdir, isfile, join, dirname, realpath, abspath

import centrosome
#from centrosome import cmorphology

import cellprofiler_core.image
import cellprofiler_core.module
import cellprofiler_core.object
import cellprofiler_core.setting

from cellprofiler_core.module.image_segmentation import ImageSegmentation
from cellprofiler_core.setting.subscriber import ImageSubscriber
from cellprofiler_core.setting.text import Pathname, Integer
from cellprofiler_core.setting import Binary, Color

from cellprofiler.modules import _help

CONFIG_FILENAME = 'neuroblastoma_mrcnn_cp_config.txt'
MODEL_DIR_KEY = 'model_code_dir'
NUCLEUS_DIR_KEY = 'nucleus_code_dir'
WEIGHTS_PATH_KEY = 'model_weights_path'
LOGS_DIR_KEY = 'model_logs_dir'
CFG_DELIMITER = '='

def check_path_fix_and_return(in_path, current_folder, should_be_dir=True):
    """
    Check if path exists. If it's relative, try to see if it's relative to the working directory or this one.
    Return the path if it was found, None otherwise.
    """
    out_path = None
    if in_path.startswith('/'):
        if should_be_dir and isdir(in_path):
            out_path =  in_path
        elif isfile(in_path):
            out_path = in_path
        else:
            if should_be_dir and isdir('.' + in_path):
                out_path = '.' + in_path
            elif isfile('.' + in_path):
                out_path = '.' + in_path
    else:
        if should_be_dir:
            if isdir(in_path):
                out_path = in_path
            elif isdir(abspath(join(current_folder, in_path))):
                out_path = abspath(join(current_folder, in_path))
        else:
            if isfile(in_path):
                out_path = in_path
            elif isfile(abspath(join(current_folder, in_path))):
                out_path = abspath(join(current_folder, in_path))
    return out_path


def load_config():
    """
    Load config that contains relative paths
    
    """
    current_folder = dirname(realpath(__file__))
    cfg_filepath = join(current_folder, CONFIG_FILENAME)
    cfg_file = open(cfg_filepath, 'r')
    lib_directories = []
    model_directories = []
    weights_path = ''
    for line in cfg_file.readlines():
        key, path = line.strip().split(CFG_DELIMITER)
        if 'logs_dir' in key:
            model_directories.append(check_path_fix_and_return(path, current_folder, should_be_dir=True))
        elif 'dir' in key:
            lib_directories.append(check_path_fix_and_return(path, current_folder, should_be_dir=True))
        else:
            weights_path = abspath(check_path_fix_and_return(path, current_folder, should_be_dir=False))
    return lib_directories, model_directories, weights_path

lib_dirs, model_dirs, weights_path = load_config()
for use_dir in lib_dirs:
    sys.path.insert(0, use_dir)
logs_dir = model_dirs[0]

from mrcnn import utils
from mrcnn import visualize
from mrcnn.visualize import display_images
import mrcnn.model as modellib

from mrcnn.model import log
import nucleus

### functions for segmentation
from segmentation_functions import segment_single_image, preprocess_image



class Neuroblastoma_MRCNN(ImageSegmentation):
    category = "Advanced"

    module_name = "Neuroblastoma_MRCNN"

    variable_revision_number = 1


    def __init__(self, logs_dir=logs_dir, weights_path=weights_path):
        self.config = nucleus.NucleusInferenceConfig()
        self.config.RPN_NMS_THRESHOLD = 1.
        self.config.DETECTION_MAX_INSTANCES = 800
        print('Set nucleus inference config RPN_NMS_THRESHOLD to {}'.format(self.config.RPN_NMS_THRESHOLD))
        self.config.USE_MINI_MASK = False
        self.config.IMAGE_MIN_SCALE = 1.
        #self.config.BATCH_SIZE = 4
        self.config.BATCH_SIZE = 1
        #self.config.IMAGES_PER_GPU = 4
        self.config.IMAGES_PER_GPU = 1
        DEVICE = "/cpu:0"

        ## should be changed at some point
        with tf.device(DEVICE):
            self.model = modellib.MaskRCNN(mode="inference",
                                    model_dir=logs_dir,
                                    config=self.config)
        #self.default_path = join(logs_dir, 'mask_rcnn_nucleus_0030.h5')
        #self.weights_pathname = join(logs_dir, 'mask_rcnn_nucleus_0030.h5')
        self.weights_pathname = weights_path
        self.model.load_weights(self.weights_pathname, by_name=True)
        #print("Successfully loaded model")
        super(Neuroblastoma_MRCNN, self).__init__()

    def create_settings(self):
        super(Neuroblastoma_MRCNN, self).create_settings()

        self.x_name.text = "Select the input image"
        self.x_name.doc = "Select the image from which you want to identify objects"
        self.y_name.text = "Select the primary objects to be identified (I hope it's nuclei)"
        self.y_name.doc = "Enter the name for your objects"

        self.mask_name = ImageSubscriber(
            "Mask",
            can_be_blank=True,
            doc=""
        )

        self.overlap_size = Integer(
            "Size of the overlap region",
            value=15,
            minval=10,
            maxval=300,
            doc="""\
                Size of the overlap between the quarters on which the
                neural network is called. Should be larger than the
                largest expected object."""
        )

        self.do_norm = Binary(
            "Normalize images to training data parameters?",
            False,
            doc="""\
                Select "*{YES}*" to normalize input images to the
mean and standard deviation of the training data. Select "*{NO}*" to
only scale images to the full intensity range instead."""
        )

        self.exclude_border_objects = Binary(
            "Discard objects touching the border of the image?",
            True,
            doc="""\
Choose "*{YES}*" to discard objects that touch the border of the image.
Choose "*{NO}*" to ignore this criterion.
Objects discarded because they touch the border are outlined in yellow in the
module’s display. Note that if a per-object thresholding method is used
or if the image has been previously cropped or masked, objects that
touch the border of the cropped or masked region may also discarded.
|image0| Removing objects that touch the image border is useful when
you do not want to make downstream measurements of objects that are not
fully within the field of view. For example, measuring the area of a
partial object would not be accurate.
.. |image0| image:: {PROTIP_RECOMMEND_ICON}
            """.format(
                **{
                    "YES": "Yes",
                    "NO": "No",
                    "PROTIP_RECOMMEND_ICON": _help.PROTIP_RECOMMEND_ICON,
                }
            ),
        )

    def settings(self):
        __settings__ = super(Neuroblastoma_MRCNN, self).settings()

        return __settings__ + [
            self.mask_name,
            self.overlap_size,
            self.exclude_border_objects,
            self.do_norm
        ]

    def visible_settings(self):
        __settings__ = super(Neuroblastoma_MRCNN, self).settings()

        __settings__ = __settings__ + [
            self.mask_name,
            self.overlap_size,
            self.exclude_border_objects,
            self.do_norm
        ]

        return __settings__

    def run(self, workspace):

        workspace.display_data.statistics = []

        overlap_size = self.overlap_size.value
        normalize = self.do_norm.value

        x_name = self.x_name.value
        y_name = self.y_name.value

        images = workspace.image_set

        x = images.get_image(x_name)

        dimensions = x.dimensions

        x_data = x.pixel_data

        ### returns image as 3 channel fake color image with or without normalization
        use_image = preprocess_image(x_data, norm_to_model_mean=normalize, config=self.config)

        mask_data = None

        if not self.mask_name.is_blank:
            mask_name = self.mask_name.value

            mask = images.get_image(mask_name)

            mask_data = mask.pixel_data
        #print('--------- VIBE CHECK ----------')
        #print('-------- x : {} ------'.format(x_data.shape))
        #print('- use image: {} ------'.format(use_image.shape))
        segmentation = segment_single_image(use_image, self.model, quarter_overlap=overlap_size, resolve_overlaps=True)

        #segmentation = self.resize(segmentation, x_data.shape)
        objects = cellprofiler_core.object.Objects()
        objects.segmented = segmentation
        objects.parent_image = x

        border_excluded_labeled_image = segmentation.copy()
        object_count = np.amax(segmentation)
        labeled_image = self.filter_on_border(x, segmentation)
        border_excluded_labeled_image[segmentation > 0] = 0
        labeled_image, object_count = centrosome.cpmorphology.relabel(labeled_image)
        
        if self.show_window:
            statistics = workspace.display_data.statistics
            statistics.append(["# of accepted objects", "%d" % object_count])
            if object_count > 0:
                areas = scipy.ndimage.sum(
                    numpy.ones(segmentation.shape),
                    segmentation,
                    numpy.arange(1, object_count + 1),
                )
                areas.sort()
                low_diameter = (
                    np.sqrt(float(areas[object_count // 10]) / numpy.pi) * 2
                )
                median_diameter = (
                    np.sqrt(float(areas[object_count // 2]) / numpy.pi) * 2
                )
                high_diameter = (
                    np.sqrt(float(areas[object_count * 9 // 10]) / numpy.pi) * 2
                )
                statistics.append(
                    ["10th pctile diameter", "%.1f pixels" % low_diameter]
                )
                statistics.append(["Median diameter", "%.1f pixels" % median_diameter])
                statistics.append(
                    ["90th pctile diameter", "%.1f pixels" % high_diameter]
                )
                object_area = numpy.sum(areas)
                total_area = numpy.product(segmentation.shape[:2])
                statistics.append(
                    [
                        "Area covered by objects",
                        "%.1f %%" % (100.0 * float(object_area) / float(total_area)),
                    ]
                )
            workspace.display_data.x_data = x.pixel_data
            workspace.display_data.y_data = segmentation
            workspace.display_data.image = x.pixel_data
            workspace.display_data.labeled_image = segmentation
            workspace.display_data.border_excluded_labels = (
                border_excluded_labeled_image
            )
            workspace.display_data.dimensions = x.dimensions

        objname = self.y_name.value
        measurements = workspace.measurements

        # Add label matrices to the object set
        objects = cellprofiler_core.object.Objects()
        objects.segmented = labeled_image
        objects.unedited_segmented = segmentation
        objects.small_removed_segmented = labeled_image.copy()
        objects.parent_image = x

        workspace.object_set.add_objects(objects, self.y_name.value)

        self.add_measurements(workspace)



    def remove_padding(self, image):
        top = self.padding[0][0]
        bot = image.shape[0] - self.padding[0][1]
        left = self.padding[1][0]
        right = image.shape[1] - self.padding[1][1]
        out = image[top:bot,left:right]
        if len(image.shape) > 2:
            front = self.padding[2][0]
            back = image.shape[2] - self.padding[2][1]
            out = image[:,:,front:back]
        return out


    def filter_on_border(self, image, labeled_image):
        """Filter out objects touching the border
        In addition, if the image has a mask, filter out objects
        touching the border of the mask.

        ## copied from cellprofiler IdentifyPrimaryObjects
        """
        if self.exclude_border_objects.value:
            border_labels = list(labeled_image[0, :])
            border_labels.extend(labeled_image[:, 0])
            border_labels.extend(labeled_image[labeled_image.shape[0] - 1, :])
            border_labels.extend(labeled_image[:, labeled_image.shape[1] - 1])
            border_labels = numpy.array(border_labels)
            #
            # the following histogram has a value > 0 for any object
            # with a border pixel
            #
            histogram = scipy.sparse.coo_matrix(
                (
                    numpy.ones(border_labels.shape),
                    (border_labels, numpy.zeros(border_labels.shape)),
                ),
                shape=(numpy.max(labeled_image) + 1, 1),
            ).todense()
            histogram = numpy.array(histogram).flatten()
            if any(histogram[1:] > 0):
                histogram_image = histogram[labeled_image]
                labeled_image[histogram_image > 0] = 0
            elif image.has_mask:
                # The assumption here is that, if nothing touches the border,
                # the mask is a large, elliptical mask that tells you where the
                # well is. That's the way the old Matlab code works and it's duplicated here
                #
                # The operation below gets the mask pixels that are on the border of the mask
                # The erosion turns all pixels touching an edge to zero. The not of this
                # is the border + formerly masked-out pixels.
                mask_border = numpy.logical_not(
                    scipy.ndimage.binary_erosion(image.mask)
                )
                mask_border = numpy.logical_and(mask_border, image.mask)
                border_labels = labeled_image[mask_border]
                border_labels = border_labels.flatten()
                histogram = scipy.sparse.coo_matrix(
                    (
                        numpy.ones(border_labels.shape),
                        (border_labels, numpy.zeros(border_labels.shape)),
                    ),
                    shape=(numpy.max(labeled_image) + 1, 1),
                ).todense()
                histogram = numpy.array(histogram).flatten()
                if any(histogram[1:] > 0):
                    histogram_image = histogram[labeled_image]
                    labeled_image[histogram_image > 0] = 0
        return labeled_image


