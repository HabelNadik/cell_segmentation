# coding=utf-8

### copy me to the plugins dir with: cp nucleus_mrcnn.py /mnt/c/Users/bhaladik/AppData/Roaming/plugins

"""
Nucleus MRCNN Cluster
==========

**Nucleus MRCNNr** identifies nuclei.

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

import os.path

import numpy as np

import numpy
import skimage
import tensorflow as tf
import scipy


import sys
from os.path import isdir, join, dirname, realpath

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

CONFIG_FILENAME = 'nucleus_mrcnn_cp_config.txt'
MODEL_DIR_KEY = 'model_code_dir'
NUCLEUS_DIR_KEY = 'nucleus_code_dir'
WEIGHTS_PATH_KEY = 'model_weights_path'
LOGS_DIR_KEY = 'model_logs_dir'
CFG_DELIMITER = '='

def load_config():
    current_folder = dirname(realpath(__file__))
    cfg_filepath = join(current_folder, CONFIG_FILENAME)
    cfg_file = open(cfg_filepath, 'r')
    lib_directories = []
    model_directories = []
    weights_path = ''
    for line in cfg_file.readlines():
        key, path = line.strip().split(CFG_DELIMITER)
        if 'logs_dir' in key:
            model_directories.append(path)
        elif 'dir' in key:
            lib_directories.append(path)
        else:
            weights_path = path
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

class Nucleus_MRCNN(ImageSegmentation):
    category = "Advanced"

    module_name = "Nucleus_MRCNN"

    variable_revision_number = 1


    def __init__(self, logs_dir=logs_dir, weights_path=weights_path):
        self.config = nucleus.NucleusInferenceConfig()
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
        super(Nucleus_MRCNN, self).__init__()

    def create_settings(self):
        super(Nucleus_MRCNN, self).create_settings()

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
        __settings__ = super(Nucleus_MRCNN, self).settings()

        return __settings__ + [
            self.mask_name,
            self.overlap_size,
            self.exclude_border_objects,
            self.do_norm
        ]

    def visible_settings(self):
        __settings__ = super(Nucleus_MRCNN, self).settings()

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
        use_image = self.preprocess_image(x_data, norm_to_model_mean=normalize)

        mask_data = None

        if not self.mask_name.is_blank:
            mask_name = self.mask_name.value

            mask = images.get_image(mask_name)

            mask_data = mask.pixel_data
        #print('--------- VIBE CHECK ----------')
        #print('-------- x : {} ------'.format(x_data.shape))
        #print('- use image: {} ------'.format(use_image.shape))
        segmentation = self.single_image_segmentation(use_image, quarter_overlap=overlap_size)

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


    ### CUSTOM FUNCTIONS FOR REQUIRED NORMALIZATION ############
    def normalize_im(self, in_im, out_max=255, out_min=0, norm_to_model_mean=True):
        if norm_to_model_mean:
            in_im = in_im - np.mean(in_im) + self.config.MEAN_PIXEL[0]
            im_min = np.amin(in_im)
            im_max = np.amax(in_im)
            out = (in_im - im_min) * (out_max/(im_max - im_min))
            #out = (out / np.std(out)) * self.config.STD_PIXEL[0]
        else:
            im_min = np.amin(in_im)
            im_max = np.amax(in_im)
            out = (in_im - im_min) * (out_max/(im_max - im_min))
        #print('Created normalized image, mapping intensities from max {} min {} to out max {} min {}'.format(im_max, im_min, np.amax(out), np.amin(out)))
        return out.astype(np.uint8)

    def preprocess_image(self, raw_image, norm_to_model_mean=False):
        norm_im = self.normalize_im(raw_image, norm_to_model_mean=norm_to_model_mean)
        if norm_im.ndim != 3:
                norm_im = skimage.color.gray2rgb(norm_im)
        if norm_im.shape[-1] == 4:
            norm_im = norm_im[..., :3]
        norm_im[:,:,1] = np.amax(norm_im[:,:,0]) - norm_im[:,:,0]
        return norm_im

    
    def single_image_segmentation(self, input_image, quarter_overlap=10, resolve_overlaps=True):
        print('Segmentation test print - non molded detection')
        print('Input image shape is {}'.format(input_image.shape))
        if (input_image.shape[0] == 512) & (input_image.shape[1] == 512):
            print('Image in res 512x512 found!')
            #self.model.config.BATCH_SIZE = 1
            #self.model.config.USE_MINI_MASK = True
            #self.model.config.IMAGE_MIN_SCALE = 2.0
            print('Set batch size to 1 and use mini mask and min scale to 2')
            print('making metas from single image')
            results = self.model.detect([input_image])
            mask_array = results[0]['masks']
            rois = results[0]['rois']
            out_im = np.zeros(input_image.shape[:2], dtype=np.uint16)
            for i in range(mask_array.shape[2]):
                roi = rois[i,:]
                out_im[roi[0]:roi[2],roi[1]:roi[3]] = (mask_array[roi[0]:roi[2],roi[1]:roi[3],i].astype(np.bool) * (i+1)) * (~(out_im[roi[0]:roi[2],roi[1]:roi[3]] > 0))
        else:
            quarters = self.quarter_image(input_image, overlap=quarter_overlap)
            results = []
            for i in range(len(quarters)):
                results += self.model.detect([quarters[i]])
            out_im = self.quartered_masks_from_detection_to_image(results, input_image.shape[:2], overlap=quarter_overlap)
        return out_im

    def quartered_masks_from_detection_to_image(self, results, image_shape, overlap=10, resolve_overlaps=True):
        quarter_borders = self.make_quarter_borders(image_shape, overlap)
        out_im = np.zeros(image_shape, dtype=np.int16)
        obj_index = 1
        num_overlaps = 0
        num_objects_without_overlap = 0
        for i in range(len(results)):
            yt,yb,xl,xr = quarter_borders[i]
            r = results[i]
            mask_array = r['masks']
            sizes = np.sum(mask_array, axis=(0,1))
            indices = np.argsort(sizes)
            for index in indices:
                mask = mask_array[:,:, index]
                ## exclude objects that touch the inner borders
                ## we only look at the top and left since we traverse the original image from top to bottom and left to right
                if yt >= overlap:
                    if np.sum(mask[0,:]) > 0:
                        continue
                if xl >= overlap:
                    if np.sum(mask[:,0] > 0):
                        continue
                # to have separated objects, take the remaining background pixels and add the object to them
                #out_im[yt:yb,xl:xr] += (~((out_im[yt:yb,xl:xr] > 0) & (mask > 0))).astype(np.int16) * (mask.astype(np.bool) * (obj_index)).astype(np.int16)
                if resolve_overlaps:
                    out_im, overlap_found = self.new_mask_to_objects(out_im, mask, obj_index, quarter_borders[i])
                    if overlap_found:
                        num_overlaps += 1
                    else:
                        num_objects_without_overlap += 1
                else:
                    out_im[yt:yb,xl:xr] += (~((out_im[yt:yb,xl:xr] > 0) & (mask > 0))).astype(np.int16) * (mask.astype(np.bool) * (obj_index)).astype(np.int16)
                obj_index += 1
        return out_im


    def new_mask_to_objects(self, seg_image, new_mask, object_index, borders, overlap_threshold=0.3):
        yt, yb, xl, xr = borders
        #seg_image[yt:yb,xl:xr] += (~((seg_image[yt:yb,xl:xr] > 0) & (new_mask > 0))).astype(np.int16) * (new_mask.astype(np.bool) * (object_index)).astype(np.int16)
        # let's test overriding objects - if two objects overlap, keep the bigger one
        old_seg_lbl = seg_image[yt:yb,xl:xr]
        old_seg = old_seg_lbl > 0
        new_seg = new_mask > 0
        found_overlap = False
        if np.sum(new_mask & old_seg) == 0: # no overlap, we can just add the object
            seg_image[yt:yb,xl:xr] += (~(old_seg & new_seg)).astype(np.int16) * (new_seg * (object_index)).astype(np.int16)
        else: # there is overlap and we need to do something about it
            ## let's try merging into a bigger object
            found_overlap = True
            overlap_labels = np.unique((new_mask & old_seg) * old_seg_lbl)
            if overlap_labels.shape[0] <= 2: ## there is only overlap with one object
                old_seg_ovl = old_seg_lbl == overlap_labels[-1]
                new_obj_area = np.sum(new_seg)
                ### do nothing if the new object is fully enclosed in the old object
                if np.sum(new_seg | old_seg_ovl) == np.sum(old_seg_ovl):
                    return seg_image, found_overlap
                else:
                    old_obj_area = np.sum(old_seg_ovl)
                    overlap_area = np.sum(new_seg & old_seg_ovl)
                    ### separate the objects roughly by the object border of the smaller object
                    ### if the overlap is smaller than the threshold 
                    if (overlap_area <= (old_obj_area * overlap_threshold)) or (overlap_area <= (new_obj_area * overlap_threshold)):
                        ## write the object which has the smaller fraction covered by overlap on top of the one which is less covered
                        ov_norm_new = overlap_area / new_obj_area
                        ov_norm_old = overlap_area / old_obj_area
                        if ov_norm_new < ov_norm_old:
                            ## remove the overlapping part of the old object and add the new object there
                            seg_image[yt:yb,xl:xr] = seg_image[yt:yb,xl:xr] * ~(new_seg & old_seg_ovl)
                            seg_image[yt:yb,xl:xr] += (~((seg_image[yt:yb,xl:xr] > 0) & new_seg)).astype(np.int16) * (new_seg * (object_index)).astype(np.int16)
                        else:
                            ## remove the overlapping part of the new object and add it
                            new_seg = new_seg & (~(new_seg & old_seg))
                            seg_image[yt:yb,xl:xr] += (~(old_seg & new_seg)).astype(np.int16) * (new_seg * (object_index)).astype(np.int16)
                    else:
                        ## merge the two objects if the overlap is bigger than the threshold
                        seg_image[yt:yb,xl:xr] += (~(old_seg & new_seg)).astype(np.int16) * (new_seg * (overlap_labels[-1])).astype(np.int16)
                    
        return seg_image, found_overlap



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

    def quarter_image(self, input_image, overlap=10):
        ### separates image into quarters, where each quarter reaches overlap many pixels into the other quarters
        ### ie actual overlap is overlap*2
        borders = self.make_quarter_borders(input_image.shape[:2], overlap)
        #print('quartering image of shape {} with the following borders'.format(input_image.shape))
        #print(borders)
        quarters = []
        for yt, yb, xl, xr in borders:
            quarters.append(input_image[yt:yb,xl:xr])
        return quarters


    def add_overlap(self, val1, val2, overlap, left_border):
        assert val1 < val2, 'incorrect order of values'
        if ((val1 - overlap) <= left_border):
            return val1, val2+overlap
        else:
            return val1-overlap, val2

    def make_quarter_borders(self, in_shape, overlap):
        y_border = in_shape[0] // 2
        x_border = in_shape[1] // 2
        assert y_border == x_border, 'unequal borders!!! images must be symmetric!!!'
        border = y_border
        im_border = in_shape[0]
        out_borders = []
        for i in range(0, im_border, border):
            for k in range(0, im_border, border):
                y_top, y_bot = self.add_overlap(i,i+border, overlap, 0)
                x_left, x_right = self.add_overlap(k,k+border, overlap, 0)
                out_borders.append([y_top,y_bot,x_left,x_right])
        return out_borders


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


