import numpy as np

import skimage

import tensorflow as tf

def new_mask_to_objects(seg_image, new_mask, object_index, borders, overlap_threshold=0.3):
    yt, yb, xl, xr = borders
    #seg_image[yt:yb,xl:xr] += (~((seg_image[yt:yb,xl:xr] > 0) & (new_mask > 0))).astype(np.int16) * (new_mask.astype(np.bool) * (object_index)).astype(np.int16)
    # let's test overriding objects - if two objects overlap, keep the bigger one
    old_seg_lbl = seg_image[yt:yb,xl:xr]
    old_seg = old_seg_lbl > 0
    new_seg = new_mask > 0
    found_overlap = False
    ## sometimes we have artifacts that are one or two pixel wide lines, we can ignore them
    if (np.amax(np.sum(new_seg, axis=0)) <= 2) or (np.amax(np.sum(new_seg, axis=1)) <= 2):
        return seg_image, found_overlap
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


def quartered_masks_from_detection_to_image(results, image_shape, overlap=10, resolve_overlaps=True):
    quarter_borders = make_quarter_borders(image_shape, overlap)
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
                out_im, overlap_found = new_mask_to_objects(out_im, mask, obj_index, quarter_borders[i])
                if overlap_found:
                    num_overlaps += 1
                else:
                    num_objects_without_overlap += 1
            else:
                out_im[yt:yb,xl:xr] += (~((out_im[yt:yb,xl:xr] > 0) & (mask > 0))).astype(np.int16) * (mask.astype(np.bool) * (obj_index)).astype(np.int16)
            obj_index += 1
    return out_im

def add_overlap(val1, val2, overlap, left_border):
    assert val1 < val2, 'incorrect order of values'
    if ((val1 - overlap) <= left_border):
        return val1, val2+overlap
    else:
        return val1-overlap, val2

def make_quarter_borders(in_shape, overlap):
    y_border = in_shape[0] // 2
    x_border = in_shape[1] // 2
    assert y_border == x_border, 'unequal borders!!! images must be symmetric!!!'
    border = y_border
    im_border = in_shape[0]
    out_borders = []
    for i in range(0, im_border, border):
        for k in range(0, im_border, border):
            y_top, y_bot = add_overlap(i,i+border, overlap, 0)
            x_left, x_right = add_overlap(k,k+border, overlap, 0)
            out_borders.append([y_top,y_bot,x_left,x_right])
    return out_borders

def quarter_image(input_image, overlap=10):
    ### separates image into quarters, where each quarter reaches overlap many pixels into the other quarters
    ### ie actual overlap is overlap*2
    borders = make_quarter_borders(input_image.shape[:2], overlap)
    #print('quartering image of shape {} with the following borders'.format(input_image.shape))
    #print(borders)
    quarters = []
    for yt, yb, xl, xr in borders:
        quarters.append(input_image[yt:yb,xl:xr])
    return quarters

def clean_up_labels(segmented_image):
    """
    Make sure that segmented image labels are consecutive and start from 1
    """
    out = np.zeros(segmented_image.shape, dtype=np.uint16)
    unique_labels = np.unique(segmented_image)
    for i, lbl in enumerate(unique_labels):
        if lbl > 0:
            out[segmented_image == lbl] = i
    return out

def segment_single_image(input_image, model, quarter_overlap=10, resolve_overlaps=True):
    print('Segmentation test print - non molded detection')
    print('Input image shape is {}'.format(input_image.shape))
    if (input_image.shape[0] == 512) & (input_image.shape[1] == 512):
        print('Image in res 512x512 found!')
        #self.model.config.BATCH_SIZE = 1
        #self.model.config.USE_MINI_MASK = True
        #self.model.config.IMAGE_MIN_SCALE = 2.0
        print('Set batch size to 1 and use mini mask and min scale to 2')
        print('making metas from single image')
        results = model.detect([input_image])
        mask_array = results[0]['masks']
        rois = results[0]['rois']
        out_im = np.zeros(input_image.shape[:2], dtype=np.uint16)
        for i in range(mask_array.shape[2]):
            roi = rois[i,:]
            out_im[roi[0]:roi[2],roi[1]:roi[3]] = (mask_array[roi[0]:roi[2],roi[1]:roi[3],i].astype(np.bool) * (i+1)) * (~(out_im[roi[0]:roi[2],roi[1]:roi[3]] > 0))
    else:
        quarters = quarter_image(input_image, overlap=quarter_overlap)
        results = []
        for i in range(len(quarters)):
            results += model.detect([quarters[i]])
        out_im = quartered_masks_from_detection_to_image(results, input_image.shape[:2], overlap=quarter_overlap)
    out_im = clean_up_labels(out_im)
    return out_im


### CUSTOM FUNCTIONS FOR REQUIRED NORMALIZATION ############
def normalize_im(in_im, out_max=255, out_min=0, norm_to_model_mean=True, config=None, mean_pixel_val=None):
    if norm_to_model_mean and (not (config is None)) or (not (mean_pixel_val is None)):
        if config is None:
            in_im = in_im - np.mean(in_im) + mean_pixel_val
        else:    
            in_im = in_im - np.mean(in_im) + config.MEAN_PIXEL[0]
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

def preprocess_image(raw_image, norm_to_model_mean=False, config=None):
    norm_im = normalize_im(raw_image, norm_to_model_mean=norm_to_model_mean, config=config)
    if norm_im.ndim != 3:
        norm_im = skimage.color.gray2rgb(norm_im)
    if norm_im.shape[-1] == 4:
        norm_im = norm_im[..., :3]
    norm_im[:,:,1] = np.amax(norm_im[:,:,0]) - norm_im[:,:,0]
    return norm_im