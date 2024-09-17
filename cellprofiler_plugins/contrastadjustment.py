"""
ContrastAdjustment
========

Adjusts contrast in input images by re-scaling input images based on percentiles

|
============ ============ ===============
Supports 2D? Supports 3D? Respects masks?
============ ============ ===============
YES          YES          YES
============ ============ ===============
"""

from skimage import exposure
import numpy as np

import cellprofiler_core.module
import cellprofiler_core.setting
from cellprofiler_core.setting.text import Float



class ContrastAdjustment(cellprofiler_core.module.ImageProcessing):
    module_name = "ContrastAdjustment"

    variable_revision_number = 1

    def create_settings(self):
        super(ContrastAdjustment, self).create_settings()

        
        self.lower_percentile = Float(
            doc="""
            lower percentile values to be set to zero. Choose higher values to increase
            brightness. The default value of 1.0 has worked well so far
            """,
            maxval=100.0,
            minval=0.0,
            text='lower percentile',
            value=1.0
        )
        self.upper_percentile = Float(
        doc="""
        upper percentile values to be set to image maximum. Choose lower values to increase brightness.
        The default value of 99 has worked well so far.
        """,
        maxval=100.0,
        minval=0.0,
        text='upper percentile',
        value=99.9
        )
        
        
    def settings(self):
        __settings__ = super(ContrastAdjustment, self).settings()
        __settings__ += [self.lower_percentile, self.upper_percentile]

        return __settings__

    def visible_settings(self):
        __settings__ = super(ContrastAdjustment, self).visible_settings()
        __settings__ += [self.lower_percentile, self.upper_percentile]
        
        return __settings__
        
    def make_adjustment(self, input_image, lower_percentile, upper_percentile):
        im = exposure.rescale_intensity(input_image)
        left, right = np.percentile(im, (self.lower_percentile.value, self.upper_percentile.value))
        adjusted_im = exposure.rescale_intensity(im, in_range=(left, right))
        return adjusted_im

    def run(self, workspace):
        self.function = self.make_adjustment

        super(ContrastAdjustment, self).run(workspace)