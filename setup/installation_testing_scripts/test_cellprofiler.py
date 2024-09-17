print('Importing packages and checking versions')
import numpy as np
print('Numpy version: {}'.format(np.__version__))
import cellprofiler
print('Cellprofiler version: {}'.format(cellprofiler.__version__))
from cellprofiler.__main__ import main
import centrosome
import centrosome.cpmorphology
import javabridge
print('Not printing versions for javabridge and centrosome')

print('No error messages above? Then we\'re fine.')