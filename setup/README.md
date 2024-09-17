# Setting up Mask-R-CNN for segmentation with CellProfiler

In general, we recommend installing the required packages with pip or mamba and require a pip version 24.2. Setting up the environment with conda has been very slow in our experience.

## Minimal working example

If you only want to check the model segmentations using a pretrained model and the provided notebooks in `samples/nucleus`, you do not have to go through the slightly more lengthy process described below, but can use conda or python to install the packages in `minimal_example_requirements.txt`:

When using conda, do:
    
    conda env create -n minimal_segmentation --file minimal_env.yml
    conda activate minimal_segmentation
    python -m pip install tensorflow==2.4.1

When using pip, :
    pip install -r minimal_example_requirements.txt

Pip/Anaconda may throw some error messages for this installation, but things should still work.

If you also want to use our pre-trained model for testing segmentations, you need to download the weights, since they are too large for the github repository. You can download them from here: https://ccrivienna-my.sharepoint.com/:f:/g/personal/ben_haladik_ccri_at/Eg-Xc7pdbVlOqYrxjt3XqwMBdXk00QuS7NwEL8GSpPwTKA?e=QgcdCR
The file is called: `mask_rcnn_nucleus_0090.h5` and needs to be copied to `example_data` to make the corresponding notebook work.

This minimal setup should work on both linux and windows machines.

## Installation
Full installation has only been tested on linux machines.
Use the script `install_packages_for_cellprofiler.sh` to install the required packages. Please make sure to be in this folder when running this script. It calls some scripts to test whether the installation worked via relative paths and these will not work if it was called from somewhere else. The script also calls the `install_packages_for_segmentation.sh`, which installs the packages for the segmentation model.
The installation itself will output quite a few warnings, but the scripts are not supposed to throw any errors. Please also check the additional software requirements at the bottom of this readme.

## Setting the model up for segmentation with CellProfiler
The folder `cellprofiler_plugins` contains the plugins that are required to run the model within CellProfiler. If you only want to use the plugin here as described and do not have your own plugins directory, it should be sufficient to just call cellprofiler with the `--plugins-directory` flag and have it point to the `cellprofiler_plugins` folder in this directory. If you already have your own plugins directory, the contents of that folder need to be copied to your local cellprofiler plugins directory to enable running the pipeline, and you will have to adjust the paths in the respective config files. Specifically, the following paths need to be adjusted to your local environment by replacing the string `path_to_this_repository` with the path to this repository:

    model_code_dir=path_to_this_repository/mrcnn
    nucleus_code_dir=path_to_this_repository/samples/nucleus
    nuclei_top_dir=path_to_this_repository/
    model_weights_path=path_to_this_repository/example_data/mask_rcnn_nucleus_0090.h5
    model_logs_dir=path_to_this_repository/logs/


## Requirements
Installation requires the following software to be installed:
* gcc/10.2.0
* Java/13.0.2
* GTK+/3.24.23-GCCcore-10.2.0
* Python/3.8.2-GCCcore-9.3.0
* pybind11/2.4.3-GCCcore-9.3.0-Python-3.8.2
* OpenBLAS/0.3.12-GCC-9.3.0
* lapack/gcc/64/3.9.0

## Notes
We are aware that tensorflow versions that are later than version 2.4 will break the model implementation.