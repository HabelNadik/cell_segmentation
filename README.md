# Mask R-CNN for Object Detection and Segmentation

This is an implementation of [Mask R-CNN](https://arxiv.org/abs/1703.06870) on Python 3, Keras, and TensorFlow. It is adapted from  the implementation at https://github.com/matterport/Mask_RCNN to be compatible with tensorflow 2 and to enable integration with CellProfiler 4.2.1. The model generates bounding boxes and segmentation masks for each instance of an object in the image. It's based on Feature Pyramid Network (FPN) and a ResNet101 backbone. Here, we focus on segmentation of nuclei only.


The repository includes:
* Source code of Mask R-CNN built on FPN and ResNet101 as originally implemented [here](https://github.com/matterport/Mask_RCNN)
* Code to prepare training data
* Code to train the model
* Code and instructions to use the model as a CellProfiler plugin


## Setup and installation
Consult the folder `setup` for instructions on installation and integration with CellProfiler.

## Testing model segmentations

Model segmentations can be tested and visualized using the jupyter notebooks in `samples/nucleus`. To run the example, you need to download the model and have the correct packages installed. You can find instructions for the minimal installation in `setup/README.md` to install the correct packages. You can download the model from: https://ccrivienna-my.sharepoint.com/:f:/g/personal/ben_haladik_ccri_at/Eg-Xc7pdbVlOqYrxjt3XqwMBdXk00QuS7NwEL8GSpPwTKA?e=QgcdCR and add it to the folder `example_data`. When you have the correct environment and downloaded the model and added it to `example_data`, you can run the example.

## Integration with CellProfiler
This repository contains code that enables model usage as a cellprofiler plugin in the `cellprofiler_plugins` folder. More details for using CellProfiler with plugins can be found [here](https://plugins.cellprofiler.org/using_plugins.html). If you have installed CellProfiler from souce using the installation script in `setup`, it should be sufficient to fill the config files with your local paths, copy the python files and config files into a plugin directory and tell CellProfiler to use it, by using the `--plugins-directory` flag when running cellprofiler in the console.

## Training the model yourself

If you want to train the model yourself, follow the steps below. We do not recommend training the model on a CPU, since it will take very long.

### Preparing training data

The folder `prepare_training_data` contains code and instructions to prepare training data for the model. In brief, you will need to download the training data from the four indicated sources and then run the jupyter notebooks located in the folder.

### Training the model

The script to train the model is `samples/nucleus/nucleus.py`.
If you have generated the training data as described in `prepare_training_data`, you should be able to start training with the following command from this folder:

    python ./samples/nucleus/nucleus.py train --dataset=./training_data --subset=stage1_train --weights=last


## Requirements
The code in this repository was tested under python 3.8.2 with tensorflow 2.4.1 and keras 2.4.3 on a linux machine. Check the installation scripts and README in the `setup` folder for details.