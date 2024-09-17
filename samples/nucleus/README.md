# Nuclei Counting and Segmentation

This sample implements segmentation of individual nuclei in microscopy images.
The `nucleus.py` file contains the coe for training. `test_nucleus_model_on_local_data.ipynb` is for visualizing segmentations for testing in the minimal example setup, or with your own trained model. 


## Command line Usage
Train a new model starting from ImageNet weights using `train` dataset (which is `stage1_train` minus validation set)
```
python3 nucleus.py train --dataset=/path/to/dataset --subset=train --weights=imagenet
```

Train a new model starting from specific weights file using the full `stage1_train` dataset
```
python3 nucleus.py train --dataset=/path/to/dataset --subset=stage1_train --weights=/path/to/weights.h5
```

Resume training a model that you had trained earlier
```
python3 nucleus.py train --dataset=/path/to/dataset --subset=train --weights=last
```
