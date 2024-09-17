# Preparing training data for Mask-R-CNN training on nuclei

We aggregate data from four different sources for model training and a small set of annotated in-house data.
The public data comes from the following publications:

- Caicedo et al. Nucleus segmentation across imaging experiments: the 2018 Data Science Bowl. *Nature Methods* (2019)
    doi: https://doi.org/10.1038/s41592-019-0612-7
    Datset link: https://bbbc.broadinstitute.org/BBBC038/
- Korfhage et al. Detection and segmentation of morphologically complex eukaryotic cells in fluorescence microscopy images via feature pyramid fusion. *PLOS Computational Biology* (2020)
    doi: https://doi.org/10.1371/journal.pcbi.1008179
    Dataset link: https://box.uni-marburg.de/index.php/s/N934NJi7IsvOphf (this link is broken, but we can provide a copy of the data upon request)
- Gunesli et al. AttentionBoost: Learning what to attend for gland segmentation in histopathological images by boosting fully convolutional networks. *IEEE Transactions on Medical Imaging* (2020)
    doi: https://doi.org/10.1109/TMI.2020.3015198
    Dataset link: http://www.cs.bilkent.edu.tr/~gunduz/downloads/NucleusSegData/
- Kromp et al. An annotated fluorescence image dataset for training nuclear segmentation methods. *Scientific Data* (2020)
    doi: https://doi.org/10.1038/s41597-020-00608-w
    Dataset link: https://identifiers.org/biostudies:S-BSST265

We also have a small internal inhouse dataset. Both the inhouse dataset and the dataset from *Korfhage et al.* can be downloaded from the following link:

https://ccrivienna-my.sharepoint.com/:f:/g/personal/ben_haladik_ccri_at/Eg-Xc7pdbVlOqYrxjt3XqwMBdXk00QuS7NwEL8GSpPwTKA?e=QgcdCR

Within that folder, the inhouse dataset is in `inhouse_dataset`, the Korfhage dataset is in `synmikro_macrophages`.

## Data preparation instructions

The datasests all come in different formats. So  we process the datasets *Korfhage et al.* and *Gunesli et al.* to bring them into the format from *Kromp et al.*. The respective notebooks for this are `ConvertGunesliData.ipynb` and `ConvertKorfhageData.ipynb` Additionally, we need to convert the annotated in-house data from annotations done with VGG image annotator into masks, which is accomplished with the script `ConvertInhouseData.ipynb`. You will need to adjust the notebooks to reflect your local directory structures.

Finally, the script `CombineTrainingData.ipynb` is used to combine all training data and split into training and testing data.