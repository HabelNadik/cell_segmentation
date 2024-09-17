#!/bin/bash

module load Python/3.8.2-GCCcore-9.3.0
export PATH="$HOME/.local/bin:$PATH"

python -m pip uninstall tensorflow -y
python -m pip uninstall Keras -y
python -m pip uninstall numpy -y
python -m pip install Keras==2.4.3
python -m pip install tensorflow==2.4.1
python -m pip uninstall scikit-image -y
python -m pip install scikit-image==0.18.1
python -m pip uninstall numpy -y
python -m pip install numpy==1.20.1
python -m pip install opencv-python==4.5.3