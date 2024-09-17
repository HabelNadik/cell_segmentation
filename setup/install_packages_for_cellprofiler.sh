#!/bin/bash

export PATH="$HOME/.local/bin:$PATH"

### the module load commands are specific for our cluster
### if by any chance you have the same setup, you may uncomment and use them directly instead of installing the modules yourself
#module load gcc/10.2.0
#module load Java/13.0.2
#module load vis/GTK+/3.24.23-GCCcore-10.2.0

#module load Python/3.8.2-GCCcore-9.3.0
#module load pybind11/2.4.3-GCCcore-9.3.0-Python-3.8.2
#module load OpenBLAS/0.3.12-GCC-9.3.0
#module load lapack/gcc/64/3.9.0

#### we first uninstall all packages to make sure that installation happens on a clean setup
python -m pip uninstall scikit-image -y
python -m pip uninstall scipy -y
python -m pip uninstall numpy -y
python -m pip uninstall h5py -y
python -m pip uninstall python-javabridge -y
python -m pip uninstall centrosome -y
python -m pip uninstall cellprofiler -y
python -m pip uninstall cellprofiler-core -y
python -m pip uninstall mahotas -y
#python -m pip uninstall Cython -y

echo "Done with uninstalling. Testing if any leftovers are still there"

ls -l ~/.local/lib/python3.8/site-packages/ | grep "centrosome"
ls -l ~/.local/lib/python3.8/site-packages/ | grep "cellprofiler"
ls -l ~/.local/lib/python3.8/site-packages/ | grep "javabridge"

find ~/.local/lib/python3.8 -name "*centrosome*"
find ~/.local/lib/python3.8 -name "*javabridge*"
find ~/.local/lib/python3.8 -name "*cellprofiler*"
find ~/.local/lib/python3.8 -name "*CellProfiler*"

echo "################"
echo "any leftovers?"
echo "################"

PYTHONDIR="$HOME/.local/lib/python3.8/site-packages"

echo "Checking if python site packages folder exists and removing all libs if that's the case"

if [ -d ${PYTHONDIR} ]; then
    echo "Site packages folder found. Removing libraries"
    echo "Location is"
    echo ${PYTHONDIR}/*
    rm -rf ${PYTHONDIR}/*
fi

### the actual installation starts here
echo "Starting installation. Beginning with upgrading pip"
python -m pip install --upgrade pip

echo "# installing flit-core"
python -m pip install flit-core --user

echo "# installing mesonpy"
python -m pip install meson-python --user

echo "# installing ninja"
python -m pip install ninja --user

echo "# Installing six"
python -m pip install six==1.15.0 --user

echo "# Installing Cython"
#python -m pip install Cython==0.29.16 --user
### trying to install higher Cython version here to be able to build scikit-image later
python -m pip install Cython==0.29.24 --user


echo "# Installing numpy"
python -m pip install numpy==1.20.1 --user --no-build-isolation 

echo "# installing numpy associated packages"
python -m pip install scikit-image==0.18.1 --user
python -m pip install scipy==1.6.1 --user

echo "# installing javabridge"
python -m pip install python-javabridge==4.0.3 --no-cache-dir --user 
echo "# Testing javabridge"
python ./installation_testing_scripts/test_javabridge.py

echo "# installing bioformats"
python -m pip install python-bioformats==4.0.5 --user --no-cache-dir
#python -m pip install centrosome==1.2.0 --no-binary :all: --user
#python -m pip install python-javabridge --use-pep517 --no-cache-dir --user
#python -m pip install python-javabridge --user
echo "# Installing attrdict"
python -m pip install attrdict --user
echo "# Installing wxPython"
python -m pip install wxPython==4.1.0 --user
#python -m pip install cellprofiler --user
#python -m pip install cellprofiler==4.2.1 --user
### call setup_cellprofiler.sh
echo "# Installing pandas"
python -m pip install pandas==1.2.3
echo "# Installing ipython"
python -m pip install ipython --user

echo "# installing centrosome"
python -m pip install centrosome==1.2.0 --user --no-cache-dir --no-build-isolation


echo "# installing mahotas"
python -m pip install mahotas==1.4.3 --user --no-cache-dir --no-build-isolation
echo "# installing inflect"
python -m pip install inflect==5.3.0 --user --no-cache-dir --no-build-isolation

cd ..
echo "Testing mahotas"
python ./installation_testing_scripts/test_mahotas.py

echo "Testing centrosome"
python ./installation_testing_scripts/test_centrosome.py
#python -m pip install .
echo "# Installing Cellprofiler"
#python -m pip install cellprofiler==4.2.1 --no-binary :all: --no-build-isolation --user
python -m pip install cellprofiler==4.2.1 --user --no-cache-dir

echo "# installing imagecodegs and imgaug"
python -m pip install imagecodecs==2021.2.26 --user
python -m pip install imgaug==0.4.0 --user

cd ~

echo "Testing centrosome"
python ./installation_testing_scripts/test_centrosome.py


echo "Testing cellprofiler"
#bash /nobackup/lab_boztug/projects/pop/code/screening/cluster_scripts/setup_scripts/setup_cellprofiler.sh
python ./installation_testing_scripts/test_cellprofiler.py
echo "Setting up installation for deep learning segmentation"
bash ./install_packages_for_segmentation.sh
echo "Testing cellprofiler again"
python ./installation_testing_scripts/test_cellprofiler.py