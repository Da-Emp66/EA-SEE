#!/bin/bash

install_custom_dataset () {
    if [ ! -f $CUSTOM_DATASET_DOWNLOAD_FILE ]; then
        gdown $CUSTOM_DATASET_DOWNLOAD_URL
    fi
    if [ ! -d $CUSTOM_DATASET_DIR ]; then
        tar -xzvf $CUSTOM_DATASET_DOWNLOAD_FILE $CUSTOM_DATASET_DIR
    fi

    rm $CUSTOM_DATASET_DOWNLOAD_FILE >/dev/null 2>&1
}

install_larger_dataset () {
    if [ ! -f $LARGER_DATASET_DOWNLOAD_FILE ]; then
        gdown $LARGER_DATASET_DOWNLOAD_URL
    fi
    if [ ! -d $LARGER_DATASET_DIR ]; then
        unzip $LARGER_DATASET_DOWNLOAD_FILE -d $LARGER_DATASET_DIR
    fi

    rm $LARGER_DATASET_DOWNLOAD_FILE >/dev/null 2>&1
}

install_premade_datasets () {
    install_custom_dataset

    install_larger_dataset
}

download_embedding_weights () {
    if [ ! -f $EMBEDDING_WEIGHTS_FILE ]; then
        gdown $EMBEDDING_WEIGHTS_DOWNLOAD_URL
    fi
}

install_miniconda () {
    if ! conda ; then
        mkdir -p ~/miniconda3
        wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
        bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
        rm ~/miniconda3/miniconda.sh
        ~/miniconda3/bin/conda init bash
        ~/miniconda3/bin/conda init zsh
    fi
}

install_dlib_with_cuda () {
    source ~/.bashrc
    conda activate $CONDA_ENV
    sudo apt-get update && sudo apt-get install gcc-11 g++-11
    # if ! nvcc ; then
    #     sudo apt-get update && sudo apt-get install nvidia-cuda-toolkit -y
    # fi
    sudo apt-get update && sudo apt-get install libavdevice-dev libavfilter-dev libavformat-dev -y
    sudo apt-get update && sudo apt-get install libavcodec-dev libswresample-dev libswscale-dev -y
    sudo apt-get update && sudo apt-get install libavutil-dev -y
    sudo apt-get update && sudo apt-get install libblas-dev -y
    sudo apt-get update && sudo apt-get upgrade -y

    if [ ! -d $BUILD_DIR ]; then
        mkdir -p $BUILD_DIR
    fi
    cd $BUILD_DIR

    if [ ! -d dlib ]; then
        git clone https://github.com/davisking/dlib.git
    fi
    cd dlib

    python setup.py install --set CMAKE_C_COMPILER='/usr/bin/gcc-11'
    # DLIB_USE_CUDA_COMPUTE_CAPABILITIES=89

    cd ../..
}

setup_environment () {
    source .env >/dev/null 2>&1

    install_miniconda

    if conda env list | grep $CONDA_ENV >/dev/null 2>/dev/null ; then
        conda activate $CONDA_ENV
    else
        conda create -n $CONDA_ENV python=$PYTHON_VERSION -y
        conda activate $CONDA_ENV
    fi

    sudo hwclock --hctosys
    sudo apt-get update && sudo apt-get install cmake g++ make -y
    sudo apt-get update && sudo apt install unzip -y

    pip3 install poetry
    poetry install
    install_dlib_with_cuda
}

# Set up environment - if not already set up
setup_environment

# Install the datasets - if not already installed
install_premade_datasets

# Download the pretrained weights - if not already downloaded
download_weights
