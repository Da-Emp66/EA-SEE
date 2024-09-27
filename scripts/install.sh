#!/bin/bash

install_dataset () {
    if [ ! -f $DATASET_FILE ]; then
        wget $DATASET_DOWNLOAD_URL
    fi
    if [ ! -d $DATASET_FILE ]; then
        bzip2 -dk $DATASET_FILE
    fi
}

download_weights () {
    if [ ! -f $WEIGHTS_FILE ]; then
        gdown $WEIGHTS_DOWNLOAD_URL
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

setup_environment () {
    source .env

    install_miniconda

    if conda env list | grep $CONDA_ENV >/dev/null 2>/dev/null ; then
        conda activate $CONDA_ENV
    else
        conda create -n $CONDA_ENV python=$PYTHON_VERSION -y
        conda activate $CONDA_ENV
    fi

    sudo apt-get update
    sudo apt-get install cmake g++ make -y

    pip3 install poetry
    poetry install
}

# Set up environment - if not already set up
setup_environment

# Install the dataset - if not already installed
install_dataset

# Download the pretrained weights - if not already installed
download_weights
