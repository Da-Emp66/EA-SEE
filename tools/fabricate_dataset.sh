#!/bin/bash

fabricate_dataset () {
    if [ ! -f $DATASET_FABRICATOR_MODEL_DOWNLOAD_FILE ]; then
        wget $DATASET_FABRICATOR_MODEL_DOWNLOAD_URL
    fi
    if [ ! -d $DATASET_FABRICATOR_MODEL_FILE ]; then
        bzip2 -dk $DATASET_FABRICATOR_MODEL_DOWNLOAD_FILE
    fi

    python3 ea_see/recognition/dataset.py --model $DATASET_FABRICATOR_MODEL_FILE --image-dir $LOCAL_IMAGE_TRAIN_DIR --dataset $DATASET_DIR/train
    python3 ea_see/recognition/dataset.py --model $DATASET_FABRICATOR_MODEL_FILE --image-dir $LOCAL_IMAGE_TEST_DIR --dataset $DATASET_DIR/valid
}

source .env >/dev/null 2>&1

conda activate $CONDA_ENV

fabricate_dataset
