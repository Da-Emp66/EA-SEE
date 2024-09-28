#!/bin/bash

fabricate_dataset () {
    if [ ! -f $DATASET_FABRICATOR_MODEL_DOWNLOAD_FILE ]; then
        wget $DATASET_FABRICATOR_MODEL_DOWNLOAD_URL
    fi
    if [ ! -d $DATASET_FABRICATOR_MODEL_FILE ]; then
        bzip2 -dk $DATASET_FABRICATOR_MODEL_DOWNLOAD_FILE
    fi

    python3 ea_see/dataset.py --model $DATASET_FABRICATOR_MODEL_FILE --image-dir $LOCAL_IMAGE_DIR --dataset $DATASET_DIR
}

source .env >/dev/null 2>&1

fabricate_dataset
