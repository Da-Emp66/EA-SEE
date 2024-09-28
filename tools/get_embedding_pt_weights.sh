#!/bin/bash

download_pretrained_weights () {
    # Retrieve the weights download file - if not already present
    if [ ! -f $CONVERSION_WEIGHTS_DOWNLOAD_FILE ]; then
        wget ${CONVERSION_WEIGHTS_DOWNLOAD_URL}
    fi

    # Unzip the weights download file - if not already unzipped
    if [ ! -d $CONVERSION_WEIGHTS_DOWNLOAD_FOLDER_UNZIPPED ]; then
        tar -xzvf ${CONVERSION_WEIGHTS_DOWNLOAD_FILE} ${CONVERSION_WEIGHTS_DOWNLOAD_FOLDER_UNZIPPED}
    fi
}

convert_weights () {
    # Convert weights from `.t7` format to `.pt` format
    python3 tools/weight_conversion.py -t7 ${CONVERSION_WEIGHTS_FILE} -pt ${WEIGHTS_FILE}
}

clean_up () {
    # Remove now unnecessary files and folders
    rm -rf ${CONVERSION_WEIGHTS_DOWNLOAD_FOLDER_UNZIPPED} >/dev/null 2>&1
    rm ${CONVERSION_WEIGHTS_DOWNLOAD_FILE} >/dev/null 2>&1
}

# Retrieve environment variables
source .env >/dev/null 2>&1

# Download the pretrained weights - if not already downloaded
download_pretrained_weights

# Convert the pretrained weights to a usable format
convert_weights

# Clean out no longer used folders and files
clean_up
