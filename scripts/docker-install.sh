#!/bin/bash

download_weights () {
    if [ ! -f $WEIGHTS_FILE ]; then
        gdown $WEIGHTS_DOWNLOAD_URL
    fi
}

# Set up environment variables
source .env >/dev/null 2>&1

# Download the pretrained weights - if not already installed
download_weights

# Build the container
docker compose build
