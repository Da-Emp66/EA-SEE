#!/bin/bash

# Install weights and build Docker container
. scripts/docker-install.sh

# Run Docker container in detached mode
docker compose up -d

# Run the client
python3 client.py 0
