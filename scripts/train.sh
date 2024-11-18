#!/bin/bash

###### PREREQUISITES ######
# - ./scripts/start.sh or ./scripts/install.sh

source .env >/dev/null 2>&1

python3 ea_see/recognition/component.py --train --dataset-type larger
