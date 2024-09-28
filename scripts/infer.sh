#!/bin/bash

###### PREREQUISITES ######
# - ./scripts/start.sh or ./scripts/install.sh

source .env >/dev/null 2>&1

for i in {0..10} ; do
    python3 ea_see/component.py --infer samples/$i.jpg
done
