#!/bin/bash

###### PREREQUISITES ######
# - ./scripts/start.sh or ./scripts/install.sh

source .env >/dev/null 2>&1

python3 ea_see/recognition/component.py --infer --image "./archive/Faces/Faces/Vijay Deverakonda_114.jpg"
python3 ea_see/recognition/component.py --infer --image "./archive/Faces/Faces/Zac Efron_90.jpg"
python3 ea_see/recognition/component.py --infer --image "./archive/Faces/Faces/Tom Cruise_57.jpg"
python3 ea_see/recognition/component.py --infer --image "./archive/Faces/Faces/Virat Kohli_48.jpg"
python3 ea_see/recognition/component.py --infer --image "./archive/Faces/Faces/Natalie Portman_104.jpg"
python3 ea_see/recognition/component.py --infer --image "./archive/Faces/Faces/Marmik_31.jpg"
python3 ea_see/recognition/component.py --infer --image "./archive/Faces/Faces/Margot Robbie_71.jpg"
