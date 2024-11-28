#!/bin/bash

###### PREREQUISITES ######
# - ./scripts/start.sh or ./scripts/install.sh

source .env >/dev/null 2>&1

echo "True class is Vijay Deverakonda"
echo "Inferring..."
python3 ea_see/recognition/component.py --infer --image "./archive/Faces/Faces/Vijay Deverakonda_114.jpg"

echo "True class is Zac Efron"
echo "Inferring..."
python3 ea_see/recognition/component.py --infer --image "./archive/Faces/Faces/Zac Efron_90.jpg"

echo "True class is Tom Cruise"
echo "Inferring..."
python3 ea_see/recognition/component.py --infer --image "./archive/Faces/Faces/Tom Cruise_57.jpg"

echo "True class is Virat Kohli"
echo "Inferring..."
python3 ea_see/recognition/component.py --infer --image "./archive/Faces/Faces/Virat Kohli_48.jpg"

echo "True class is Natalie Portman"
echo "Inferring..."
python3 ea_see/recognition/component.py --infer --image "./archive/Faces/Faces/Natalie Portman_104.jpg"

echo "True class is Marmik"
echo "Inferring..."
python3 ea_see/recognition/component.py --infer --image "./archive/Faces/Faces/Marmik_31.jpg"

echo "True class is Margot Robbie"
echo "Inferring..."
python3 ea_see/recognition/component.py --infer --image "./archive/Faces/Faces/Margot Robbie_71.jpg"
