#!/bin/bash

python Main.py --mode train --model vit --dataset svhn
python Main.py --mode train --model vit --dataset caltech256
python Main.py --mode train --retrain --model resnet
