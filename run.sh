#!/bin/bash

python Main.py --mode train --dataset svhn --retrain --model resnet
python Main.py --mode train --dataset kminst --retrain
