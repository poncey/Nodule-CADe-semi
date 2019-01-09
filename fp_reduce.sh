#!/usr/bin/env bash

# Execute scripts in fold
python exclusion.py -b 32 --epochs 200 --lr 0.001 --epsilon 3  --lamb 1.0 --test 1 --semi-spv 0 --fold 0 --save-dir result_argument_lambda --argument 4
python exclusion.py -b 32 --epochs 200 --lr 0.001 --epsilon 3  --lamb 1.0 --test 1 --semi-spv 0 --fold 1 --save-dir result_argument_lambda --argument 4
python exclusion.py -b 32 --epochs 200 --lr 0.001 --epsilon 3  --lamb 1.0 --test 1 --semi-spv 0 --fold 2 --save-dir result_argument_lambda --argument 4

python exclusion.py -b 32 --epochs 200 --lr 0.001 --epsilon 3  --lamb 1.0 --test 1 --semi-spv 1 --fold 0 --save-dir result_argument_lambda --argument 4
python exclusion.py -b 32 --epochs 200 --lr 0.001 --epsilon 3  --lamb 1.0 --test 1 --semi-spv 1 --fold 1 --save-dir result_argument_lambda --argument 4
python exclusion.py -b 32 --epochs 200 --lr 0.001 --epsilon 3  --lamb 1.0 --test 1 --semi-spv 1 --fold 2 --save-dir result_argument_lambda --argument 4
