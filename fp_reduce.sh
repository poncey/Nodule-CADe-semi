#!/usr/bin/env bash

# Execute scripts in fold
python exclusion.py -b 16 --epochs 200 --lr 0.01 --epsilon 2.5  --lamb 10 --test 1 --semi-spv 1 --fold 0 --save-dir result_jyuu_lambda
python exclusion.py -b 16 --epochs 200 --lr 0.01 --epsilon 2.5  --lamb 10 --test 1 --semi-spv 1 --fold 1 --save-dir result_jyuu_lambda
python exclusion.py -b 16 --epochs 200 --lr 0.01 --epsilon 2.5  --lamb 10 --test 1 --semi-spv 1 --fold 2 --save-dir result_jyuu_lambda

python exclusion.py -b 16 --epochs 200 --lr 0.01 --epsilon 2.5  --lamb 100 --test 1 --semi-spv 0 --fold 0 --save-dir result_hyaku_lambda
python exclusion.py -b 16 --epochs 200 --lr 0.01 --epsilon 2.5  --lamb 100 --test 1 --semi-spv 0 --fold 1 --save-dir result_kyaku_lambda
python exclusion.py -b 16 --epochs 200 --lr 0.01 --epsilon 2.5  --lamb 100 --test 1 --semi-spv 0 --fold 2 --save-dir result_kyaku_lambda
