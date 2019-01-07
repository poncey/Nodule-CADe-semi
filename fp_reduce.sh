# Execute scripts in fold
python exclusion.py -b 16 --epochs 120 --lr 0.01 --epsilon 2.5 --test 1 --semi-spv 1 --fold 0 --save-dir result_exclusion
python exclusion.py -b 16 --epochs 120 --lr 0.01 --epsilon 2.5 --test 1 --semi-spv 1 --fold 1 --save-dir result_exclusion
python exclusion.py -b 16 --epochs 120 --lr 0.01 --epsilon 2.5 --test 1 --semi-spv 1 --fold 2 --save-dir result_exclusion

python exclusion.py -b 16 --epochs 120 --lr 0.01 --epsilon 2.5 --test 1 --semi-spv 0 --fold 0 --save-dir result_exclusion
python exclusion.py -b 16 --epochs 120 --lr 0.01 --epsilon 2.5 --test 1 --semi-spv 0 --fold 1 --save-dir result_exclusion
python exclusion.py -b 16 --epochs 120 --lr 0.01 --epsilon 2.5 --test 1 --semi-spv 0 --fold 2 --save-dir result_exclusion
