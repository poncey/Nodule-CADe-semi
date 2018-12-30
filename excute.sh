# Excute codes in fold

# python detection.py --model detector.res18 -b 16 --epochs 100 --save-dir res18 --save-freq 25 --test 1 --fold 0
python detection.py --model detector.res18 -b 16 --epochs 100 --save-dir res18 --save-freq 25 --test 1 --fold 1
python detection.py --model detector.res18 -b 16 --epochs 100 --save-dir res18 --save-freq 25 --test 1 --fold 2