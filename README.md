# Nodule-CADe-semi

Three-dimensional pulmonary nodule detection network based on RPN（RegionProposal Network）and VAT.

## Dependencies
SimpleITK() numpy(1.14.3) matplotlib(2.2.2)  scikit-image(0.13.1) scipy(1.1.0) pytorch(0.4.0)

## Instruction for runing
Training
1. Install all dependencies
2. Prepare stage1 data, LUNA data, and LUNA segment results [link here](https://luna16.grand-challenge.org/download/), unzip them to separate folders
3. Go to ./training and open config_training.py
4. Filling in stage1_data_path, luna_raw, luna_segment with the path mentioned above
5. Filling in luna_data, preprocess_result_path, with tmp folders
6. bash run_training.sh and wait for the finishing of training (it may take several days)

## Testing
1. unzip the stage 2 data
2. go to root folder
3. open config_submit.py, filling in datapath with the stage 2 data path
4. python main.py
5. get the results from prediction.csv

## Method introduction
### Dataset
the Lung Nodule Analysis 2016 dataset (abbreviated as LUNA)are used to train the model.The LUNA dataset includes 1186 nodule labels in 888
patients annotated by radiologists
### Preprocessing
1. Mask extraction
2. Convex hull and dilation 
3. Intensity normalization
### Three-dimensional CNN
1. Patch-based input for training
2. Network structure---the detector network consists of a U-Net backbone and an RPN output layer
3. Positive sample balancing
4. Hard negative mining
5. Image splitting during testing

