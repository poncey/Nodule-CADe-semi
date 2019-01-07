from reducer.data import ExclusionDataset, load_data

luna_dir = '/home/user/wuyunheng/work/DataBowl3/data/luna/preprocessed_luna_data'
index_dir = 'reducer/detect_post'
nodule_dir = 'nodule-data'
fold = 2
# Generate data
train = ExclusionDataset(luna_dir, index_dir, fold=fold, phase='train')
X_train, y_train = load_data(train, nodule_dir)
print "X_train.shape: ", X_train.shape, "y_train.shape: ", y_train.shape

test = ExclusionDataset(luna_dir, index_dir, fold=fold, phase='test')
X_test, y_test, uids, centres = load_data(test, nodule_dir)
print "X_test.shape: ", X_test.shape, "y_test.shape: ", y_test.shape

unlabeled = ExclusionDataset(luna_dir, index_dir, fold=fold, phase='unlabeled')
X_ul = load_data(unlabeled, nodule_dir)
print "X_ul.shape:", X_ul.shape

