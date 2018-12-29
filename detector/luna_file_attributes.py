import numpy as np
import os
np.random.seed(114514)

k_fold = 5

file_id = np.asarray(['{:0>3}'.format(i) for i in range(888)])
file_index = np.asarray([i % k_fold for i in range(888)])

k = np.arange(len(file_index))
np.random.shuffle(k)
file_id = file_id[k]

unlabel_samples = int(0.4 * len(file_id))

unlabel_file_id = file_id[:unlabel_samples]
label_file_id = file_id[unlabel_samples:]

if not os.path.exists('./luna_file_id'):
    os.mkdir('./luna_file_id')
np.save('./luna_file_id/file_id_unlabel.npy', unlabel_file_id)


for k in range(k_fold):
    test_file_id = file_id[file_index == k]
    train_file_id = file_id[file_index != k]

    total_train_samples = int(0.4 * len(train_file_id))
    total_train_file_id = train_file_id[:total_train_samples]
    rpn_train_file_id = train_file_id[total_train_samples:]

    if not os.path.exists('./luna_file_id/subset_fold{:d}'.format(k)):
        os.makedirs('./luna_file_id/subset_fold{:d}'.format(k))
    np.save("./luna_file_id/subset_fold{:d}".format(k) + "/file_id_total_train.npy", total_train_file_id)
    np.save("./luna_file_id/subset_fold{:d}".format(k) + "/file_id_rpn_train.npy", rpn_train_file_id)
    np.save("./luna_file_id/subset_fold{:d}".format(k) + "/file_id_test.npy", test_file_id)
