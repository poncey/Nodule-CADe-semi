import numpy as np
import os
np.random.seed(114514)

k_fold = 10

file_id = np.asarray(['{:0>3}'.format(i) for i in range(888)])
file_index = np.asarray([i % k_fold for i in range(888)])

k = np.arange(len(file_index))
np.random.shuffle(k)
file_id = file_id[k]

for i in range(k_fold):
    train_file_id = file_id[file_index!=i]
    val_file_id = train_file_id[:88]
    train_file_id = train_file_id[88:]
    test_file_id = file_id[file_index==i]
    if not os.path.exists('./luna_file_id/subset_{:d}'.format(i)):
        os.mkdir('./luna_file_id/subset_{:d}'.format(i))
    np.save("./luna_file_id/subset_{:d}".format(i) + "/train_file_id.npy",train_file_id)
    np.save("./luna_file_id/subset_{:d}".format(i) + "/val_file_id.npy",val_file_id)
    np.save("./luna_file_id/subset_{:d}".format(i) + "/test_file_id.npy",test_file_id)