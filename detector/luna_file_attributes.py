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
   pass
