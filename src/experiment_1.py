import os
os.system(('python train.py --model ALECE --data STATS --wl_type static --delete_data_feature 1 --n_epoch 150 --experiments_dir ../experiment_1/delete'))
os.system(('python train.py --model ALECE --data STATS --wl_type static --delete_data_feature 0 --n_epoch 150 --experiments_dir ../experiment_1/notdelete'))
