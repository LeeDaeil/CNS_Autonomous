"""
Builder: Daeil Lee 2021-01-03

"""
import os
import shutil
from datetime import datetime
from PZR_bubblegeneration_Fin import SAC_Discrete

# 1. Initial config
backup_fold_name = '/Back'
save_fold_names = ['/Model', '/DB', '/DB_ep', '/DB_ep_srd', '/DB_ep_net', '/TFBoard']

# 2. Make folders and back up save folders
dirpath = os.path.dirname(os.path.realpath(__file__))
dirpath_files = os.listdir(dirpath)
backpath = dirpath + backup_fold_name
backpath_files_nub = len(os.listdir(backpath))

check_dirpath = [os.path.isdir(dirpath + fold_) for fold_ in save_fold_names]

if True in check_dirpath:
    current_time = datetime.now()
    # Make Ver fold in backup fold
    os.mkdir(backpath + f'/Ver_{backpath_files_nub}')

    # Move before file, fold, and info
    for fold_ in save_fold_names:
        if not os.path.isdir(dirpath + fold_):
            pass    # Cann't find save_fold
        else:
            origin_path = dirpath + fold_
            target_path = backpath + f'/Ver_{backpath_files_nub}' + fold_
            shutil.move(origin_path, target_path)

    # Log
    with open(backpath + f'/Ver_{backpath_files_nub}/Back_up_log.txt', 'w') as f:
        f.write(f'Back time: {current_time}')

# Make fold
[os.mkdir(dirpath + fold_) for fold_ in save_fold_names]

# 3. Call RL Model and run
Agents = SAC_Discrete.SAC()