import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

from concurrent.futures import ThreadPoolExecutor, wait


# 1. Initial config
DB_fold_name = '/DB_ep'
names = [
    'q1_loss', 'q2_loss', 'p_loss'
]

# 2. Make folders and back up save folders
dirpath = os.path.dirname(os.path.realpath(__file__))
dirpath_files = os.listdir(dirpath + DB_fold_name)
dirpath_files = [f'.{DB_fold_name}/{dirpath_file}' for dirpath_file in dirpath_files][0:2]
dirpath_files_nub = len(dirpath_files)
print(dirpath_files)

# 3. Load file and plot

DB_GlBAL = {}

def read_db(file, i, names):
    global DB_GlBAL

    print(f'Call {file}')
    temp = pd.read_csv(file, names=names)
    DB_GlBAL[i] = {'DB': temp, 'File_Name': file}
    return temp, file, i

pool = ThreadPoolExecutor(dirpath_files_nub)
futures = [pool.submit(read_db, file, i, names) for file, i in zip(dirpath_files, range(dirpath_files_nub))]
wait(futures)

print(DB_GlBAL[0]['DB'])