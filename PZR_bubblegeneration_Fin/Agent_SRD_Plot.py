import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

from concurrent.futures import ThreadPoolExecutor, wait

# 1. Initial config
DB_fold_name = '/DB_ep_srd'
s_dim = 4
names = [f's{_}' for _ in range(s_dim)] + ['r', 'd']
want_files = [2648, 3788, 632, 4169]

# 2. Make folders and back up save folders
dirpath = os.path.dirname(os.path.realpath(__file__))
dirpath_files = os.listdir(dirpath + DB_fold_name)
dirpath_files = [f'.{DB_fold_name}/{dirpath_file}' for dirpath_file in dirpath_files]

temp = []
for w_file in want_files:
    temp.append(dirpath_files[dirpath_files.index(f'.{DB_fold_name}/{w_file}.txt')])
dirpath_files = temp

dirpath_files_nub = len(dirpath_files)
print(dirpath_files, dirpath_files_nub)

# 3. Load file and plot

DB_GlBAL = {}

def read_db(file, i, names):
    global DB_GlBAL

    try:
        print(f'Call {file}')
        db = []
        with open(file, 'r') as f:
            while True:
                one_line = f.readline()
                if one_line != '':
                    s, rd = one_line.split(']')
                    s = s.replace('[', '').split(',')
                    s = list(map(float, s))

                    rd = rd.replace('\n', '').split(',')[1:]
                    r = float(rd[0])
                    d = bool(rd[1])
                    s += [r, d]
                    db.append(s)
                else:
                    break
    except Exception as e:
        print(e)

    db = np.array(db)
    temp = pd.DataFrame(db, columns=names)
    DB_GlBAL[i] = {'DB': temp, 'File_Name': file}
    return temp, file, i

pool = ThreadPoolExecutor(dirpath_files_nub)
futures = [pool.submit(read_db, file, i, names) for file, i in zip(dirpath_files, range(dirpath_files_nub))]
wait(futures)

for i in range(dirpath_files_nub):
    plt.plot(DB_GlBAL[i]['DB']['s0'], label=f"{DB_GlBAL[i]['File_Name']}")
plt.legend()
plt.show()