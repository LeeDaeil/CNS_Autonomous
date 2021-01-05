import matplotlib.pyplot as plt
import matplotlib.gridspec as grid
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
dirpath_files = [f'.{DB_fold_name}/{dirpath_file}' for dirpath_file in dirpath_files]
dirpath_files_nub = len(dirpath_files)
print(dirpath_files)

# 3. Load file and plot

DB_GlBAL = {}
DB_TOTAL = {}

def read_db(file, i, names):
    global DB_GlBAL, DB_TOTAL
    print(f'Call {file}')

    if 'tot.txt' in file:
        temp = pd.read_csv(file, names=['i', 'q1', 'q1_av', 'q2', 'q2_av',
                                        'p', 'p_av', 'r', 'r_av'])
        DB_TOTAL = {'DB': temp, 'File_Name': file}
    else:
        temp = pd.read_csv(file, names=names)
        DB_GlBAL[i] = {'DB': temp, 'File_Name': file}
        return temp, file, i

pool = ThreadPoolExecutor(dirpath_files_nub)
futures = [pool.submit(read_db, file, i, names) for file, i in zip(dirpath_files, range(dirpath_files_nub))]
wait(futures)

# Plot ----------
fig = plt.figure()
gs = grid.GridSpec(4, 2, figure=fig)

axes = [
    fig.add_subplot(gs[0:1, 0:1]),
    fig.add_subplot(gs[1:2, 0:1]),
    fig.add_subplot(gs[2:3, 0:1]),
    fig.add_subplot(gs[3:4, 0:1]),

    fig.add_subplot(gs[0:1, 1:2]),
    fig.add_subplot(gs[1:2, 1:2]),
    fig.add_subplot(gs[2:3, 1:2]),
    fig.add_subplot(gs[3:4, 1:2]),
]

# for i in range(0, dirpath_files_nub - 1, 100):
#     axes[0].plot(DB_GlBAL[i]['DB']['q1_loss'], label=f'q1_loss {i}')
#     axes[1].plot(DB_GlBAL[i]['DB']['q2_loss'], label=f'q2_loss {i}')
#     axes[2].plot(DB_GlBAL[i]['DB']['q1_loss'], label=f'q1_loss {i}')
#     axes[2].plot(DB_GlBAL[i]['DB']['q2_loss'], label=f'q2_loss {i}')
#     axes[3].plot(DB_GlBAL[i]['DB']['p_loss'], label=f'p_loss {i}')

DB_TOTAL['DB'].sort_values(by='i', ascending=False, inplace=True)
DB_TOTAL['DB'].reset_index(drop=True, inplace=True)

axes[0].plot(DB_TOTAL['DB']['i'], DB_TOTAL['DB']['q1'], label='q1-loss')
axes[1].plot(DB_TOTAL['DB']['i'], DB_TOTAL['DB']['q1_av'], label='q1-av_loss')
axes[2].plot(DB_TOTAL['DB']['i'], DB_TOTAL['DB']['q2'], label='q2-loss')
axes[3].plot(DB_TOTAL['DB']['i'], DB_TOTAL['DB']['q2_av'], label='q2-av_loss')

axes[4].plot(DB_TOTAL['DB']['i'], DB_TOTAL['DB']['p'], label='p_loss')
axes[5].plot(DB_TOTAL['DB']['i'], DB_TOTAL['DB']['p_av'], label='p-av_loss')
axes[6].plot(DB_TOTAL['DB']['i'], DB_TOTAL['DB']['r'], label='r')
axes[7].plot(DB_TOTAL['DB']['i'], DB_TOTAL['DB']['r_av'], label='r-av')

for ax in axes:
    ax.legend()

plt.show()