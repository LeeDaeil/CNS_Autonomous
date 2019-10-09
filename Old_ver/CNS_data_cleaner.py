import multiprocessing
import socket
from copy import deepcopy
from collections import deque
from struct import unpack
from time import sleep

class CnsDataCleaner(multiprocessing.Process):
    def __init__(self, mother_mem):
        multiprocessing.Process.__init__(self)
        # initial socket

        self.data_share_mem = mother_mem
        self.nub = 1

    def run(self):

        while True:
            if len(self.data_share_mem['Nub']) > 10:
                self.data_share_mem['Clean'] = True
            if self.data_share_mem['Clean']:
                print('Clean mother memory....')
                self.update_mem(deque_line=5)
                self.data_share_mem['Clean'] = False
            sleep(0.5)

    def update_mem(self, deque_line=5):
        idx = 0
        data_ = deepcopy(self.data_share_mem)

        with open('./db.txt', 'r') as f:
            while True:
                temp_ = f.readline().split('\t')
                if temp_[0] == '':  # if empty space -> break
                    break
                sig = 0 if temp_[1] == 'INTEGER' else 1
                data_['Single'][temp_[0]] = {'Sig': sig, 'Val': 0, 'Num': idx}
                data_['List'][temp_[0]] = {'Sig': sig, 'Val': [], 'Num': idx}
                data_['List_Deque'][temp_[0]] = {'Sig': sig, 'Val': deque(maxlen=deque_line), 'Num': idx}

        data_['Nub'] = []
        data_['Inter'] = 1

        # Overwrite in mother memory
        for __ in data_.keys():
            self.data_share_mem[__] = data_[__]
