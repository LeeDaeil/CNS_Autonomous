import multiprocessing
import socket
from copy import deepcopy
from collections import deque
from struct import unpack
from time import sleep

class CnsDataShare(multiprocessing.Process):
    def __init__(self, mother_mem, ip='192.168.236.1', port=7000):
        multiprocessing.Process.__init__(self)
        # initial socket
        self.ip, self.port = ip, port  # remote computer
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((self.ip, self.port))

        self.data_share_mem = mother_mem

        # Make data_share_mem structure
        self.Make_shared_mem_structure(deque_line=5)

    # 0. While the process
    def run(self):
        print('Start_data_share process....')

        while True:
            self.update_mem()

    # 2. update mem from read CNS
    def update_mem(self):
        data = self.read_socketdata()
        # Deepcopy data structure of the mother memory
        data_ = deepcopy(self.data_share_mem)

        for i in range(0, 4000, 20):
            sig = unpack('h', data[24+i: 26+i])[0]
            para = '12sihh' if sig == 0 else '12sfhh'
            pid, val, sig, idx = unpack(para, data[8+i:28+i])
            pid = pid.decode().rstrip('\x00') # remove '\x00'
            if pid != '':
                data_['Single'][pid]['Val'] = val
                data_['List'][pid]['Val'].append(val)
                data_['List_Deque'][pid]['Val'].append(val)
        data_['Nub'].append(data_['Inter'])
        data_['Inter'] += 1

        for __ in data_.keys():
            self.data_share_mem[__] = data_[__]

    # (sub) make shared memory
    def Make_shared_mem_structure(self, deque_line = 5):
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
        # Overwrite in mother memory
        for __ in data_.keys():
            self.data_share_mem[__] = data_[__]

    # (sub) socket part function
    def read_socketdata(self):
        data, addr = self.sock.recvfrom(4008)
        return data
