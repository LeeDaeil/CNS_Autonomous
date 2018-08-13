import threading
import socket
from collections import deque
from struct import unpack
from time import sleep

class CnsDataShare(threading.Thread):
    def __init__(self, mother_mem, mother_mem_list, mother_mem_deque, mother_memory_nub,
                 clean, ip='192.168.0.29', port=7000):

        threading.Thread.__init__(self)
        # initial socket
        self.ip, self.port = ip, port  # remote computer
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((self.ip, self.port))

        self.data_share_mem = mother_mem
        self.data_share_mem_list = mother_mem_list
        self.data_share_mem_deque = mother_mem_deque
        self.data_share_mem__nub = mother_memory_nub
        self.nub = 0

        self.clean = clean

        # Make data_share_mem structure
        self.Make_shared_mem_structure(deque_line=5)

    # 0. While the process
    def run(self):
        print('Start_data_share process....')

        while True:
            if self.clean['Sig']:
                for __ in self.data_share_mem_list.keys():
                    self.data_share_mem_list[__]['Val'] = []
                self.data_share_mem__nub['Nub'] = []
                self.nub = 0
                self.clean['Sig'] = False
            self.update_mem()

    # 2. update mem from read CNS
    def update_mem(self):
        data = self.read_socketdata()
        for i in range(0, 4000, 20):
            sig = unpack('h', data[24+i: 26+i])[0]
            para = '12sihh' if sig == 0 else '12sfhh'
            pid, val, sig, idx = unpack(para, data[8+i:28+i])
            pid = pid.decode().rstrip('\x00') # remove '\x00'
            if pid != '':
                self.data_share_mem[pid]['Val'] = val
                self.data_share_mem_list[pid]['Val'].append(val)
                self.data_share_mem_deque[pid]['Val'].append(val)
        self.data_share_mem__nub['Nub'].append(self.nub)
        self.nub += 1

    # (sub) make shared memory
    def Make_shared_mem_structure(self, deque_line = 5):
        idx = 0
        # with open('./db.txt', 'r') as f:   # use unitest
        with open('./db.txt', 'r') as f:
            while True:
                temp_ = f.readline().split('\t')
                if temp_[0] == '':  # if empty space -> break
                    break
                sig = 0 if temp_[1] == 'INTEGER' else 1
                self.data_share_mem[temp_[0]] = {'Sig': sig, 'Val': 0, 'Num': idx}
                self.data_share_mem_list[temp_[0]] = {'Sig': sig, 'Val': [], 'Num': idx}
                self.data_share_mem_deque[temp_[0]] = {'Sig': sig, 'Val': deque(maxlen=deque_line), 'Num': idx}
                idx += 1

    # (sub) socket part function
    def read_socketdata(self):
        data, addr = self.sock.recvfrom(4008)
        return data
