import socket
import pickle
from struct import pack, unpack
from numpy import shape


class DataShare:
    def __init__(self, ip, port):
        # socket part
        self.ip, self.port = ip, port    # remote computer

        # cns-data memory
        self.mem = {}                   # {'PID val': {'Sig': sig, 'Val': val, 'Num': idx }}
        self.list_mem = {}              # {'PID val': {'Sig': sig, 'Val': [val], 'Num': idx }}

    # 1. memory reset and refresh UDP
    def reset(self):
        self.mem, self.list_mem = {}, {}
        self.initial_DB()
        for i in range(5):
            self.read_socketdata()
        print('Memory and UDP network reset ready...')

    # 2. update mem from read CNS
    def update_mem(self):
        data = self.read_socketdata()
        for i in range(0, 4000, 20):
            sig = unpack('h', data[24+i: 26+i])[0]
            para = '12sihh' if sig == 0 else '12sfhh'
            pid, val, sig, idx = unpack(para, data[8+i:28+i])
            pid = pid.decode().rstrip('\x00') # remove '\x00'
            if pid != '':
                self.mem[pid]['Val'] = val
                self.list_mem[pid]['Val'].append(val)

    # 3. change value and send
    def sc_value(self, para, val, cns_ip, cns_port):
        self.change_value(para, val)
        self.send_data(para, cns_ip, cns_port)

    # 4. dump list_mem as pickle (binary file)
    def save_list_mem(self, file_name):
        with open(file_name, 'wb') as f:
            print('{}_list_mem save done'.format(file_name))
            pickle.dump(self.list_mem, f)

    # (sub) socket part function
    def read_socketdata(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM) # socket definition
        sock.bind((self.ip, self.port))
        data, addr = sock.recvfrom(4008)
        sock.close()
        return data

    # (sub) initial memory
    def initial_DB(self):
        idx = 0
        #with open('./db.txt', 'r') as f:   # use unitest
        with open('./CNS/db.txt', 'r') as f:
            while True:
                temp_ = f.readline().split('\t')
                if temp_[0] == '':      # if empty space -> break
                    break
                sig = 0 if temp_[1] == 'INTEGER' else 1
                self.mem[temp_[0]] = {'Sig': sig, 'Val': 0, 'Num': idx}
                self.list_mem[temp_[0]] = {'Sig': sig, 'Val': [], 'Num': idx}
                idx += 1

    # (sub) change value in memory
    def change_value(self, para, val):  # 'para' and 'val' is list
        for i in range(shape(para)[0]):
            self.mem[para[i]]['Val'] = val[i]

    # (sub) send value form memory to CNS
    def send_data(self, para, cns_ip, cns_port):    # 'para' is list, 'cns_ip' is char, 'cns_port' is int
        UDP_header = b'\x00\x00\x00\x10\xa8\x0f'
        buffer = b'\x00'*4008
        temp_data = b''

        # make temp_data to send CNS
        for i in range(shape(para)[0]):
            pid_temp = b'\x00'*12
            pid_temp = bytes(para[i], 'ascii') + pid_temp[len(para[i]):]    # pid + \x00 ..

            para_sw = '12sihh' if self.mem[para[i]]['Sig'] == 0 else '12sfhh'

            temp_data += pack(para_sw,
                              pid_temp,
                              self.mem[para[i]]['Val'],
                              self.mem[para[i]]['Sig'],
                              self.mem[para[i]]['Num'])


        buffer = UDP_header + pack('h', shape(para)[0]) + temp_data + buffer[len(temp_data):]

        # send buffer data to CNS
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)  # socket definition
        sock.sendto(buffer,(cns_ip, cns_port))
        sock.close()

if __name__ == '__main__':

    # unit test
    test = DataShare('192.168.0.3', 7000) # current computer ip / port
    test.reset()
    for i in range(0,20):
        # read CNS data
        test.update_mem()
        print(test.mem['UOVER'])

        # change and send control para
        test.sc_value(['KSWO33', 'KSWO32'], [1, 0], '192.168.0.9', 7001) # CNS computer ip / port


