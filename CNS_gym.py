from numpy import shape
from sklearn.preprocessing import Normalizer
from struct import pack
from time import sleep
import socket


class gym:
    def __init__(self, mother_mem):
        self.mother_mem = mother_mem
        self.prameter_db = self.read_state_DB()

        self.Min_Max = [[0,    0,   0,   0,   0,   0,   0,   0,  0.0, -12.0,  1.0,
                        200.0, 270.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 0.01],
                       [228, 228, 228, 228, 228, 228, 228, 228, 10.0,  -3.0, 20.0,
                        350.0, 330.0, 350.0, 350.0, 350.0, 350.0, 350.0, 350.0, 0.20]]
        self.normalizer = Normalizer().fit(self.Min_Max)

    # 1. update current state
    def update_state(self):
        current_state = []
        for i in range(shape(self.prameter_db)[0]):
            current_state.append(self.mother_mem[self.prameter_db[i]]['Val'])
        # normalization
        current_state = self.normalizer.transform([current_state])  # (20,) -> (1,20)
        return current_state

    # 2. step
    def step(self, action, iter = 0, human = True):
        # 3.1 make action

        if human:
            pass
        else:
            if action == 0:
                self.Send_CNS(['KSWO33', 'KSWO32'], [1, 0], '192.168.0.11', 7001)
            elif action == 1:
                self.Send_CNS(['KSWO33', 'KSWO32'], [0, 0], '192.168.0.11', 7001)
            else:
                self.Send_CNS(['KSWO33', 'KSWO32'], [0, 1], '192.168.0.11', 7001)

        sleep(1)    # 액션 후 반응을 수집하는 것임

        next_state = self.update_state()
        reward, done = self.make_condition(iter)

        return next_state, reward, done

    # (sub 3.1) make reward
    def make_condition(self, iter=0):
        power = self.mother_mem['QPROREL']['Val']

        if iter <= 30:
            if power >= 0.025:
                return 1, True
            else:
                return 0, False
        else:  # iter = 31
            if power >= 0.025:
                return 1, True
            else:  # iter = 31 and power < 0.025
                return -1, True

    # (sub) read state DB
    def read_state_DB(self):
        temp_ = []
        with open('./Pa.ini', 'r') as f:
            while True:
                line_ = f.readline()
                if line_ == '':
                    break
                temp_.append(line_.split(',')[1])
        return temp_

    #################################################################
    ### SEND_CNS
    def Send_CNS(self, para, val, cns_ip, cns_port):
        for i in range(shape(para)[0]):
            self.mother_mem[para[i]]['Val'] = val[i]
        UDP_header = b'\x00\x00\x00\x10\xa8\x0f'
        buffer = b'\x00' * 4008
        temp_data = b''

        # make temp_data to send CNS
        for i in range(shape(para)[0]):
            pid_temp = b'\x00' * 12
            pid_temp = bytes(para[i], 'ascii') + pid_temp[len(para[i]):]  # pid + \x00 ..

            para_sw = '12sihh' if self.mother_mem[para[i]]['Sig'] == 0 else '12sfhh'

            temp_data += pack(para_sw,
                              pid_temp,
                              self.mother_mem[para[i]]['Val'],
                              self.mother_mem[para[i]]['Sig'],
                              self.mother_mem[para[i]]['Num'])

        buffer = UDP_header + pack('h', shape(para)[0]) + temp_data + buffer[len(temp_data):]

        # send buffer data to CNS
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)  # socket definition
        sock.sendto(buffer, (cns_ip, cns_port))
        sock.close()