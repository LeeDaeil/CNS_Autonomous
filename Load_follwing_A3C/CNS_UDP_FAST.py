import socket
from struct import unpack, pack
from time import sleep
from numpy import shape


class CNS:
    def __init__(self, threrad_name, CNS_IP, CNS_Port, Remote_IP, Remote_Port):
        if True:
            # thread name
            self.th_name = threrad_name
            # Ip, Port
            self.Remote_ip, self.Remote_port = Remote_IP, Remote_Port
            self.CNS_ip, self.CNS_port = CNS_IP, CNS_Port
            # Read Socket
            self.resv_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.resv_sock.bind((self.Remote_ip, self.Remote_port))
            # Send Socket
            self.send_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            # SIZE BUFFER
            self.size_buffer_mem = 46008
            # SEND TICK
            self.want_tick = 10

        if True:
            # memory
            self.mem = self.make_mem_structure()

    def make_mem_structure(self):
        # 초기 shared_mem의 구조를 선언한다.
        idx = 0
        shared_mem = {}
        with open('./db.txt', 'r') as f:
            while True:
                temp_ = f.readline().split('\t')
                if temp_[0] == '':  # if empty space -> break
                    break
                sig = 0 if temp_[1] == 'INTEGER' else 1
                shared_mem[temp_[0]] = {'Sig': sig, 'Val': 0, 'Num': idx}
                idx += 1
        # 다음과정을 통하여 shared_mem 은 PID : { type. val, num }를 가진다.
        return shared_mem

    def update_mem(self):
        data, _ = self.resv_sock.recvfrom(self.size_buffer_mem)
        data = data[8:]
        # print(len(data)) data의 8바이트를 제외한 나머지 버퍼의 크기
        for i in range(0, len(data), 20):
            sig = unpack('h', data[16 + i: 18 + i])[0]
            para = '12sihh' if sig == 0 else '12sfhh'
            pid, val, sig, idx = unpack(para, data[i:20 + i])
            pid = pid.decode().rstrip('\x00')  # remove '\x00'
            if pid != '':
                self.mem[pid]['Val'] = val

    def _send_control_signal(self, para, val):
        '''
        조작 필요없음
        :param para:
        :param val:
        :return:
        '''
        for i in range(shape(para)[0]):
            self.mem[para[i]]['Val'] = val[i]
        UDP_header = b'\x00\x00\x00\x10\xa8\x0f'
        buffer = b'\x00' * 4008
        temp_data = b''

        # make temp_data to send CNS #
        for i in range(shape(para)[0]):
            pid_temp = b'\x00' * 12
            pid_temp = bytes(para[i], 'ascii') + pid_temp[len(para[i]):]  # pid + \x00 ..

            para_sw = '12sihh' if self.mem[para[i]]['Sig'] == 0 else '12sfhh'

            temp_data += pack(para_sw,
                              pid_temp,
                              self.mem[para[i]]['Val'],
                              self.mem[para[i]]['Sig'],
                              self.mem[para[i]]['Num'])

        buffer = UDP_header + pack('h', shape(para)[0]) + temp_data + buffer[len(temp_data):]

        self.send_sock.sendto(buffer, (self.CNS_ip, self.CNS_port))

    def _send_malfunction_signal(self, Mal_nub, Mal_opt, Mal_time):
        '''
        CNS_04_18.tar 버전에서 동작함.
        :param Mal_nub: Malfunction 번호
        :param Mal_opt: Malfunction operation
        :param Mal_time: Malfunction의 동작하는 시간
        :return:
        '''
        if Mal_time == 0:
            Mal_time = 5
        else:
            Mal_time = Mal_time * 5
        return self._send_control_signal(['KFZRUN', 'KSWO280', 'KSWO279', 'KSWO278'],
                                         [10, Mal_nub, Mal_opt, Mal_time])

    def run_cns(self):
        para = []
        sig = []
        # if self.mem['QPROREL']['Val'] >= 0.04 and self.mem['KBCDO17']['Val'] <= 1800:
        #     if self.mem['KBCDO17']['Val'] < 1780: # 1780 -> 1872
        #         para.append('KSWO213')
        #         sig.append(1)
        #     elif self.mem['KBCDO17']['Val'] >= 1780:
        #         para.append('KSWO213')
        #         sig.append(0)
        # if self.mem['KBCDO19']['Val'] >= 1780 and self.mem['KLAMPO224']['Val'] == 0: # and self.mem['QPROREL']['Val'] >= 0.15:
        #     para.append('KSWO244')
        #     sig.append(1)
        para.append('KFZRUN')
        # sig.append(3)
        sig.append(self.want_tick+100)     # 400 - 100 -> 300 tick 20 sec
        return self._send_control_signal(para, sig)

    def init_cns(self, initial_nub):
        # UDP 통신에 쌇인 데이터를 새롭게 하는 기능
        self._send_control_signal(['KFZRUN', 'KSWO277'], [5, initial_nub])
        while True:
            self.update_mem()
            if self.mem['KFZRUN']['Val'] == 6:
                # initial 상태가 완료되면 6으로 되고, break
                break
            elif self.mem['KFZRUN']['Val'] == 5:
                # 아직완료가 안된 상태
                pass
            else:
                # 4가 되는 경우: 이전의 에피소드가 끝나고 4인 상태인데
                self._send_control_signal(['KFZRUN'], [5])
                pass
            # sleep(1)

    def run_freeze_CNS(self):
        old_cont = self.mem['KCNTOMS']['Val'] + self.want_tick
        self.run_cns()
        while True:
            self.update_mem()
            new_cont = self.mem['KCNTOMS']['Val']
            if old_cont == new_cont:
                if self.mem['KFZRUN']['Val'] == 4:
                    # 1회 run 완료 시 4로 변환
                    break
                else:
                    pass
            else:
                pass


if __name__ == '__main__':
    module = CNS('Main', '192.168.0.9', 7101, '192.168.0.10', 7101)
    module.init_cns(1)
    print(module.mem['KFZRUN']['Val'], module.mem['KCNTOMS']['Val'])
    module._send_malfunction_signal(12, 100100, 10)
    sleep(1)
    print(module.mem['KFZRUN']['Val'], module.mem['KCNTOMS']['Val'])
    for _ in range(20):
        module.run_freeze_CNS()
        print(module.mem['KFZRUN']['Val'], module.mem['KCNTOMS']['Val'])

    module.init_cns(2)
    print(module.mem['KFZRUN']['Val'], module.mem['KCNTOMS']['Val'])
    for _ in range(5):
        module.run_freeze_CNS()
        print(module.mem['KFZRUN']['Val'], module.mem['KCNTOMS']['Val'])