import socket
import time as t

class UDP_net:
    def __init__(self, mode, settime=5, timemode=False):
        '''
        Network
        CNS (192.168.0.9, 8001) -> Remote(192.168.0.3, 8000)
        '''
        self.mode = mode # True : CNS, False: Remote
        self.timemode = timemode
        self.settime = settime
        self.CNS = {'ip':'192.168.0.11', 'port':7002}
        self.Remote = {'ip': '192.168.0.29', 'port': 8002}
        self.sock = ''

        self.buffer_size = 100

    def make_sock(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        if self.timemode:
            self.sock.settimeout(self.settime)
        self.bind_sock()

    def bind_sock(self):
        if self.mode:
            self.sock.bind((self.CNS['ip'], self.CNS['port']))
        else:
            self.sock.bind((self.Remote['ip'], self.Remote['port']))

    def read_data(self):
        try:
            data, addr = self.sock.recvfrom(self.buffer_size)
            return data
        except socket.timeout:
            return print('Socket time out...')

    def send_message(self, mesg):
        if self.mode:
            self.sock.sendto(mesg.encode(), (self.Remote['ip'], self.Remote['port']))
        else:
            self.sock.sendto(mesg.encode(), (self.CNS['ip'], self.CNS['port']))

    def close_sock(self):
        self.sock.close()



if __name__ == "__main__":
    UDP = UDP_net()
    while True:
        UDP.make_sock()

        UDP.send_message('Remote->CNS')
        print('Go Meg')
        try:
            UDP.read_data()
        except socket.timeout:
            print('Timeout')

        UDP.close_sock()
        t.sleep(1)
