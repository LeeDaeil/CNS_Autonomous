import torch
from torch import multiprocessing as mp
from torch import nn, functional, optim
from AB_PPO.Net_Model_Torch import PPOModel
from AB_PPO.CNS_UDP_FAST import CNS

import time
import pandas as pd

learning_rate = 0.0002
update_interval = 5
gamma = 0.98
max_train_ep = 1800
max_test_ep = 2000

class Work_info:  # 데이터 저장 및 초기 입력 변수 선정
    def __init__(self):
        self.CURNET_COM_IP = '192.168.0.10'
        self.CNS_IP_LIST = ['192.168.0.9', '192.168.0.7', '192.168.0.4']
        self.CNS_PORT_LIST = [7100, 7200, 7300]
        self.CNS_NUMBERS = [1, 0, 0]

        self.TimeLeg = 10

class Agent(mp.Process):
    def __init__(self, GlobalNet, MEM, CNS_ip, CNS_port, Remote_ip, Remote_port):
        mp.Process.__init__(self)
        # Network info
        self.LocalNet = PPOModel(nub_para=2, time_leg=10)
        self.LocalNet.load_state_dict(GlobalNet.state_dict())
        self.optimizer = optim.Adam(GlobalNet.parameters(), lr=learning_rate)
        # CNS
        self.CNS = CNS(self.name, CNS_ip, CNS_port, Remote_ip, Remote_port)
        # SharedMem
        self.mem = MEM
        # Work info
        self.W = Work_info()

    def run(self):
        while True:
            self.CNS.init_cns(initial_nub=1)
            time.sleep(1)
            self.CNS._send_malfunction_signal(12, 100100, 15)
            time.sleep(1)


    # ==============================================================================================================
    # 제어 신호 보내는 파트
    def send_action_append(self, pa, va):
        for _ in range(len(pa)):
            self.para.append(pa[_])
            self.val.append(va[_])

    def send_action(self, act):
        # 전송될 변수와 값 저장하는 리스트
        self.para = []
        self.val = []

        # 최종 파라메터 전송
        self.CNS._send_control_signal(self.para, self.val)
    #
    # ==============================================================================================================
if __name__ == '__main__':
    W_info = Work_info()

    global_model = PPOModel(nub_para=2, time_leg=W_info.TimeLeg)
    global_model.share_memory()

    # Make shared mem
    MEM = mp.Manager().dict({'Val': []})

    workers = []
    for cnsip, com_port, max_iter in zip(W_info.CNS_IP_LIST, W_info.CNS_PORT_LIST, W_info.CNS_NUMBERS):
        if max_iter != 0:
            for i in range(1, max_iter + 1):
                workers.append(Agent(GlobalNet=global_model,
                                     MEM=MEM,
                                     CNS_ip=cnsip,  CNS_port=com_port + i,
                                     Remote_ip=W_info.CURNET_COM_IP, Remote_port=com_port + i))

    [_.start() for _ in workers]
    [_.join() for _ in workers]
