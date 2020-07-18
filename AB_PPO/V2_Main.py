import torch
import numpy as np
from torch import multiprocessing as mp
from torch import nn, functional, optim

from AB_PPO.CNS_UDP_FAST import CNS
from AB_PPO.COMMONTOOL import TOOL
from AB_PPO.V2_Net_Model_Torch import *

import time
import copy
from collections import deque
import pandas as pd


class Work_info:  # 데이터 저장 및 초기 입력 변수 선정
    def __init__(self):
        self.CURNET_COM_IP = '192.168.0.10'
        self.CNS_IP_LIST = ['192.168.0.9', '192.168.0.7', '192.168.0.4']
        self.CNS_PORT_LIST = [7100, 7200, 7300]
        self.CNS_NUMBERS = [5, 0, 0]

        self.TimeLeg = 10

        # TO CNS_UDP_FASE.py
        self.UpdateIterval = 5

    def WInfoWarp(self):
        Info = {
            'Iter': 0
        }
        print('초기 Info Share mem로 선언')
        return Info


class Agent(mp.Process):
    def __init__(self, GlobalNet, MEM, CNS_ip, CNS_port, Remote_ip, Remote_port):
        mp.Process.__init__(self)
        # Network info
        self.GlobalNet = GlobalNet
        self.LocalNet = NETBOX()
        for _ in range(0, self.LocalNet.NubNET):
            self.LocalNet.NET[_].load_state_dict(self.GlobalNet.NET[_].state_dict())
        self.LocalOPT = NETOPTBOX(NubNET=self.LocalNet.NubNET, NET=self.GlobalNet.NET)
        # CNS
        self.CNS = CNS(self.name, CNS_ip, CNS_port, Remote_ip, Remote_port)
        # SharedMem
        self.mem = MEM
        self.LocalMem = copy.deepcopy(self.mem)
        # Work info
        self.W = Work_info()
        print(f'Make -- {self}')

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
    # 입력 출력 값 생성
    def InitialStateSet(self):
        self.PhyPara = ['ZINST58', 'ZINST63']
        self.PhyState = {_:deque(maxlen=self.W.TimeLeg) for _ in self.PhyPara}

        self.COMPPara = ['BFV122', 'BPV145']
        self.COMPState = {_: deque(maxlen=self.W.TimeLeg) for _ in self.COMPPara}

    def MakeStateSet(self):
        # 값을 쌓음 (return Dict)
        [self.PhyState[_].append(self.PreProcessing(_, self.CNS.mem[_]['Val'])) for _ in self.PhyPara]
        [self.COMPState[_].append(self.PreProcessing(_, self.CNS.mem[_]['Val'])) for _ in self.COMPPara]

        # Tensor로 전환
        self.S_Py = torch.tensor([self.PhyState[key] for key in self.PhyPara])
        self.S_Py = self.S_Py.reshape(1, self.S_Py.shape[0], self.S_Py.shape[1])
        self.S_Comp = torch.tensor([self.COMPState[key] for key in self.COMPPara])
        self.S_Comp = self.S_Comp.reshape(1, self.S_Comp.shape[0], self.S_Comp.shape[1])

        # Old 1개 리스트
        self.S_ONE_Py = [self.PhyState[key][-1] for key in self.PhyPara]
        self.S_ONE_Comp = [self.COMPState[key][-1] for key in self.COMPPara]

    def PreProcessing(self, para, val):
        if para == 'ZINST58': val = round(val/1000, 7)      # 가압기 압력
        if para == 'ZINST63': val = round(val/100, 7)       # 가압기 수위
        return val

    # ==============================================================================================================

    def run(self):
        while True:
            self.CNS.init_cns(initial_nub=1)
            print('DONE initial')
            time.sleep(1)
            # self.CNS._send_malfunction_signal(12, 100100, 15)
            # time.sleep(1)

            # Get iter
            self.CurrentIter = self.mem['Iter']
            self.mem['Iter'] += 1
            print(self.CurrentIter)

            # Initial
            done = False
            self.InitialStateSet()

            while not done:
                for t in range(self.W.TimeLeg):
                    self.CNS.run_freeze_CNS()
                    self.MakeStateSet()

                for __ in range(15):
                    spy_lst, scomp_lst, a_lst, r_lst = [], [], [], []
                    a_dict = {_: [] for _ in range(self.LocalNet.NubNET)}
                    r_dict = {_: [] for _ in range(self.LocalNet.NubNET)}
                    # Sampling
                    for t in range(5):
                        TimeDB = {
                            'Netout': {}, # 0: .. 1:..
                        }
                        for nubNet in range(self.LocalNet.NubNET):
                            NetOut = self.LocalNet.NET[nubNet].GetPredictActorOut(x_py=self.S_Py, x_comp=self.S_Comp)
                            NetOut = NetOut.tolist()[0][0]  # (1, 1) -> (1, ) -> ()
                            TimeDB['Netout'][nubNet] = NetOut
                            a_dict[nubNet] = NetOut

                        spy_lst.append(self.S_Py.tolist()[0])  # (1, 2, 10) -list> (2, 10)
                        scomp_lst.append(self.S_Comp.tolist()[0])  # (1, 2, 10) -list> (2, 10)

                        old_before = {0: 0, 1: 0}
                        for nubNet in range(self.LocalNet.NubNET):
                            old_before[nubNet] = self.S_ONE_Py[nubNet] + TimeDB['Netout'][nubNet] * 100

                        self.CNS.run_freeze_CNS()
                        self.MakeStateSet()

                        r = {0: 0, 1: 0}

                        for nub_val in range(0, 2):
                            if self.S_ONE_Py[nub_val] - 0.0001 < old_before[nub_val] < self.S_ONE_Py[nub_val] + 0.0001:
                                r[nub_val] = 0.1
                            else:
                                r[nub_val] = -1
                        if r[0] == 0.1 and r[1] == 0.1:
                            t_r = 0.1
                        else:
                            t_r = -0.1
                        # t_r = r[0] + r[1]
                        # r_lst.append(t_r)

                        for nubNet in range(self.LocalNet.NubNET):      # 보상 네트워크별로 저장
                            r_dict[nubNet].append(r[nubNet])

                        print(self.CurrentIter, TimeDB['Netout'], self.S_ONE_Py[0] - 0.0001, old_before[0], self.S_ONE_Py[0],
                              self.S_ONE_Py[0] + 0.0001, '|',
                              self.S_ONE_Py[1] - 0.0001, old_before[1], self.S_ONE_Py[1], self.S_ONE_Py[1] + 0.0001,
                              '|', r[0], r[1], t_r)
                    # ==================================================================================================
                    # Train

                    gamma = 0.98
                    spy_fin = self.S_Py  # (1, 2, 10)
                    scomp_fin = self.S_Comp  # (1, 2, 10)
                    spy_batch = torch.tensor(spy_lst, dtype=torch.float)
                    scomp_batch = torch.tensor(scomp_lst, dtype=torch.float)

                    # 각 네트워크 별 Advantage 계산
                    for nubNet in range(self.LocalNet.NubNET):
                        R = 0.0 if done else self.LocalNet.NET[nubNet].GetPredictCrticOut(spy_fin, scomp_fin).item()
                        td_target_lst = []
                        for reward in r_dict[nubNet][::-1]:
                            R = gamma * R + reward
                            td_target_lst.append([R])
                        td_target_lst.reverse()

                        td_target = torch.tensor(td_target_lst)
                        value = self.LocalNet.NET[nubNet].GetPredictCrticOut(spy_batch, scomp_batch)
                        advantage = td_target - value

                        PreVal = self.LocalNet.NET[nubNet].GetPredictActorOut(spy_batch, scomp_batch)

                        loss = -torch.log(PreVal) * advantage.detach() + \
                               nn.functional.smooth_l1_loss(self.LocalNet.NET[nubNet].GetPredictCrticOut(spy_batch, scomp_batch),
                                                            td_target.detach())

                        self.LocalOPT.NETOPT[nubNet].zero_grad()
                        loss.mean().backward()
                        for global_param, local_param in zip(self.GlobalNet.NET[nubNet].parameters(),
                                                             self.LocalNet.NET[nubNet].parameters()):
                            global_param._grad = local_param.grad
                        self.LocalOPT.NETOPT[nubNet].step()
                        self.LocalNet.NET[nubNet].load_state_dict(self.GlobalNet.NET[nubNet].state_dict())

                print('DONE EP')
                break


if __name__ == '__main__':
    W_info = Work_info()
    GlobalModel = NETBOX()
    [GlobalModel.NET[_].share_memory() for _ in range(0, GlobalModel.NubNET)]   # Net 들을 Shared memory 에 선언

    # Make shared mem
    MEM = mp.Manager().dict(W_info.WInfoWarp())

    workers = []
    for cnsip, com_port, max_iter in zip(W_info.CNS_IP_LIST, W_info.CNS_PORT_LIST, W_info.CNS_NUMBERS):
        if max_iter != 0:
            for i in range(1, max_iter + 1):
                workers.append(Agent(GlobalNet=GlobalModel,
                                     MEM=MEM,
                                     CNS_ip=cnsip, CNS_port=com_port + i,
                                     Remote_ip=W_info.CURNET_COM_IP, Remote_port=com_port + i))

    [_.start() for _ in workers]
    [_.join() for _ in workers]