import torch
import numpy as np
import random as ran
from torch import multiprocessing as mp
from torch import nn, functional, optim

from AB_PPO.CNS_UDP_FAST import CNS
from AB_PPO.COMMONTOOL import TOOL
from AB_PPO.V3_Net_Model_Torch import *

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
            time.sleep(1)

            size, maltime = ran.randint(100, 600), ran.randint(3, 5) * 5
            self.CNS._send_malfunction_signal(36, size, maltime)
            time.sleep(1)
            print(f'DONE initial {size}, {maltime}')

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
                    a_prob = {_: [] for _ in range(self.LocalNet.NubNET)}
                    r_dict = {_: [] for _ in range(self.LocalNet.NubNET)}
                    done_dict = {_: [] for _ in range(self.LocalNet.NubNET)}
                    # Sampling
                    t_max = 5
                    for t in range(t_max):
                        TimeDB = {
                            'Netout': {}, # 0: .. 1:..
                        }
                        for nubNet in range(self.LocalNet.NubNET):
                            NetOut = self.LocalNet.NET[nubNet].GetPredictActorOut(x_py=self.S_Py, x_comp=self.S_Comp)
                            NetOut = NetOut.view(-1)    # (1, 2) -> (2, )
                            act = torch.distributions.Categorical(NetOut).sample().item()  # 2개 중 샘플링해서 값 int 반환
                            # TOOL.ALLP(act, 'act')
                            NetOut = NetOut.tolist()[act]
                            # TOOL.ALLP(NetOut, 'NetOut')

                            TimeDB['Netout'][nubNet] = NetOut
                            a_dict[nubNet].append([act])
                            a_prob[nubNet].append([NetOut])

                        spy_lst.append(self.S_Py.tolist()[0])  # (1, 2, 10) -list> (2, 10)
                        scomp_lst.append(self.S_Comp.tolist()[0])  # (1, 2, 10) -list> (2, 10)

                        # CNS + 1 Step
                        self.CNS.run_freeze_CNS()
                        self.MakeStateSet()
                        # 보상 및 종료조건 계산
                        r = {0: 0, 1: 0}
                        for nubNet in range(self.LocalNet.NubNET):      # 보상 네트워크별로 계산 및 저장

                            if self.CNS.mem['KCNTOMS']['Val'] < maltime:
                                if act == 1:    # Malfunction
                                    r[nubNet] = -1
                                else:
                                    r[nubNet] = 1
                            else:
                                if act == 1:    # Malfunction
                                    r[nubNet] = 1
                                else:
                                    r[nubNet] = -1

                            r_dict[nubNet].append(r[nubNet])

                            # 종료 조건 계산
                            if __ == 14 and t == t_max-1:
                                done_dict[nubNet].append(0)
                                done = True
                            else:
                                done_dict[nubNet].append(1)

                        print(self.CurrentIter, r[0], NetOut)

                    # ==================================================================================================
                    # Train

                    gamma = 0.98
                    lmbda = 0.95

                    # 1 .. 10
                    spy_batch = torch.tensor(spy_lst, dtype=torch.float)
                    scomp_batch = torch.tensor(scomp_lst, dtype=torch.float)
                    # 2 .. 10 + (1 Last value)
                    spy_lst.append(self.S_Py.tolist()[0])
                    scomp_lst.append(self.S_Comp.tolist()[0])
                    spy_fin = torch.tensor(spy_lst[1:], dtype=torch.float)
                    scomp_fin = torch.tensor(scomp_lst[1:], dtype=torch.float)

                    # 각 네트워크 별 Advantage 계산
                    for nubNet in range(self.LocalNet.NubNET):
                        # GAE
                        # r_dict[nubNet]: (5,) -> (5,1)
                        # Netout : (5,1)
                        # done_dict[nubNet]: (5,) -> (5,1)
                        td_target = torch.tensor(r_dict[nubNet], dtype=torch.float).view(t_max, 1) + \
                                    gamma * self.LocalNet.NET[nubNet].GetPredictCrticOut(spy_fin, scomp_fin) * \
                                    torch.tensor(done_dict[nubNet], dtype=torch.float).view(t_max, 1)
                        delta = td_target - self.LocalNet.NET[nubNet].GetPredictCrticOut(spy_batch, scomp_batch)
                        delta = delta.detach().numpy()

                        adv_list = []
                        adv_ = 0.0
                        for reward in delta[::-1]:
                            adv_ = gamma * adv_ * lmbda + reward[0]
                            adv_list.append([adv_])
                        adv_list.reverse()
                        adv = torch.tensor(adv_list, dtype=torch.float)

                        PreVal = self.LocalNet.NET[nubNet].GetPredictActorOut(spy_batch, scomp_batch)
                        Preval_a = PreVal.gather(1, torch.tensor(a_dict[nubNet]))

                        # Ratio 계산 a/b == exp(log(a) - log(b))
                        Preval_old_a_prob = torch.tensor(a_prob[nubNet], dtype=torch.float)
                        ratio = torch.exp(torch.log(Preval_a) - torch.log(Preval_old_a_prob))

                        # surr1, 2
                        eps_clip = 0.1
                        surr1 = ratio * adv
                        surr2 = torch.clamp(ratio, 1 - eps_clip, 1 + eps_clip) * adv

                        min_val = torch.min(surr1, surr2)
                        smooth_l1_loss = nn.functional.smooth_l1_loss(self.LocalNet.NET[nubNet].GetPredictCrticOut(spy_batch, scomp_batch), td_target.detach())

                        loss = - min_val + smooth_l1_loss

                        self.LocalOPT.NETOPT[nubNet].zero_grad()
                        loss.mean().backward()
                        for global_param, local_param in zip(self.GlobalNet.NET[nubNet].parameters(),
                                                             self.LocalNet.NET[nubNet].parameters()):
                            global_param._grad = local_param.grad
                        self.LocalOPT.NETOPT[nubNet].step()
                        self.LocalNet.NET[nubNet].load_state_dict(self.GlobalNet.NET[nubNet].state_dict())

                        # TOOL.ALLP(advantage.mean())
                        print(self.CurrentIter, 'AgentNub: ', nubNet,
                              'adv: ', adv.mean().item(), 'loss: ', loss.mean().item(),
                              '= - min_val(', min_val.mean().item(), ') + Smooth(', smooth_l1_loss.mean().item(), ')')

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