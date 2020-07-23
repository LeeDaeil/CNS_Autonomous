import torch
import numpy as np
import random as ran
import matplotlib.pylab as plt
from torch import multiprocessing as mp
from torch import nn, functional, optim

from AB_PPO.CNS_UDP_FAST import CNS
from AB_PPO.COMMONTOOL import TOOL
from AB_PPO.V4_3_2_Net_Model_Torch import *

import time
import copy
from collections import deque
import pandas as pd


class Work_info:  # 데이터 저장 및 초기 입력 변수 선정
    def __init__(self):
        self.CURNET_COM_IP = '192.168.0.10'
        self.CNS_IP_LIST = ['192.168.0.9', '192.168.0.7', '192.168.0.4']
        self.CNS_PORT_LIST = [7100, 7200, 7300]
        self.CNS_NUMBERS = [1, 0, 0]

        self.TimeLeg = 15

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
        # GP Setting
        self.fig_dict = {i_: plt.figure(figsize=(13, 13)) for i_ in ["ZINST58", "ZINST63", "ZVCT", "BFV122", "BPV145"]}
        self.ax_dict = {i_: self.fig_dict[i_].add_subplot() for i_ in ["ZINST58", "ZINST63", "ZVCT", "BFV122", "BPV145"]}
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
        self.PhyPara = ['ZINST58', 'ZINST63', 'ZVCT']
        self.PhyState = {_: deque(maxlen=self.W.TimeLeg) for _ in self.PhyPara}

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
        if para == 'ZINST58': val = round(val/1000, 5)      # 가압기 압력
        if para == 'ZINST63': val = round(val/100, 4)       # 가압기 수위
        if para == 'ZVCT': val = round(val/100, 4)          # VCT 수위
        if para == 'BFV122': val = round(val, 2)            # BF122 Pos
        if para == 'BPV145': val = round(val, 2)            # BPV145 Pos
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
            # 진단 모듈 Tester !
            if self.CurrentIter != 0 and self.CurrentIter % 30 == 0:
                print(self.CurrentIter, 'Yes Test')
                self.PrognosticMode = True
            else:
                print(self.CurrentIter, 'No Test')
                self.PrognosticMode = False

            # Initial
            done = False
            self.InitialStateSet()

            # GP 이전 데이터 Clear
            [self.ax_dict[i_].clear() for i_ in ["ZINST58", "ZINST63", "ZVCT", "BFV122", "BPV145"]]

            while not done:
                fulltime = 15
                t_max = 5       # total iteration = fulltime * t_max
                tun = [1000, 100, 100, 1, 1]
                ro = [2, 2, 2, 2, 2]
                ProgRecodBox = {"ZINST58": [], "ZINST63": [], "ZVCT": [], "BFV122": [], "BPV145": []}   # recode 초기화

                if self.PrognosticMode:
                    # Test Mode
                    for t in range(self.W.TimeLeg):
                        self.CNS.run_freeze_CNS()
                        self.MakeStateSet()
                        [ProgRecodBox[i_].append(round(self.CNS.mem[i_]['Val'], r_)/t_) for i_, t_, r_ in zip(ProgRecodBox.keys(), tun, ro)]

                    for __ in range(fulltime*t_max):    # total iteration
                        if __ != 0 and __ % 10 == 0:     # 10Step 마다 예지
                            # copy self.S_Py, self.S_Comp
                            copySPy, copySComp = self.S_Py, self.S_Comp
                            copyRecodBox = {"ZINST58": [], "ZINST63": [], "ZVCT": [], "BFV122": [], "BPV145": []}   # recode 초기화
                            # TOOL.ALLP(copyRecodBox["ZINST58"], "CopySPy")
                            for PredictTime in range(__, fulltime*t_max):   # 시간이 갈수록 예지하는 시간이 줄어듬.
                                # 예지 시작
                                save_ragular_para = {_: 0 for _ in range(self.LocalNet.NubNET)}
                                for nubNet in range(0, self.LocalNet.NubNET):
                                    NetOut = self.LocalNet.NET[nubNet].GetPredictActorOut(x_py=copySPy, x_comp=copySComp)
                                    NetOut = NetOut.view(-1)    # (1, 2) -> (2, )
                                    act_ = NetOut.argmax().item()    # 행열에서 최대값을 추출 후 값 반환
                                    save_ragular_para[nubNet] = (act_ - 100)/100  # act_ 값이 값의 증감으로 변경
                                TOOL.ALLP(save_ragular_para, "save_ragular_para")

                                # copySPy, copySComp에 값 추가
                                # copySpy
                                copySPyLastVal = copySPy[:, :, -1:]  # [1, 3, 10] -> [1, 3, 1] 마지막 변수 가져옴.

                                add_val = tensor([[
                                    [round(save_ragular_para[0]/1000, 5)],
                                    [round(save_ragular_para[1]/100, 4)],
                                    [round(save_ragular_para[2]/100, 4)]
                                ]])
                                TOOL.ALLP(copySPyLastVal, "copySPyLastVal")
                                TOOL.ALLP(add_val, "add_val")
                                copySPyLastVal = copySPyLastVal + add_val     # 마지막 변수에 예측된 값을 더해줌.

                                copySPy = torch.cat((copySPy, copySPyLastVal), dim=2)   # 본래 텐서에 값을 더함.
                                # 반올림
                                TOOL.ALLP(copySPy.data.numpy(), "COPYSPY")
                                copySPy = np.around(copySPy.data.numpy(), decimals=5)
                                TOOL.ALLP(copySPy, "COPYSPY_Round")
                                copySPy = torch.tensor(copySPy)
                                copySPy = copySPy[:, :, 1:]     # 맨뒤의 값을 자름.
                                # TOOL.ALLP(copySPy.data.numpy(), "copySPy Next")

                                # copySComp
                                copySCompLastVal = copySComp[:, :, -1:]  # [1, 3, 10] -> [1, 3, 1] 마지막 변수 가져옴.
                                # TOOL.ALLP(copySCompLastVal.data.numpy(), "COPYSCOMP")
                                # copySpy와 다르게 copy SComp는 이전의 제어 값을 그대로 사용함.
                                #TODO
                                # 자기자신 자체
                                copySCompLastVal = tensor([[[round(save_ragular_para[3], 2)],
                                                            [round(save_ragular_para[4], 2)]]])

                                copySComp = torch.cat((copySComp, copySCompLastVal), dim=2)  # 본래 텐서에 값을 더함.
                                # 반올림
                                copySComp = np.around(copySComp.data.numpy(), decimals=3)
                                copySComp = torch.tensor(copySComp)
                                copySComp = copySComp[:, :, 1:]  # 맨뒤의 값을 자름.
                                # 결과값 Recode
                                copyRecodBox["ZINST58"].append(copySPyLastVal[0, 0, 0].item())
                                copyRecodBox["ZINST63"].append(copySPyLastVal[0, 1, 0].item())
                                copyRecodBox["ZVCT"].append(copySPyLastVal[0, 2, 0].item())

                                copyRecodBox["BFV122"].append(copySComp[0, 0, 0].item())
                                copyRecodBox["BPV145"].append(copySComp[0, 1, 0].item())
                            # 예지 종료 결과값 Recode 그래픽화
                            [self.ax_dict[i_].plot(ProgRecodBox[i_] + copyRecodBox[i_],
                                              label=f"{i_}_{__}") for i_ in ["ZINST58", "ZINST63", "ZVCT", "BFV122", "BPV145"]]

                        # plt.show()
                        # CNS + 1 Step
                        self.CNS.run_freeze_CNS()
                        self.MakeStateSet()
                        [ProgRecodBox[i_].append(round(self.CNS.mem[i_]['Val'], r_)/t_) for i_, t_, r_ in zip(ProgRecodBox.keys(), tun, ro)]

                    # END Test Mode CODE
                    [self.ax_dict[i_].grid() for i_ in ["ZINST58", "ZINST63", "ZVCT", "BFV122", "BPV145"]]
                    [self.ax_dict[i_].legend() for i_ in ["ZINST58", "ZINST63", "ZVCT", "BFV122", "BPV145"]]
                    [self.fig_dict[i_].savefig(f"{self.CurrentIter}_{i_}.png") for i_ in ["ZINST58", "ZINST63", "ZVCT", "BFV122", "BPV145"]]
                    print('END TEST')

                else:
                    # Train Mode
                    for t in range(self.W.TimeLeg):
                        self.CNS.run_freeze_CNS()
                        self.MakeStateSet()

                    for __ in range(fulltime):
                        spy_lst, scomp_lst, a_lst, r_lst = [], [], [], []
                        a_dict = {_: [] for _ in range(self.LocalNet.NubNET)}
                        a_now = {_: 0 for _ in range(self.LocalNet.NubNET)}
                        a_prob = {_: [] for _ in range(self.LocalNet.NubNET)}
                        r_dict = {_: [] for _ in range(self.LocalNet.NubNET)}
                        done_dict = {_: [] for _ in range(self.LocalNet.NubNET)}
                        # Sampling
                        for t in range(t_max):
                            NetOut_dict = {_: 0 for _ in range(self.LocalNet.NubNET)}
                            for nubNet in range(0, self.LocalNet.NubNET):
                                # TOOL.ALLP(self.S_Py, 'S_Py')
                                # TOOL.ALLP(self.S_Comp, 'S_Comp')
                                NetOut = self.LocalNet.NET[nubNet].GetPredictActorOut(x_py=self.S_Py, x_comp=self.S_Comp)
                                NetOut = NetOut.view(-1)    # (1, 2) -> (2, )
                                # TOOL.ALLP(NetOut, 'Netout before Categorical')

                                act = torch.distributions.Categorical(NetOut).sample().item()  # 2개 중 샘플링해서 값 int 반환
                                # TOOL.ALLP(act, 'act')
                                NetOut = NetOut.tolist()[act]
                                # TOOL.ALLP(NetOut, 'NetOut')
                                NetOut_dict[nubNet] = NetOut
                                # TOOL.ALLP(NetOut_dict, f'NetOut{nubNet}')

                                a_now[nubNet] = act
                                a_dict[nubNet].append([act])
                                a_prob[nubNet].append([NetOut])

                            spy_lst.append(self.S_Py.tolist()[0])  # (1, 2, 10) -list> (2, 10)
                            scomp_lst.append(self.S_Comp.tolist()[0])  # (1, 2, 10) -list> (2, 10)

                            # old val to compare the new val
                            ComparedPara = ["ZINST58", "ZINST63", "ZVCT", "BFV122", "BPV145"]
                            ComparedParaRound = [2, 2, 2, 2, 2]
                            self.old_cns = {para: round(self.CNS.mem[para]['Val'], pr) for para, pr in zip(ComparedPara,ComparedParaRound)}
                            # TOOL.ALLP(self.old_cns, "old_CNS")

                            # CNS + 1 Step
                            self.CNS.run_freeze_CNS()
                            self.MakeStateSet()
                            self.new_cns = {para: round(self.CNS.mem[para]['Val'], pr) for para, pr in zip(ComparedPara,ComparedParaRound)}

                            # 보상 및 종료조건 계산
                            r = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
                            pa = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
                            for nubNet in range(0, self.LocalNet.NubNET):      # 보상 네트워크별로 계산 및 저장
                                if nubNet == 0:
                                    if self.CNS.mem['KCNTOMS']['Val'] < maltime:
                                        if a_now[nubNet] == 1:    # Malfunction
                                            r[nubNet] = -1
                                        else:
                                            r[nubNet] = 1
                                    else:
                                        if a_now[nubNet] == 1:    # Malfunction
                                            r[nubNet] = 1
                                        else:
                                            r[nubNet] = -1
                                else:
                                    predict_a = round((a_now[nubNet] - 100) / 100, 2)
                                    pa[nubNet] = predict_a
                                    # TODO
                                    #  변수 타입에 따라서 로직 변화함.
                                    want_para_reward = {1: "ZVCT", 2: "ZINST58", 3: "ZINST63", 4: "BFV122", 5: "BPV145"}
                                    if nubNet < 4:
                                        if self.new_cns[want_para_reward[nubNet]] == self.old_cns[want_para_reward[nubNet]] + predict_a:
                                            r[nubNet] = 1
                                        else:
                                            DeZVCT = self.new_cns[want_para_reward[nubNet]] - (self.old_cns[want_para_reward[nubNet]] + predict_a)
                                            if DeZVCT < 0: # 예측된 값이 더 크다.
                                                # 12.2 - 12.1 -> 0.1
                                                # r[nubNet] = 1 - ((self.old_cns[want_para_reward[nubNet]] + predict_a) - self.new_cns[want_para_reward[nubNet]])
                                                r[nubNet] = - ((self.old_cns[want_para_reward[nubNet]] + predict_a) - self.new_cns[want_para_reward[nubNet]])
                                            else:   # 예측된 값이 더 작다.
                                                # 12.2 - 12.1 -> 0.3
                                                # r[nubNet] = 1 - ( - (self.old_cns[want_para_reward[nubNet]] + predict_a) + self.new_cns[want_para_reward[nubNet]])
                                                r[nubNet] = - (- (self.old_cns[want_para_reward[nubNet]] + predict_a) + self.new_cns[want_para_reward[nubNet]])
                                        r[nubNet] = round(r[nubNet], 3)     # 0.100 나와서 2자리에서 반올림.
                                    else:
                                        if self.new_cns[want_para_reward[nubNet]] == predict_a:
                                            r[nubNet] = 1
                                        else:
                                            DeZVCT = self.new_cns[want_para_reward[nubNet]] - predict_a
                                            if DeZVCT < 0: # 예측된 값이 더 크다.
                                                r[nubNet] = - (predict_a - self.new_cns[want_para_reward[nubNet]])
                                            else:
                                                r[nubNet] = - ( - predict_a + self.new_cns[want_para_reward[nubNet]])
                                        r[nubNet] = round(r[nubNet], 3)  # 0.100 나와서 3자리에서 반올림.

                                r_dict[nubNet].append(r[nubNet])
                                # TOOL.ALLP(r[nubNet], "r_nubNet")
                                # TOOL.ALLP(pa[nubNet], "pa_nubNet")
                                # 종료 조건 계산
                                if __ == 14 and t == t_max-1:
                                    done_dict[nubNet].append(0)
                                    done = True
                                else:
                                    done_dict[nubNet].append(1)

                            def dp_want_val(val, name):
                                return f"{name}: {self.CNS.mem[val]['Val']:4.4f}"

                            print(self.CurrentIter, f"{r[0]:6}|{r[1]:6}|{r[2]:6}|{r[3]:6}|{r[4]:6}|{r[5]:6}|",
                                  f'{NetOut_dict[0]:0.4f}', f'{NetOut_dict[1]:0.4f}',
                                  f'{NetOut_dict[2]:0.4f}', f'{NetOut_dict[3]:0.4f}',
                                  f'{NetOut_dict[4]:0.4f}', f'{NetOut_dict[5]:0.4f}',
                                  f"TIME: {self.CNS.mem['KCNTOMS']['Val']:5}",
                                  # dp_want_val('PVCT', 'VCT pressure'),
                                  f"VCT Level: {self.new_cns['ZVCT']}",
                                  f"{self.old_cns['ZVCT'] + pa[1]:5.2f} + {pa[1]:5.2f}",
                                  f"PZR pre: {self.new_cns['ZINST58']}",
                                  f"{self.old_cns['ZINST58'] + pa[2]:5.2f} + {pa[2]:5.2f}",
                                  f"PZR Level: {self.new_cns['ZINST63']}",
                                  f"{self.old_cns['ZINST63'] + pa[3]:5.2f} + {pa[3]:5.2f}",
                                  f"BFV122: {self.new_cns['BFV122']}",
                                  f"{self.new_cns['BFV122'] + pa[4]:5.2f} + {pa[4]:5.2f}",
                                  f"BFV122: {self.new_cns['BPV145']}",
                                  f"{self.new_cns['BPV145'] + pa[5]:5.2f} + {pa[5]:5.2f}",
                                  # dp_want_val('UPRT', 'PRT temp'), dp_want_val('ZINST48', 'PRT pressure'),
                                  # dp_want_val('ZINST36', 'Let-down flow'), dp_want_val('BFV122', 'Charging Valve pos'),
                                  # dp_want_val('BPV145', 'Let-down Valve pos'),
                                  )

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
                        for nubNet in range(0, self.LocalNet.NubNET):
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
                            PreVal = PreVal.gather(1, torch.tensor(a_dict[nubNet])) # PreVal_a
                            # TOOL.ALLP(PreVal, f"Preval {nubNet}")

                            # Ratio 계산 a/b == exp(log(a) - log(b))
                            # TOOL.ALLP(a_prob[nubNet], f"a_prob {nubNet}")
                            Preval_old_a_prob = torch.tensor(a_prob[nubNet], dtype=torch.float)
                            ratio = torch.exp(torch.log(PreVal) - torch.log(Preval_old_a_prob))
                            # TOOL.ALLP(ratio, f"ratio {nubNet}")

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
                            # print(self.CurrentIter, 'AgentNub: ', nubNet,
                            #       'adv: ', adv.mean().item(), 'loss: ', loss.mean().item(),
                            #       '= - min_val(', min_val.mean().item(), ') + Smooth(', smooth_l1_loss.mean().item(), ')')

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