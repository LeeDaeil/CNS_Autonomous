import random as ran
from torch import multiprocessing as mp

from CNS_UDP_FAST import CNS
from COMMONTOOL import RLMem
from Emergency.V6_1_Net_Model_Emergency import *

import time
import copy


class Work_info:  # 데이터 저장 및 초기 입력 변수 선정
    def __init__(self):
        self.CURNET_COM_IP = '192.168.0.10'
        self.CNS_IP_LIST = ['192.168.0.7', '192.168.0.4', '192.168.0.2']
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
        # 부모 네트워크의 정보를 자식 네트워크로 업데이트
        for _ in range(0, self.LocalNet.NubNET):
            self.LocalNet.NET[_].load_state_dict(self.GlobalNet.NET[_].state_dict())
        # 옵티마이저 생성
        self.LocalOPT = NETOPTBOX(NubNET=self.LocalNet.NubNET, NET=self.GlobalNet.NET)

        # Work info
        self.W = Work_info()
        # RLMem info
        self.RLMem = RLMem(net_nub=self.LocalNet.NubNET)

        # CNS
        self.CNS = CNS(self.name, CNS_ip, CNS_port, Remote_ip, Remote_port, Max_len=self.W.TimeLeg)
        self.CNS.LoggerPath = 'V6_1_EOP'
        # SharedMem
        self.mem = MEM
        self.LocalMem = copy.deepcopy(self.mem)

        # 사용되는 파라메터
        self.PARA_info = {
            # 변수명 : {'Div': 몇으로 나눌 것인지, 'Round': 반올림, 'Type': 어디에 저장할 것인지.}
            'ZINST58': {'Div': 1000, 'Round': 5, 'Type': 'P'},
            'ZINST63': {'Div': 100, 'Round': 4, 'Type': 'P'},
            'ZVCT': {'Div': 100, 'Round': 4, 'Type': 'P'},
            'BFV122': {'Div': 1, 'Round': 2, 'Type': 'F'},
            'BPV145': {'Div': 1, 'Round': 2, 'Type': 'F'},
            'BPV122C': {'Div': 2, 'Round': 2, 'Type': 'C'},
            'BPV145C': {'Div': 2, 'Round': 2, 'Type': 'C'},
        }
        ## 사용되는 파라메터가 db_add.txt에 있는지 확인하는 모듈
        if self.mem['Iter'] == 0:
            # 사용되는 파라메터가 db_add.txt에 있는지 체크
            for _ in self.PARA_info.keys():
                if not f'v{_}' in self.CNS.mem.keys():
                    print(f'v{_} 값이 없음 db_add.txt에 추가할 것')
            # 역으로 db_add에 있으나 사용되지 않은 파라메터 출력
            for _ in self.CNS.mem.keys():
                if _[0] == 'v': # 첫글자가 v이면..
                    if not _[1:] in self.PARA_info.keys():
                        print(f'{_} 값이 없음 self.PARA_info에 추가할 것')
        ## -----------------------------------------------

        # GP Setting
        # self.fig_dict = {i_: plt.figure(figsize=(13, 13)) for i_ in ["ZINST58", "ZINST63", "ZVCT", "BFV122", "BPV145", "BFV122_CONT", "BPV145_CONT"]}
        # self.ax_dict = {i_: self.fig_dict[i_].add_subplot() for i_ in ["ZINST58", "ZINST63", "ZVCT", "BFV122", "BPV145", "BFV122_CONT", "BPV145_CONT"]}
        print(f'Make -- {self}')

    # ==============================================================================================================
    # 제어 신호 보내는 파트
    def send_action_append(self, pa, va):
        for _ in range(len(pa)):
            self.para.append(pa[_])
            self.val.append(va[_])

    def send_action(self, act=0, BFV122=0, PV145=0):
        # 전송될 변수와 값 저장하는 리스트
        self.para = []
        self.val = []

        if act == 0:
            self.send_action_append(["KSWO100", "KSWO89"], [1, 1])   # BFV122 Man,  PV145 Man

        if PV145 == 0:
            self.send_action_append(["KSWO90", "KSWO91"], [0, 0])  # PV145 Stay
        elif PV145 == 1:
            self.send_action_append(["KSWO90", "KSWO91"], [0, 1])  # PV145 Up
        elif PV145 == 2:
            self.send_action_append(["KSWO90", "KSWO91"], [1, 0])  # PV145 Down

        if BFV122 == 0:
            self.send_action_append(["KSWO101", "KSWO102"], [0, 0])  # BFV122 Stay
        elif BFV122 == 1:
            self.send_action_append(["KSWO101", "KSWO102"], [0, 1])  # BFV122 Up
        elif BFV122 == 2:
            self.send_action_append(["KSWO101", "KSWO102"], [1, 0])  # BFV122 Down

        # 최종 파라메터 전송
        self.CNS._send_control_signal(self.para, self.val)
    #
    # ==============================================================================================================
    # 입력 출력 값 생성

    def PreProcessing(self):
        # Network용 입력 값 재처리
        for k in self.PARA_info.keys():
            if self.PARA_info[k]['Type'] != 'C':    # Control 변수를 제외한 변수만 재처리
                self.CNS.mem[f'v{k}']['Val'] = TOOL.RoundVal(self.CNS.mem[k]['Val'],
                                                         self.PARA_info[k]['Div'],
                                                         self.PARA_info[k]['Round'])

        # Network에 사용되는 값 업데이트
        if True:
            # Tensor로 전환
            # self.S_Py = torch.tensor([self.PhyState[key] for key in self.PhyPara])
            S_py_list, S_Comp_list = [], []
            for k in self.PARA_info.keys():
                if self.PARA_info[f'{k}']['Type'] == 'P':
                    S_py_list.append(self.CNS.mem[f'{k}']['List'])
                if self.PARA_info[f'{k}']['Type'] == 'F':
                    S_Comp_list.append(self.CNS.mem[f'{k}']['List'])

            self.S_Py = torch.tensor(S_py_list)
            self.S_Py = self.S_Py.reshape(1, self.S_Py.shape[0], self.S_Py.shape[1])
            self.S_Comp = torch.tensor(S_Comp_list)
            self.S_Comp = self.S_Comp.reshape(1, self.S_Comp.shape[0], self.S_Comp.shape[1])

    def CNSStep(self):
        self.CNS.run_freeze_CNS()   # CNS에 취득한 값을 메모리에 업데이트
        self.PreProcessing()        # 취득된 값에 기반하여 db_add.txt의 변수명에 해당하는 값을 재처리 및 업데이트
        self.CNS._append_val_to_list()  # 최종 값['Val']를 ['List']에 저장

    def run(self):
        while True:
            # Get iter
            self.CurrentIter = self.mem['Iter']
            self.mem['Iter'] += 1
            # Mal function initial
            size, maltime = ran.randint(100, 600), ran.randint(30, 100) * 5
            # CNS initial
            self.CNS.reset(initial_nub=1, mal=True, mal_case=36, mal_opt=size, mal_time=maltime, file_name=self.CurrentIter)
            print(f'DONE initial {size}, {maltime}')

            # 진단 모듈 Tester !
            if self.CurrentIter != 0 and self.CurrentIter % 100 == 0:
                print(self.CurrentIter, 'Yes Test')
                self.PrognosticMode = True
            else:
                print(self.CurrentIter, 'No Test')
                self.PrognosticMode = False

            # Initial
            done = False

            # GP 이전 데이터 Clear
            # [self.ax_dict[i_].clear() for i_ in ["ZINST58", "ZINST63", "ZVCT", "BFV122", "BPV145", "BFV122_CONT", "BPV145_CONT"]]

            while not done:
                fulltime = 2
                t_max = 2       # total iteration = fulltime * t_max
                ep_iter = 0
                tun = [1000, 100, 100, 1, 1]
                ro = [5, 4, 4, 2, 2]

                # ProgRecodBox = {"Time": [], "ZINST58": [], "ZINST63": [], "ZVCT": [], "BFV122": [], "BPV145": [], "BFV122_CONT": [], "BPV145_CONT": []}   # recode 초기화
                # Timer = 0

                if self.PrognosticMode: # TODO 작업 필요함... 0817
                    for i in range(0, 2):
                        if i == 0: # Automode
                            # 초기 제어 Setting 보내기
                            self.send_action()
                            time.sleep(1)

                        # Test Mode
                        for save_time_leg in range(self.W.TimeLeg):
                            self.CNS.run_freeze_CNS()
                            self.MakeStateSet()
                            Timer, ProgRecodBox = self.Recode(ProgRecodBox, Timer, S_Py=self.S_Py, S_Comp=self.S_Comp)

                        for t in range(fulltime * t_max):  # total iteration
                            if t == 0 or t % 10 == 0: # 0스텝 또는 10 스텝마다 예지
                                copySPy, copySComp = copy.deepcopy(self.S_Py), copy.deepcopy(self.S_Comp)   # 내용만 Copy
                                copyRecodBox = copy.deepcopy(ProgRecodBox)
                                Temp_Timer = copy.deepcopy(Timer)
                                for PredictTime in range(t, fulltime * t_max):  # 시간이 갈수록 예지하는 시간이 줄어듬.
                                    save_ragular_para = {_: 0 for _ in range(self.LocalNet.NubNET)}
                                    # 예지된 값 생산
                                    for nubNet in range(0, self.LocalNet.NubNET):
                                        NetOut = self.LocalNet.NET[nubNet].GetPredictActorOut(x_py=copySPy, x_comp=copySComp)
                                        NetOut = NetOut.view(-1)  # (1, 2) -> (2, )
                                        act_ = NetOut.argmax().item()  # 행열에서 최대값을 추출 후 값 반환
                                        if nubNet in [0, 6, 7]:
                                            save_ragular_para[nubNet] = act_
                                        elif nubNet in [1]:
                                            save_ragular_para[nubNet] = round((act_ - 100) / 100000, 5)
                                        elif nubNet in [2, 3]:
                                            save_ragular_para[nubNet] = round((act_ - 100) / 10000, 4)
                                        elif nubNet in [4, 5]:
                                            save_ragular_para[nubNet] = round((act_ - 100) / 100, 2)
                                    # 예지된 값 저장 및 종료

                                    #
                                    copySPyLastVal = copySPy[:, :, -1:]  # [1, 3, 10] -> [1, 3, 1] 마지막 변수 가져옴.
                                    add_val = tensor([[[save_ragular_para[1]],
                                                       [save_ragular_para[2]],
                                                       [save_ragular_para[3]]]])
                                    copySPyLastVal = copySPyLastVal + add_val  # 마지막 변수에 예측된 값을 더해줌.
                                    copySPy = torch.cat((copySPy, copySPyLastVal), dim=2)  # 본래 텐서에 값을 더함.
                                    # copySPy = torch.tensor(copySPy)
                                    copySPy = copySPy[:, :, 1:]     # 맨뒤의 값을 자름.

                                    copySCompLastVal = copySComp[:, :, -1:]  # [1, 3, 10] -> [1, 3, 1] 마지막 변수 가져옴.
                                    copySCompLastVal = tensor([[[save_ragular_para[4]], [save_ragular_para[5]],
                                                                [save_ragular_para[6] / 2], [save_ragular_para[7] / 2],
                                                                ]])
                                    copySComp = torch.cat((copySComp, copySCompLastVal), dim=2)  # 본래 텐서에 값을 더함.
                                    # copySComp = torch.tensor(copySComp)
                                    copySComp = copySComp[:, :, 1:]  # 맨뒤의 값을 자름.

                                    # Recode
                                    Temp_Timer, copyRecodBox = self.Recode(copyRecodBox, Temp_Timer, S_Py=copySPy, S_Comp=copySComp)

                                # 예지 종료 결과값 Recode 그래픽화
                                [self.ax_dict[i_].plot(copyRecodBox["Time"], copyRecodBox[i_], label=f"{i_}_{t}") for i_ in ["ZINST58", "ZINST63", "ZVCT", "BFV122", "BPV145", "BFV122_CONT", "BPV145_CONT"]]
                                [self.fig_dict[i_].savefig(f"{i_}_{self.CurrentIter}_{t}.png") for i_ in ["ZINST58", "ZINST63", "ZVCT", "BFV122", "BPV145", "BFV122_CONT", "BPV145_CONT"]]

                            a_now = {_: 0 for _ in range(self.LocalNet.NubNET)}
                            for nubNet in range(0, self.LocalNet.NubNET):
                                # TOOL.ALLP(self.S_Py, 'S_Py')
                                # TOOL.ALLP(self.S_Comp, 'S_Comp')
                                NetOut = self.LocalNet.NET[nubNet].GetPredictActorOut(x_py=self.S_Py, x_comp=self.S_Comp)
                                NetOut = NetOut.view(-1)    # (1, 2) -> (2, )
                                # TOOL.ALLP(NetOut, 'Netout before Categorical')
                                act = torch.distributions.Categorical(NetOut).sample().item()  # 2개 중 샘플링해서 값 int 반환

                                if nubNet in [0, 6, 7]:
                                    a_now[nubNet] = act
                                elif nubNet in [1]:
                                    a_now[nubNet] = round((act - 100) / 100000, 5)
                                elif nubNet in [2, 3]:
                                    a_now[nubNet] = round((act - 100) / 10000, 4)
                                elif nubNet in [4, 5]:
                                    a_now[nubNet] = round((act - 100) / 100, 2)
                            # Send Act to CNS!
                            self.send_action(act=0, BFV122=a_now[6], PV145=a_now[7])

                            # CNS + 1 Step
                            self.CNS.run_freeze_CNS()
                            self.MakeStateSet(BFV122=a_now[6], PV145=a_now[7])
                            # Recode
                            Timer, ProgRecodBox = self.Recode(ProgRecodBox, Timer, S_Py=self.S_Py, S_Comp=self.S_Comp)

                        # END Test Mode CODE
                        [self.ax_dict[i_].grid() for i_ in ["ZINST58", "ZINST63", "ZVCT", "BFV122", "BPV145", "BFV122_CONT", "BPV145_CONT"]]
                        [self.ax_dict[i_].legend() for i_ in ["ZINST58", "ZINST63", "ZVCT", "BFV122", "BPV145",  "BFV122_CONT", "BPV145_CONT"]]
                        if i == 0:
                            [self.fig_dict[i_].savefig(f"{i_}_{self.CurrentIter}_M.png") for i_ in ["ZINST58", "ZINST63", "ZVCT", "BFV122", "BPV145", "BFV122_CONT", "BPV145_CONT"]]
                        else:
                            [self.fig_dict[i_].savefig(f"{i_}_{self.CurrentIter}_A.png") for i_ in ["ZINST58", "ZINST63", "ZVCT", "BFV122", "BPV145", "BFV122_CONT", "BPV145_CONT"]]
                        print('END TEST')
                else:
                    # Train Mode

                    # 초기 제어 Setting 보내기
                    self.send_action()
                    time.sleep(1)

                    # Time Leg 만큼 데이터 수집만 수행
                    for t in range(self.W.TimeLeg + 1):
                        self.CNSStep()

                    # 실제 훈련 시작 부분
                    for __ in range(fulltime):
                        # spy_lst, scomp_lst, a_lst, r_lst = [], [], [], []
                        # a_dict = {_: [] for _ in range(self.LocalNet.NubNET)}
                        # a_now = {_: 0 for _ in range(self.LocalNet.NubNET)}
                        # a_now_orgin = {_: 0 for _ in range(self.LocalNet.NubNET)}
                        # a_prob = {_: [] for _ in range(self.LocalNet.NubNET)}
                        #
                        # r_dict = {_: [] for _ in range(self.LocalNet.NubNET)}
                        # done_dict = {_: [] for _ in range(self.LocalNet.NubNET)}
                        #
                        # y_predict = {_: [] for _ in range(self.LocalNet.NubNET)}
                        # y_answer = {_: [] for _ in range(self.LocalNet.NubNET)}
                        self.RLMem.CleanTrainMem()
                        # Sampling
                        for t in range(t_max):
                            NetOut_dict = {_: 0 for _ in range(self.LocalNet.NubNET)}
                            for nubNet in range(0, self.LocalNet.NubNET):
                                # TOOL.ALLP(self.S_Py, 'S_Py')
                                # TOOL.ALLP(self.S_Comp, 'S_Comp')

                                # 입력 변수들에서 Actor 네트워크의 출력을 받음.
                                NetOut = self.LocalNet.NET[nubNet].GetPredictActorOut(x_py=self.S_Py, x_comp=self.S_Comp)
                                NetOut = NetOut.view(-1)    # (1, 2) -> (2, )
                                # TOOL.ALLP(NetOut, 'Netout before Categorical')

                                # act 계산 이때 act는 int 값.
                                act = torch.distributions.Categorical(NetOut).sample().item()  # 2개 중 샘플링해서 값 int 반환
                                # TOOL.ALLP(act, 'act')

                                # act의 확률 값을 반환
                                NetOut = NetOut.tolist()[act]
                                # TOOL.ALLP(NetOut, f'NetOut{nubNet}')

                                # act와 확률 값 저장
                                self.RLMem.SaveNetOut(nubNet, NetOut, act)

                                # NetOut_dict[nubNet] = NetOut
                                # TOOL.ALLP(NetOut_dict, f'NetOut{nubNet}')

                                modify_act = 0
                                if nubNet in [0, 6, 7]:
                                    modify_act = act
                                elif nubNet in [1]:
                                    modify_act = round((act - 100) / 100000, 5)
                                elif nubNet in [2, 3]:
                                    modify_act = round((act - 100) / 10000, 4)
                                elif nubNet in [4, 5]:
                                    modify_act = round((act - 100) / 100, 2)

                                # 수정된 act 저장 <- 주로 실제 CNS의 제어 변수에 이용하기 위해서 사용
                                self.RLMem.SaveModNetOut(nubNet, modify_act)

                                # a_now_orgin[nubNet] = act
                                # a_dict[nubNet].append([act])        # for training
                                # a_prob[nubNet].append([NetOut])     # for training

                            # 훈련용 상태 저장
                            self.RLMem.SaveState(self.S_Py, self.S_Comp)
                            # spy_lst.append(self.S_Py.tolist()[0])  # (1, 3, 15) -list> (3, 15)
                            # scomp_lst.append(self.S_Comp.tolist()[0])  # (1, 3, 15) -list> (3, 15)

                            # old val to compare the new val
                            self.old_phys = self.S_Py[:, :, -1:].data.reshape(3).tolist() # (3,)
                            self.old_comp = self.S_Comp[:, :, -1:].data.reshape(2).tolist() # (3,)
                            self.old_cns = [    # "ZINST58", "ZINST63", "ZVCT", "BFV122", "BPV145"
                                round(self.old_phys[0], 5), round(self.old_phys[1], 4), round(self.old_phys[2], 4),
                                round(self.old_comp[0], 2), round(self.old_comp[1], 2)
                            ]
                            # TOOL.ALLP(self.old_cns, "old_CNS")

                            # Send Act to CNS!
                            # self.send_action(act=0,
                            #                  BFV122=self.RLMem.GetAct(6),
                            #                  PV145=self.RLMem.GetAct(7))

                            # CNS + 1 Step
                            self.CNS.run_freeze_CNS()
                            # self.MakeStateSet(BFV122=a_now[6], PV145=a_now[7])

                            self.new_phys = self.S_Py[:, :, -1:].data.reshape(3).tolist()  # (3,)
                            self.new_comp = self.S_Comp[:, :, -1:].data.reshape(2).tolist()  # (3,)
                            self.new_cns = [  # "ZINST58", "ZINST63", "ZVCT", "BFV122", "BPV145"
                                round(self.new_phys[0], 5), round(self.new_phys[1], 4), round(self.new_phys[2], 4),
                                round(self.new_comp[0], 2), round(self.new_comp[1], 2)
                            ]

                            # Recode
                            # Timer, ProgRecodBox = self.Recode(ProgRecodBox, Timer, S_Py=self.S_Py, S_Comp=self.S_Comp)

                            # 보상 및 종료조건 계산
                            # r = {_: 0 for _ in range(0, self.LocalNet.NubNET)}
                            for nubNet in range(0, self.LocalNet.NubNET):      # 보상 네트워크별로 계산 및 저장
                                if nubNet in [0]:
                                    if self.CNS.mem['KCNTOMS']['Val'] < maltime:
                                        if self.RLMem.int_mod_action[nubNet] == 1:    # Malfunction
                                            self.RLMem.SaveReward(nubNet, -1)
                                        else:
                                            self.RLMem.SaveReward(nubNet, 1)
                                    else:
                                        if self.RLMem.int_mod_action[nubNet] == 1:    # Malfunction
                                            self.RLMem.SaveReward(nubNet, 1)
                                        else:
                                            self.RLMem.SaveReward(nubNet, -1)
                                elif nubNet in [1, 2, 3]:
                                    Dealta = self.new_cns[nubNet-1] - (self.old_cns[nubNet-1] +
                                                                       self.RLMem.int_mod_action[nubNet])
                                    bound = {1: 0.00001, 2: 0.0001, 3: 0.0001}
                                    if Dealta < - bound[nubNet]:
                                        self.RLMem.SaveReward(nubNet, -1)
                                        # r[nubNet] = - ((self.old_cns[nubNet - 1] + self.RLMem.int_mod_action[nubNet]) - self.new_cns[nubNet-1])
                                    elif Dealta > bound[nubNet]:
                                        self.RLMem.SaveReward(nubNet, -1)
                                        # r[nubNet] = - (- (self.old_cns[nubNet - 1] + self.RLMem.int_mod_action[nubNet]) + self.new_cns[nubNet - 1])
                                    else:
                                        self.RLMem.SaveReward(nubNet, 1)
                                    # TOOL.ALLP(Dealta, f"Dealta")
                                    # TOOL.ALLP(r[nubNet], f"{nubNet} R nubnet")
                                    # if r[nubNet] == 1:
                                    #     pass
                                    # else:
                                    #     if nubNet in [1]:
                                    #         r[nubNet] = round(round(r[nubNet], 5) * 1000, 2)  # 0.000__ => 0.__
                                    #     elif nubNet in [2, 3]:
                                    #         r[nubNet] = round(round(r[nubNet], 4) * 100, 2)  # 0.00__ => 0.__

                                    # TOOL.ALLP(r[nubNet], f"{nubNet} R nubnet round")
                                    # print(self.new_cns[nubNet-1], self.old_cns[nubNet-1], self.RLMem.int_mod_action[nubNet])
                                elif nubNet in [4, 5]:
                                    Dealta = self.new_cns[nubNet - 1] - self.RLMem.int_mod_action[nubNet]
                                    if Dealta < -0.01:
                                        # r[nubNet] = - ((self.RLMem.int_mod_action[nubNet]) - self.new_cns[nubNet - 1])
                                        self.RLMem.SaveReward(nubNet, - 1)
                                    elif Dealta > 0.01:
                                        # r[nubNet] = - (- (self.RLMem.int_mod_action[nubNet]) + self.new_cns[nubNet - 1])
                                        self.RLMem.SaveReward(nubNet, - 1)
                                    else:
                                        self.RLMem.SaveReward(nubNet, 1)
                                    # TOOL.ALLP(Dealta, f"Dealta")
                                    # TOOL.ALLP(r[nubNet], f"{nubNet} R nubnet")
                                    # r[nubNet] = round(r[nubNet], 3)
                                    # TOOL.ALLP(r[nubNet], f"{nubNet} R nubnet round")
                                    # print(self.new_cns[nubNet - 1], self.old_cns[nubNet - 1], self.RLMem.int_mod_action[nubNet])
                                elif nubNet in [6, 7]:
                                    Dealta = self.new_cns[1] - 0.55 # normal PZR level # 0.30 - 0.55 = - 0.25 # 0.56 - 0.55 = 0.01
                                    if Dealta < -0.005:      # 0.53 - 0.55 = - 0.02
                                        self.RLMem.SaveReward(nubNet, (self.new_cns[1] - 0.55) * 10)     # # 0.53 - 0.55 = - 0.02
                                    elif Dealta > 0.005:     # 0.57 - 0.55 = 0.02
                                        self.RLMem.SaveReward(nubNet, (0.55 - self.new_cns[1]) * 10)      # 0.55 - 0.57 = - 0.02
                                    else:
                                        self.RLMem.SaveReward(nubNet, 1)

                                # r_dict[nubNet].append(r[nubNet])

                                # 종료 조건 계산
                                if __ == 14 and t == t_max-1:
                                    done = True
                                self.RLMem.SaveDone(nubNet, done)

                            def dp_want_val(val, name):
                                return f"{name}: {self.CNS.mem[val]['Val']:4.4f}"

                            DIS = f"[{self.CurrentIter:3}]" + f"TIME: {self.CNS.mem['KCNTOMS']['Val']:5}|"
                            # for _ in r.keys():
                            #     DIS += f"{r[_]:6} |"
                            # for _ in NetOut_dict.keys():
                            #     DIS += f"[{NetOut_dict[_]:0.4f}-{self.RLMem.int_mod_action[_]:4}]"
                            # for para, _ in zip(["ZINST58", "ZINST63", "ZVCT", "BFV122", "BPV145"], [0, 1, 2, 3, 4]):
                            #     DIS += f"| {para}: {self.old_cns[_]:5.2f} | {self.new_cns[_]:5.2f}"
                            print(DIS)

                            # Logger
                            TOOL.log_add(file_name=f"{self.name}.txt", ep=self.CurrentIter, ep_iter=ep_iter, x=self.old_cns)
                            ep_iter += 1

                        # ==================================================================================================
                        # Train

                        gamma = 0.98
                        lmbda = 0.95

                        # 1 .. 10
                        spy_batch, scomp_batch = self.RLMem.GetBatch()
                        # 2 .. 10 + (1 Last value)
                        spy_fin, scomp_fin = self.RLMem.GetFinBatch(self.S_Py, self.S_Comp)

                        # 각 네트워크 별 Advantage 계산
                        # for nubNet in range(0, 6):
                        for nubNet in range(0, self.LocalNet.NubNET):
                            # GAE
                            # r_dict[nubNet]: (5,) -> (5,1)
                            # Netout : (5,1)
                            # done_dict[nubNet]: (5,) -> (5,1)
                            td_target = torch.tensor(self.RLMem.list_reward_temp[nubNet], dtype=torch.float).view(t_max, 1) + \
                                        gamma * self.LocalNet.NET[nubNet].GetPredictCrticOut(spy_fin, scomp_fin) * \
                                        torch.tensor(self.RLMem.list_done_temp[nubNet], dtype=torch.float).view(t_max, 1)
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
                            PreVal = PreVal.gather(1, torch.tensor(self.RLMem.list_action_temp[nubNet])) # PreVal_a
                            # TOOL.ALLP(PreVal, f"Preval {nubNet}")

                            # Ratio 계산 a/b == exp(log(a) - log(b))
                            # TOOL.ALLP(a_prob[nubNet], f"a_prob {nubNet}")
                            Preval_old_a_prob = torch.tensor(self.RLMem.list_porb_action_temp[nubNet], dtype=torch.float)
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