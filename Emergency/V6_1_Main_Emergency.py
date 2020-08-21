import random as ran
import os
import shutil
import matplotlib.pyplot as plt
import numpy as np
from torch import multiprocessing as mp

from CNS_UDP_FAST import CNS
from COMMONTOOL import RLMem
from Emergency.V6_1_Net_Model_Emergency import *

import time
import copy
# =======================================================
MAKE_FILE_PATH = './V6_1_EOP'
# os.mkdir(MAKE_FILE_PATH)
# =======================================================

class Work_info:  # 데이터 저장 및 초기 입력 변수 선정
    def __init__(self):
        self.CURNET_COM_IP = '192.168.0.29'
        self.CNS_IP_LIST = ['192.168.0.103', '192.168.0.4', '192.168.0.2']
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
        global MAKE_FILE_PATH

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

        # CNS
        self.CNS = CNS(self.name, CNS_ip, CNS_port, Remote_ip, Remote_port, Max_len=self.W.TimeLeg)
        self.CNS.LoggerPath = MAKE_FILE_PATH
        # SharedMem
        self.mem = MEM
        self.LocalMem = copy.deepcopy(self.mem)

        # 사용되는 파라메터
        self.PARA_info = {
            # 변수명 : {'Div': 몇으로 나눌 것인지, 'Round': 반올림, 'Type': 어디에 저장할 것인지.}
            'ZINST78': {'Div': 1000, 'Round': 5, 'Type': 'P'},
            'ZINST77': {'Div': 1000, 'Round': 5, 'Type': 'P'},
            'ZINST76': {'Div': 1000, 'Round': 5, 'Type': 'P'},
            'ZINST75': {'Div': 1000, 'Round': 5, 'Type': 'P'},
            'ZINST74': {'Div': 1000, 'Round': 5, 'Type': 'P'},
            'ZINST73': {'Div': 1000, 'Round': 5, 'Type': 'P'},

            'KLAMPO134': {'Div': 1, 'Round': 2, 'Type': 'F'},
            'KLAMPO135': {'Div': 1, 'Round': 2, 'Type': 'F'},
            'KLAMPO136': {'Div': 1, 'Round': 2, 'Type': 'F'},
            'WAFWS1': {'Div': 100, 'Round': 4, 'Type': 'F'},
            'WAFWS2': {'Div': 100, 'Round': 4, 'Type': 'F'},
            'WAFWS3': {'Div': 100, 'Round': 4, 'Type': 'F'},
            'KLAMPO9': {'Div': 1, 'Round': 2, 'Type': 'F'},

            'BPV122C': {'Div': 2, 'Round': 2, 'Type': 'C'},
            'BPV145C': {'Div': 2, 'Round': 2, 'Type': 'C'},

            'Reward0': {'Div': 1, 'Round': 2, 'Type': 'C'},
            'Reward1': {'Div': 1, 'Round': 2, 'Type': 'C'},
        }
        self.PARA_info_For_save = [
            'KCNTOMS',
            # P
            'ZINST78',  'ZINST77',  'ZINST76',      'vZINST78',  'vZINST77',  'vZINST76',
            'ZINST75',  'ZINST74',  'ZINST73',      'vZINST75',  'vZINST74',  'vZINST73',
            # F
            'WAFWS1',   'WAFWS2',   'WAFWS3',       'vWAFWS1',   'vWAFWS2',   'vWAFWS3',
        ]
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

        # RLMem info
        self.RLMem = RLMem(net_nub=self.LocalNet.NubNET, para_info=self.PARA_info_For_save)

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

    def send_Control(self, Order):
        return self.send_action_append(self.OrderBook[Order]['pa'], self.OrderBook[Order]['va'])

    def send_action(self, act=0, AuxVal=0):
        # Order Book
        self.OrderBook = {
            'Aux1ValveStay': {'pa': ["KSWO142", "KSWO143"], 'va': [0, 0]},
            'Aux1ValveDown': {'pa': ["KSWO142", "KSWO143"], 'va': [1, 0]},
            'Aux1ValveUp':   {'pa': ["KSWO142", "KSWO143"], 'va': [0, 1]},
            'Aux2ValveStay': {'pa': ["KSWO151", "KSWO152"], 'va': [0, 0]},
            'Aux2ValveDown': {'pa': ["KSWO151", "KSWO152"], 'va': [1, 0]},
            'Aux2ValveUp':   {'pa': ["KSWO151", "KSWO152"], 'va': [0, 1]},
            'Aux3ValveStay': {'pa': ["KSWO154", "KSWO155"], 'va': [0, 0]},
            'Aux3ValveDown': {'pa': ["KSWO154", "KSWO155"], 'va': [1, 0]},
            'Aux3ValveUp':   {'pa': ["KSWO154", "KSWO155"], 'va': [0, 1]},
            'Aux1PumpOn':   {'pa': ["KSWO141"], 'va': [1]},
            'Aux2PumpOn':   {'pa': ["KSWO150"], 'va': [1]},
            'Aux3PumpOn':   {'pa': ["KSWO153"], 'va': [1]},
        }

        # 전송될 변수와 값 저장하는 리스트
        self.para = []
        self.val = []

        # if act == 0:
        #     self.send_action_append(["KSWO100", "KSWO89"], [1, 1])   # BFV122 Man,  PV145 Man

        # 1] Aux valve control logic
        if True:
            AuxCaseBox, AuxCaseBoxCount = {}, 0
            for Aux1 in ['Aux1ValveStay', 'Aux1ValveDown', 'Aux1ValveUp']:
                for Aux2 in ['Aux2ValveStay', 'Aux2ValveDown', 'Aux2ValveUp']:
                    for Aux3 in ['Aux3ValveStay', 'Aux3ValveDown', 'Aux3ValveUp']:
                        AuxCaseBox[AuxCaseBoxCount] = [Aux1, Aux2, Aux3]
                        AuxCaseBoxCount += 1
            GetAux1, GetAux2, GetAux3 = AuxCaseBox[AuxVal]
            print(GetAux1, GetAux2, GetAux3,  AuxVal)
            self.send_Control(GetAux1)
            self.send_Control(GetAux2)
            self.send_Control(GetAux3)

        ## --- if then 절차서 로직 호출
        self.IFTHENProcedures()

        # 최종 파라메터 전송
        if self.para != []:
            self.CNS._send_control_signal(self.para, self.val)

    def IFTHENProcedures_Step_initial(self):
        self.IFTHENProcedures_STEP = {
            0: True,        # Trip check
            1: False,       # Turbine check
            2: False,       # SI Check
            3: False,       # 주급수 잠김
            4: False,       # 보조급수 펌프 운전 중인지 확인
            5: False,       # 보조급수 유량 제어 시작
            6: False,       # Last !!
        }
        self.IFTHENProcedures_ORDER = {
            5: False, 'Goal_5': 'Set_up33', # Goal_5= 'Set_up33', 'StayIn50'
        }
        pass

    def IFTHENProcedures(self):
        para, val = [], []
        print(f'현재 절차 {self.IFTHENProcedures_STEP}')
        # 현재 절차서의 위치 확인
        for NubPro in range(len(self.IFTHENProcedures_STEP)):
            if self.IFTHENProcedures_STEP[NubPro]:
                get_current_step = NubPro

        while True:
            # 1] 현재 위치가 마지막 부분인지 확인, 마지막이면 종료
            if self.IFTHENProcedures_STEP[len(self.IFTHENProcedures_STEP) - 1]:
                print(f'모든 절차 수행 완료 {self.IFTHENProcedures_STEP}')
                break
            # 2] 현재 위치의 로직 수행
            step_done = False
            # print(f'{get_current_step}-체크')
            if get_current_step == 0 and self.CNS.mem['KLAMPO9']['Val'] == 1: step_done = True
            if get_current_step == 1 and self.CNS.mem['KLAMPO195']['Val'] == 1: step_done = True
            if get_current_step == 2 and self.CNS.mem['KLAMPO6']['Val'] == 1: step_done = True
            if get_current_step == 3:
                if self.CNS.mem['BFV478']['Val'] == 0 and self.CNS.mem['BFV479']['Val'] == 0 and \
                        self.CNS.mem['BFV488']['Val'] == 0 and self.CNS.mem['BFV489']['Val'] == 0 and \
                        self.CNS.mem['BFV498']['Val'] == 0 and self.CNS.mem['BFV499']['Val'] == 0:
                    step_done = True
            if get_current_step == 4:
                if self.CNS.mem['KLAMPO134']['Val'] == 1 and self.CNS.mem['KLAMPO135']['Val'] == 1 and \
                        self.CNS.mem['KLAMPO136']['Val'] == 1:
                    step_done = True
                else:
                    if self.CNS.mem['KLAMPO134']['Val'] == 0:
                        self.send_Control('Aux1PumpOn')
                    if self.CNS.mem['KLAMPO135']['Val'] == 0:
                        self.send_Control('Aux2PumpOn')
                    if self.CNS.mem['KLAMPO136']['Val'] == 0:
                        self.send_Control('Aux3PumpOn')
            if get_current_step == 5:
                self.IFTHENProcedures_ORDER[5] = True   # 강화학습 모듈 동작!
                if self.CNS.mem['WAFWS1']['Val'] > 10 and self.CNS.mem['WAFWS2']['Val'] > 10 and self.CNS.mem['WAFWS3']['Val'] > 10:
                    step_done = True
                    self.IFTHENProcedures_ORDER['Goal_5'] = 'StayIn50'  # 강화학습 모듈 모드 변경.

            # 3] 값 업데이트: 현재 수행한 절차가 성공적이면 이렇게 아니면 탈출
            if step_done:
                self.IFTHENProcedures_STEP[get_current_step] = False
                self.IFTHENProcedures_STEP[get_current_step + 1] = True
                # 4] 해당 절차를 수행하였음으로 다음 절차로 진입하기 위해 현재 바라보는 스텝의 번호를 +1 증가시킴
                get_current_step += 1
            else:
                break
        pass

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
                    S_py_list.append(self.CNS.mem[f'v{k}']['List'])
                if self.PARA_info[f'{k}']['Type'] == 'F':
                    S_Comp_list.append(self.CNS.mem[f'v{k}']['List'])

            self.S_Py = torch.tensor(S_py_list)
            self.S_Py = self.S_Py.reshape(1, self.S_Py.shape[0], self.S_Py.shape[1])
            self.S_Comp = torch.tensor(S_Comp_list)
            self.S_Comp = self.S_Comp.reshape(1, self.S_Comp.shape[0], self.S_Comp.shape[1])

    def UpdateActInfoToCNSMEM(self):
        self.CNS.mem['vBPV122C']['Val'] = 2

        # 보상 결과를 저장
        for nubNet in range(self.LocalNet.NubNET):
            self.CNS.mem[f'vReward{nubNet}']['Val'] = self.RLMem.GetReward(nubNet)

    def CNSStep(self):
        self.CNS.run_freeze_CNS()   # CNS에 취득한 값을 메모리에 업데이트
        self.PreProcessing()        # 취득된 값에 기반하여 db_add.txt의 변수명에 해당하는 값을 재처리 및 업데이트
        self.CNS._append_val_to_list()  # 최종 값['Val']를 ['List']에 저장

    def CalculateReward(self):
        for nubNet in range(0, self.LocalNet.NubNET):  # 보상 네트워크별로 계산 및 저장
            if nubNet in [0]:
                if self.CNS.mem['KCNTOMS']['Val'] < self.maltime: # 비상 주입 전
                    if self.RLMem.int_mod_action[nubNet] == 1:  # Malfunction
                        self.RLMem.SaveReward(nubNet, -1)
                    else:
                        self.RLMem.SaveReward(nubNet, 1)
                else:
                    if self.RLMem.int_mod_action[nubNet] == 1:  # Malfunction
                        self.RLMem.SaveReward(nubNet, 1)
                    else:
                        self.RLMem.SaveReward(nubNet, -1)
            if nubNet in [1]:
                if not self.IFTHENProcedures_ORDER[5]:   # 비상 주입 전
                    self.RLMem.SaveReward(nubNet, 0)
                else:
                    r_ = 0
                    r_1, r_2, r_3, r_4 = 0, 0, 0, 0
                    if self.IFTHENProcedures_ORDER['Goal_5'] == 'Set_up33':
                        if self.CNS.mem['WAFWS1']['Val'] > 10 and self.CNS.mem['WAFWS2']['Val'] > 10 and self.CNS.mem['WAFWS3']['Val'] > 10:
                            r_1 = 0.01
                        else:
                            r_2 = -0.01

                    elif self.IFTHENProcedures_ORDER['Goal_5'] == 'StayIn50':
                        for sg in ['ZINST78', 'ZINST77', 'ZINST76']:    # 증기 발생기 Naro 범위 6~50퍼 유지
                            if 6 < self.CNS.mem[sg]['Val'] < 50:
                                # if self.RLMem.GetAct(1) != 0: # TODO 굳이 움직일 필요가 없는데 제어하는 것이 좋은 방향일지 생각해보기
                                r_3 = 0.01
                            else:
                                r_4 = -0.01
                    else:
                        print('ERROR!!-SG')
                    r_ = r_1 + r_2 + r_3 + r_4
                    print(r_1, r_2, r_3, r_4)
                    self.RLMem.SaveReward(nubNet, r_)

            # 종료 조건 계산
            # TODO 종료 조건 입력하기
            if self.ep_iter > 100:
                done = True
            else:
                done = False

            self.RLMem.SaveDone(nubNet, done)

        fin_done = done # TODO 지금은 1개니까 여러개 done 조건 생기면 추가할 것
        return fin_done

    def ShowDIS(self):
        def dp_want_val(val, name):
            return f"{name}: {self.CNS.mem[val]['Val']:4.4f}"

        DIS = f"[{self.CurrentIter:3}]" + f"TIME: {self.CNS.mem['KCNTOMS']['Val']:5}|"
        DIS += "[Reward "
        for _ in range(self.RLMem.net_nub):
            DIS += f"{self.RLMem.float_reward[_]:6} "
        DIS += "][Prob "
        for _ in range(self.RLMem.net_nub):
            DIS += f"{self.RLMem.float_porb_action[_]:2.4f} "
        DIS += "][Act "
        for _ in range(self.RLMem.net_nub):
            DIS += f"{self.RLMem.int_action[_]:6} "
        DIS += "]"
        # for _ in r.keys():
        #     DIS += f"{r[_]:6} |"
        # for _ in NetOut_dict.keys():
        #     DIS += f"[{NetOut_dict[_]:0.4f}-{self.RLMem.int_mod_action[_]:4}]"
        # for para, _ in zip(["ZINST58", "ZINST63", "ZVCT", "BFV122", "BPV145"], [0, 1, 2, 3, 4]):
        #     DIS += f"| {para}: {self.old_cns[_]:5.2f} | {self.new_cns[_]:5.2f}"
        print(DIS)

    def SaveOldNew(self):

        nub_phy_val = np.shape(self.S_Py)[1]
        nub_comp_val = np.shape(self.S_Comp)[1]

        phys = self.S_Py[:, :, -1:].data.reshape(nub_phy_val).tolist()  # (3,)
        comp = self.S_Comp[:, :, -1:].data.reshape(nub_comp_val).tolist()  # (3,)

        cns = [  # "ZINST58", "ZINST63", "ZVCT", "BFV122", "BPV145"
            phys[0], phys[1], phys[2], phys[3], phys[4], phys[5],
            comp[0], comp[1], comp[2], comp[3], comp[4], comp[5], comp[6],
            # round(phys[0], 5), round(phys[1], 4), round(phys[2], 4),
            # round(comp[0], 2), round(comp[1], 2)
        ]
        return phys, comp, cns

    def DrawFig(self):
        fig = plt.figure(figsize=(10, 9), constrained_layout=True)
        gs = fig.add_gridspec(5, 2)
        axs = [fig.add_subplot(gs[0:2, 0:1]),   # 0
               fig.add_subplot(gs[2:5, 0:1]),   # 1

               fig.add_subplot(gs[0:1, 1:2]),   # 2
               fig.add_subplot(gs[1:2, 1:2]),   # 3
               fig.add_subplot(gs[2:3, 1:2]),   # 4
               fig.add_subplot(gs[3:4, 1:2]),   # 5
               fig.add_subplot(gs[4:5, 1:2]),   # 6
               ]

        print('=' * 20)
        print(self.RLMem.GetGPAllAProb(0))

        # 원하는 값에 정보를 입력
        axs[0].plot(self.RLMem.GetGPX(), self.RLMem.GetGPY('WAFWS1'), label='Aux1')
        axs[0].plot(self.RLMem.GetGPX(), self.RLMem.GetGPY('WAFWS2'), label='Aux2')
        axs[0].plot(self.RLMem.GetGPX(), self.RLMem.GetGPY('WAFWS3'), label='Aux3')
        axs[0].grid()

        axs[1].plot(self.RLMem.GetGPX(), self.RLMem.GetGPY('ZINST78'), label='Sg1')
        axs[1].plot(self.RLMem.GetGPX(), self.RLMem.GetGPY('ZINST77'), label='Sg2')
        axs[1].plot(self.RLMem.GetGPX(), self.RLMem.GetGPY('ZINST76'), label='Sg3')
        axs[1].grid()
        #
        axs[2].plot(self.RLMem.GetGPX(), self.RLMem.GetGPAct(0), label='Agent0')
        axs[2].plot(self.RLMem.GetGPX(), self.RLMem.GetGPAct(1), label='Agent1')
        axs[2].grid()

        axs[3].plot(self.RLMem.GetGPX(), self.RLMem.GetGPR(0), label='Agent0')
        axs[3].plot(self.RLMem.GetGPX(), self.RLMem.GetGPR(1), label='Agent1')
        axs[3].grid()

        axs[4].plot(self.RLMem.GetGPX(), self.RLMem.GetGPAProb(0), label='Agent0')
        axs[4].plot(self.RLMem.GetGPX(), self.RLMem.GetGPAProb(1), label='Agent1')
        axs[4].grid()

        axs[5].plot(self.RLMem.GetGPX(), self.RLMem.GetGPAllAProb(0), label='Agent0')
        axs[5].grid()

        axs[6].plot(self.RLMem.GetGPX(), self.RLMem.GetGPAllAProb(1), label='Agent1')
        axs[6].grid()

        fig.savefig(fname=f'{self.CNS.LoggerPath}/img/{self.CurrentIter}.png', dpi=300, facecolor=None)

        # 그래프 정보 및 저장용 정보 덤프
        self.RLMem.DumpAllData(log_path=f'{self.CNS.LoggerPath}/log/{self.CurrentIter}.txt')
        pass

    def run(self):
        while True:
            # Get iter
            self.CurrentIter = self.mem['Iter']
            self.mem['Iter'] += 1
            # Mal function initial
            self.size, self.maltime = 10010, 20 #ran.randint(10020, 10030), ran.randint(2, 10) * 5
            self.malnub = 13
            # CNS initial
            self.CNS.reset(initial_nub=1, mal=True, mal_case=self.malnub, mal_opt=self.size, mal_time=self.maltime,
                           file_name=self.CurrentIter)
            print(f'DONE initial {self.size}, {self.maltime}')
            # If-then 절차서 모니터링 변수 초기화
            self.IFTHENProcedures_Step_initial()

            # 진단 모듈 Tester !    # TODO 수정할 것
            # if self.CurrentIter != 0 and self.CurrentIter % 100 == 0:
            #     print(self.CurrentIter, 'Yes Test')
            #     self.PrognosticMode = True
            # else:
            #     print(self.CurrentIter, 'No Test')
            #     self.PrognosticMode = False

            print(self.CurrentIter, 'No Test')
            self.PrognosticMode = False

            # Initial
            done = False

            # GP 이전 데이터 Clear
            # [self.ax_dict[i_].clear() for i_ in ["ZINST58", "ZINST63", "ZVCT", "BFV122", "BPV145", "BFV122_CONT", "BPV145_CONT"]]

            while not done:
                fulltime = 3 # 600 초? 10분..?
                self.t_max = 5       # total iteration = fulltime * self.t_max # TODO 나중에 지울 것 Prognostic mode에 만 적용중...
                self.ep_iter = 0
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

                        for t in range(fulltime * self.t_max):  # total iteration
                            if t == 0 or t % 10 == 0: # 0스텝 또는 10 스텝마다 예지
                                copySPy, copySComp = copy.deepcopy(self.S_Py), copy.deepcopy(self.S_Comp)   # 내용만 Copy
                                copyRecodBox = copy.deepcopy(ProgRecodBox)
                                Temp_Timer = copy.deepcopy(Timer)
                                for PredictTime in range(t, fulltime * self.t_max):  # 시간이 갈수록 예지하는 시간이 줄어듬.
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
                        if t == self.W.TimeLeg:
                            # 마지막 부분으로 이전까지 s s' 저장 s s' 저장 ... 을 s s' a 저장을 위해서 저장까지 돌지 않음.
                            self.CNS.run_freeze_CNS()  # CNS에 취득한 값을 메모리에 업데이트
                            self.PreProcessing()  # 취득된 값에 기반하여 db_add.txt의 변수명에 해당하는 값을 재처리 및 업데이트
                        else:
                            self.CNSStep()
                    
                    # 훈련용 메모리 초기화
                    self.RLMem.CleanTrainMem()
                    # 카운터 초기화 
                    UpCount = 0
                    UpCountLimit = 5    # 5번 마다 업데이트
                    
                    # 실제 훈련 시작 부분
                    for __ in range(fulltime):

                        # 1] 현 상태에 대한 정보를 토대로 네트워크 출력 값 계산
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

                            #-------------------------------------------------------
                            # 특정 모델이 어떤 조건에서는 특정 값만 만들도록 하는 부분
                            if nubNet == 1: # AUX Controller
                                if self.IFTHENProcedures_ORDER[5]: # 해당 액션이 수행이 가능한 부분
                                    pass
                                else:
                                    act = 0
                            #
                            # -------------------------------------------------------

                            # act의 확률 값을 반환
                            AllNetOut = NetOut.tolist()
                            NetOut = NetOut.tolist()[act]
                            # TOOL.ALLP(NetOut, f'NetOut{nubNet}')

                            # act와 확률 값 저장
                            self.RLMem.SaveNetOut(nubNet, AllNetOut, NetOut, act)
                            # TOOL.ALLP(NetOut_dict, f'NetOut{nubNet}')

                            # act의 값 수정. ==========================================
                            modify_act = 0
                            if nubNet in [0, 1]:
                                modify_act = act

                            # 수정된 act 저장 <- 주로 실제 CNS의 제어 변수에 이용하기 위해서 사용
                            self.RLMem.SaveModNetOut(nubNet, modify_act)

                        # 2] 훈련용 상태 저장
                        self.RLMem.SaveState(self.S_Py, self.S_Comp)

                        # 2.1] 그래프용 상태 저장
                        self.RLMem.SaveGPState(self.CNS.mem)

                        # 3] 보상 계산을 위해 이전 값을 저장
                        self.old_phys, self.old_comp, self.old_cns = self.SaveOldNew()

                        # 4] 네트워크 출력 값에 따라서 제어 신호 전송
                        self.send_action(AuxVal=self.RLMem.GetAct(1))
                        # self.send_action(act=0,
                        #                  BFV122=self.RLMem.GetAct(6),
                        #                  PV145=self.RLMem.GetAct(7))

                        # 5] 제어 신호 정보까지 합쳐서 현재 정보와 제어 정보를 메모리에 저장 및 로깅
                        self.UpdateActInfoToCNSMEM()    # 액션 값을 메모리에 업데이트
                        self.CNS._append_val_to_list()  # 최종 값['Val']를 ['List']에 저장

                        # 6] CNS + 1 초!
                        self.CNS.run_freeze_CNS()
                        self.PreProcessing()  # 취득된 값에 기반하여 db_add.txt의 변수명에 해당하는 값을 재처리 및 업데이트

                        # 7] 보상 계산을 위해 이전 값을 저장
                        self.new_phys, self.new_comp, self.new_cns = self.SaveOldNew()

                        # 8] 새로운 상태에 대한 보상 조건 계산 및 종료조건 계산
                        done = self.CalculateReward()

                        # 정보 프린트
                        self.ShowDIS()

                        # Logger
                        TOOL.log_add(file_name=f"{self.name}.txt", ep=self.CurrentIter,
                                     ep_iter=self.ep_iter, x=self.old_cns, mal_nub=self.malnub, mal_opt=self.size,
                                     mal_time=self.maltime)
                        self.ep_iter += 1
                        UpCount += 1

                        # 9] 일정 간격으로 네트워크 훈련 및 업데이트 및 종료되면 Train
                        if UpCount == UpCountLimit or done:
                            # ==================================================================================================
                            # Train
                            gamma = 0.98
                            lmbda = 0.95
    
                            # 1] 저장된 훈련용 메모리에서 1 .. 10까지 값을 배치로 추출
                            spy_batch, scomp_batch = self.RLMem.GetBatch()
                            # 2] 저장된 훈련용 메모리에 GAE를 방법을 사용하기 위해서 마지막 상태 값을 추가하여 2 .. 10 + (1 Last value)
                            spy_fin, scomp_fin = self.RLMem.GetFinBatch(self.S_Py, self.S_Comp)
    
                            # 3] 각 네트워크 별 Advantage 계산
                            # for nubNet in range(0, 6):
                            for nubNet in range(0, self.LocalNet.NubNET):
                                # 3.1] GAE 방법 사용
                                # r_dict[nubNet]: (5,) -> (5,1)
                                # Netout : (5,1)
                                # done_dict[nubNet]: (5,) -> (5,1)
                                td_target = torch.tensor(self.RLMem.list_reward_temp[nubNet], dtype=torch.float).view(UpCountLimit, 1) + \
                                            gamma * self.LocalNet.NET[nubNet].GetPredictCrticOut(spy_fin, scomp_fin) * \
                                            torch.tensor(self.RLMem.list_done_temp[nubNet], dtype=torch.float).view(UpCountLimit, 1)
                                delta = td_target - self.LocalNet.NET[nubNet].GetPredictCrticOut(spy_batch, scomp_batch)
                                delta = delta.detach().numpy()

                                # 3.2] Advantage 계산
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
    
                                # 3.3] Ratio 계산 a/b == exp(log(a) - log(b))
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

                                # 3.4] 최종 Global Net에 가중치를 업데이트  # TODO 이부분에 비율적으로 업데이트 고려할 것.
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

                            # ==================================================================================================
                            # UpCount 초기화 및 훈련용 메모리 초기화
                            UpCount = 0
                            self.RLMem.CleanTrainMem()

                # END
                self.DrawFig()
                print(f'{self.CurrentIter}--DONE EP')
                break


if __name__ == '__main__':

    os.mkdir(MAKE_FILE_PATH)

    fold_list = ['{}/log'.format(MAKE_FILE_PATH),
                 '{}/img'.format(MAKE_FILE_PATH),
                 ]
    # '{}/log/each_log'.format(MAKE_FILE_PATH),
    # '{}/log'.format(MAKE_FILE_PATH),
    # '{}/img'.format(MAKE_FILE_PATH)]
    for __ in fold_list:
        if os.path.isdir(__):
            shutil.rmtree(__)
            time.sleep(1)
            os.mkdir(__)
        else:
            os.mkdir(__)

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