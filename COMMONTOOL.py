import numpy as np
import os
from datetime import datetime
from collections import deque
import torch

class TOOL:
    @staticmethod
    def ALLP(val, comt=''):
        print(f"======{comt}======")
        print(f'{val}')
        print(f'TOP\t\tTYPE: {type(val)} \t SHAPE: {np.shape(val)}')
        for _ in range(0, len(np.shape(val)) + 1):
            if _ == 0:
                pass
            elif _ == 1:
                TOOL._transfer_(val[0])
                print(f'SUB[0]\t\tTYPE: {type(val[0])} \t SHAPE: {np.shape(val[0])}')
            elif _ == 2:
                TOOL._transfer_(val[0][0])
                print(f'SUB[0][0]\t\tTYPE: {type(val[0][0])} \t SHAPE: {np.shape(val[0][0])}')
            elif _ == 3:
                TOOL._transfer_(val[0][0][0])
                print(f'SUB[0][0][0]\tTYPE: {type(val[0][0][0])} \t SHAPE: {np.shape(val[0][0][0])}')
            elif _ == 4:
                TOOL._transfer_(val[0][0][0][0])
                print(f'SUB[0][0][0][0]\tTYPE: {type(val[0][0][0][0])} \t SHAPE: {np.shape(val[0][0][0][0])}')
            else:
                print('OVER!!')
        print("============")

    @staticmethod
    def _transfer_(x):
        if str(type(x)) == "<class 'torch.Tensor'>":
            return print(x.data.numpy())
        else:
            return print(x)

    @staticmethod
    def log_add(file_name, ep, ep_iter, x, mal_nub=0, mal_opt=0, mal_time=0, opt='Default'):   #x 받아서 텍스트에 저장
        if opt == 'Default':
            with open(file_name, 'a') as f:
                if ep_iter == 0:
                    f.write(f"{'=' * 30}\n{datetime.today().strftime('%Y/%m/%d %H:%M:%S')}\n"
                            f"  Mal: {mal_nub:5} | Size : {mal_opt:10}\tMaltime : {mal_time:10}\n"
                            f"{'=' * 30}\n")
                    f.write(f"{ep:5} | {ep_iter:5} | {x}\n")
                else:
                    f.write(f"{ep:5} | {ep_iter:5} | {x}\n")

        else:
            pass
        pass

    @staticmethod
    def log_ini(file_name):      # 파일존재 확인 및 초기화
        if os.path.isdir(file_name):
            os.remove(file_name)
        else:
            with open(file_name, 'w') as f:
                f.write(f"{'='*30}\n{datetime.today().strftime('%Y/%m/%d %H:%M:%S')}\n{'='*30}\n")

    @staticmethod
    def DBlogger(name="Default0"):
        file_nub, file_name = 0, "Default0"
        while True:
            if not os.path.isfile(file_name + '.txt'):
                logger = file_name
                break
            else:
                file_nub += 1
                file_name[-1] = f'{file_nub}'
        return logger

    @staticmethod
    def RoundVal(val, div, r):
        return round(val / div, r)

class DB:
    def __init__(self, max_leg=1):
        self.DB_logger = TOOL.DBlogger(name="DB_log0")
        self.max_leg = max_leg
        self.Phypara_list = ["ZINST58", "ZINST63", "ZVCT"]
        self.Coppara_list = ["BFV122", "BPV145"]
        self.Costom_list = ["BFV122_CONT", "BPV145_CONT"]
        self.NetPhyInput = {para: deque(maxlen=self.max_leg) for para in self.Phypara_list}
        self.NetCopInput = {para: deque(maxlen=self.max_leg) for para in self.Coppara_list}
        self.NetCtmInput = {para: deque(maxlen=self.max_leg) for para in self.Costom_list}

    def appendmem(self, mem):
        for para in self.Phypara_list:
            self.NetPhyInput[para].append(mem[para]['Val'])
        for para in self.Coppara_list:
            self.NetCopInput[para].append(mem[para]['Val'])

    def appendCtm(self, val={}):
        assert isinstance(val, dict), "Val이 Dict가 아님."
        assert len(val.keys()) != len(self.Costom_list), f"Costom_list {self.Costom_list}: " \
                                                         f"Val {val.keys()} Key 에러"
        for para in self.Costom_list:
            self.NetCtmInput[para].append(val[para])

    def to_torch(self):
        # [n, val] -> # [1, n, val]
        return torch.tensor([self.NetPhyInput]), torch.tensor([self.NetCopInput]), torch.tensor([self.NetCtmInput])

class RLMem:
    def __init__(self, net_nub, para_info):
        self.para_info = para_info
        self.net_nub = net_nub
        # 액션
        self.int_action = {_: 0 for _ in range(self.net_nub)}
        self.list_action = {_: [] for _ in range(self.net_nub)}
        self.list_action_temp = {_: [] for _ in range(self.net_nub)}

        # 수정된 액션
        self.int_mod_action = {_: 0 for _ in range(self.net_nub)}
        self.list_mod_action = {_: [] for _ in range(self.net_nub)}
        self.list_mod_action_temp = {_: [] for _ in range(self.net_nub)}

        # 액션 Proba
        self.float_porb_action = {_: 0 for _ in range(self.net_nub)}
        self.list_porb_action = {_: [] for _ in range(self.net_nub)}
        self.list_porb_action_temp = {_: [] for _ in range(self.net_nub)}

        # 액션 Proba 전체
        self.float_allporb_action = {_: 0 for _ in range(self.net_nub)}
        self.list_allporb_action = {_: [] for _ in range(self.net_nub)}

        # 상태
        self.spy_list = []
        self.spy_list_temp = []

        self.scomp_list = []
        self.scomp_list_temp = []

        # 그래프용 상태
        self.GP_List = {
            # 'Para' : []
            _: [] for _ in self.para_info
        }

        # 보상
        self.float_reward = {_: 0 for _ in range(self.net_nub)}
        self.list_reward = {_: [] for _ in range(self.net_nub)}
        self.list_reward_temp = {_: [] for _ in range(self.net_nub)}

        # 종료
        self.bool_done = {_: 0 for _ in range(self.net_nub)}
        self.list_done = {_: [] for _ in range(self.net_nub)}
        self.list_done_temp = {_: [] for _ in range(self.net_nub)}

    def SaveNetOut(self, NetNub, AllNetOut, NetOut, Act, NetType):
        # print(NetNub, AllNetOut, NetOut, Act)
        # print(NetNub, np.shape(AllNetOut), np.shape(NetOut), np.shape(Act))
        # 액션
        self.int_action[NetNub] = Act
        self.list_action[NetNub].append(Act)            # 저장용 변수
        if NetType:
            self.list_action_temp[NetNub].append([Act])     # 훈련용 변수
        else:
            self.list_action_temp[NetNub].append(Act)     # 훈련용 변수

        # 액션 확률
        self.float_porb_action[NetNub] = NetOut
        self.list_porb_action[NetNub].append(NetOut)        # 저장용 변수
        if NetType:
            self.list_porb_action_temp[NetNub].append([NetOut]) # 훈련용 변수
        else:
            self.list_porb_action_temp[NetNub].append(NetOut)  # 훈련용 변수

        # 액션 전체 확률
        self.float_allporb_action[NetNub] = AllNetOut
        self.list_allporb_action[NetNub].append(AllNetOut)  # 저장용 변수

    def SaveModNetOut(self, NetNub, Act):
        # 수정된 액션 <- 주로 실제 CNS의 제어 변수에 이용하기 위해서 사용
        self.int_mod_action[NetNub] = Act
        self.list_mod_action[NetNub].append(Act)        # 저장용 변수
        self.list_mod_action_temp[NetNub].append(Act)   # 훈련용 변수

    def SaveState(self, spy, scomp):

        nub_phy_val = np.shape(spy)[1]
        nub_comp_val = np.shape(scomp)[1]

        phys = spy[:, :, -1:].data.reshape(nub_phy_val).tolist()  # (3,)
        comps = scomp[:, :, -1:].data.reshape(nub_comp_val).tolist()  # (3,)

        self.spy_list.append(phys)           # (1, 3, 15) -list> (3, 15)
        self.spy_list_temp.append(spy.tolist()[0])      # (1, 3, 15) -list> (3, 15)

        self.scomp_list.append(comps)       # (1, 3, 15) -list> (3, 15)
        self.scomp_list_temp.append(scomp.tolist()[0])  # (1, 3, 15) -list> (3, 15)

    def SaveGPState(self, cns_mem):
        for para in self.GP_List.keys():
            self.GP_List[para].append(cns_mem[para]['Val'])

    def SaveReward(self, NetNub, Reward):
        self.float_reward[NetNub] = Reward
        self.list_reward[NetNub].append(Reward)
        self.list_reward_temp[NetNub].append(Reward)

    def SaveDone(self, NetNub, Done):
        # Done == True: 0, False: 1
        if Done:    # True
            d = 0
        else:       # False
            d = 1

        self.bool_done[NetNub] = d
        self.list_done[NetNub].append(d)
        self.list_done_temp[NetNub].append(d)

    def GetAct(self, NetNub):
        return self.int_mod_action[NetNub]

    def GetReward(self, NetNub):
        return self.float_reward[NetNub]

    def GetActProb(self, NetNub, Act):
        return self.float_allporb_action[NetNub][Act]

    def GetAllProb(self, NetNub):
        return self.float_allporb_action[NetNub]

    def GetBatch(self):
        spy_batch = torch.tensor(self.spy_list_temp, dtype=torch.float)
        scomp_batch = torch.tensor(self.scomp_list_temp, dtype=torch.float)
        return spy_batch, scomp_batch

    def GetFinBatch(self, spy, scomp):
        self.spy_list_temp.append(spy.tolist()[0])  # (1, 3, 15) -list> (3, 15)
        self.scomp_list_temp.append(scomp.tolist()[0])  # (1, 3, 15) -list> (3, 15)
        spy_batch = torch.tensor(self.spy_list_temp[1:], dtype=torch.float)
        scomp_batch = torch.tensor(self.scomp_list_temp[1:], dtype=torch.float)
        return spy_batch, scomp_batch

    def GetGPX(self):
        return self.GP_List['KCNTOMS']

    def GetGPY(self, para):
        if para not in self.GP_List.keys(): raise KeyError(f"{para}가 PARA_info_For_save 에 존재하지 않음.")
        return self.GP_List[para]

    def GetGPR(self, nubNet):
        return self.list_reward[nubNet]

    def GetGPAct(self, nubNet):
        return self.list_mod_action[nubNet]

    def GetGPAllAProb(self, nubNet):
        return self.list_allporb_action[nubNet]

    def GetGPAProb(self, nubNet):
        return self.list_porb_action[nubNet]

    def DumpAllData(self, log_path):        # TODO 아직 3개 이상의 네트워크가 구현되어 있지 않음.
        with open(log_path, 'w') as f:
            s = ''
            # title
            for NetNub in range(self.net_nub - 1):
                if np.shape(self.list_action[NetNub][0]) != ():
                    for _ in range(np.shape(self.list_action[NetNub][0])[0]):
                        s += f'Agent{NetNub}Act{_},'
                    for _ in range(np.shape(self.list_action[NetNub][0])[0]):
                        s += f'Agent{NetNub}ModAct{_},'
                    for _ in range(np.shape(self.list_action[NetNub][0])[0]):
                        s += f'Agent{NetNub}ActProb{_},'
                else:
                    s += f'Agent{NetNub}Act,'
                    s += f'Agent{NetNub}ModAct,'
                    s += f'Agent{NetNub}ActProb,'

                for actline in range(len(self.list_allporb_action[NetNub][0])):
                    s += f'Agent{NetNub}ALLAct{actline},'

                s += f'Agent{NetNub}Reward,'
                s += f'Agent{NetNub}Done,'

            for spy_item in range(len(self.spy_list[0])):
                s += f'SPY{spy_item},'
            for scomp_item in range(len(self.scomp_list[0])):
                s += f'SCOMP{scomp_item},'

            for GPKey in self.GP_List.keys():
                s += f'{GPKey},'
            s += '\n'
            # DB
            for lineNub in range(len(self.spy_list)):
                for NetNub in range(self.net_nub - 1):
                    if np.shape(self.list_action[NetNub][0]) != ():
                        for _ in range(np.shape(self.list_action[NetNub][0])[0]):
                            s += f'{self.list_action[NetNub][lineNub][_]},'
                        for _ in range(np.shape(self.list_action[NetNub][0])[0]):
                            s += f'{self.list_mod_action[NetNub][lineNub][_]},'
                        for _ in range(np.shape(self.list_action[NetNub][0])[0]):
                            s += f'{self.list_porb_action[NetNub][lineNub][_]},'
                    else:
                        s += f'{self.list_action[NetNub][lineNub]},'
                        s += f'{self.list_mod_action[NetNub][lineNub]},'
                        s += f'{self.list_porb_action[NetNub][lineNub]},'

                    for actline in range(len(self.list_allporb_action[NetNub][0])):
                        s += f'{self.list_allporb_action[NetNub][lineNub][actline]},'

                    s += f'{self.list_reward[NetNub][lineNub]},'
                    s += f'{self.list_done[NetNub][lineNub]},'

                for spy_item in range(len(self.spy_list[lineNub])):
                    s += f'{self.spy_list[lineNub][spy_item]},'
                for scomp_item in range(len(self.scomp_list[lineNub])):
                    s += f'{self.scomp_list[lineNub][scomp_item]},'

                for GPKey in self.GP_List.keys():
                    s += f'{self.GP_List[GPKey][lineNub]},'
                s += '\n'
            f.write(s)

        self.CleanEP()
        pass

    def CleanEP(self):
        # 액션
        self.int_action = {_: 0 for _ in range(self.net_nub)}
        self.list_action = {_: [] for _ in range(self.net_nub)}
        self.list_action_temp = {_: [] for _ in range(self.net_nub)}

        # 수정된 액션
        self.int_mod_action = {_: 0 for _ in range(self.net_nub)}
        self.list_mod_action = {_: [] for _ in range(self.net_nub)}
        self.list_mod_action_temp = {_: [] for _ in range(self.net_nub)}

        # 액션 Proba
        self.float_porb_action = {_: 0 for _ in range(self.net_nub)}
        self.list_porb_action = {_: [] for _ in range(self.net_nub)}
        self.list_porb_action_temp = {_: [] for _ in range(self.net_nub)}

        # 액션 Proba 전체
        self.float_allporb_action = {_: 0 for _ in range(self.net_nub)}
        self.list_allporb_action = {_: [] for _ in range(self.net_nub)}

        # 상태
        self.spy_list = []
        self.spy_list_temp = []

        self.scomp_list = []
        self.scomp_list_temp = []

        # 그래프용 상태
        self.GP_List = {
            # 'Para' : []
            _: [] for _ in self.para_info
        }

        # 보상
        self.float_reward = {_: 0 for _ in range(self.net_nub)}
        self.list_reward = {_: [] for _ in range(self.net_nub)}
        self.list_reward_temp = {_: [] for _ in range(self.net_nub)}

        # 종료
        self.bool_done = {_: 0 for _ in range(self.net_nub)}
        self.list_done = {_: [] for _ in range(self.net_nub)}
        self.list_done_temp = {_: [] for _ in range(self.net_nub)}

    def CleanTrainMem(self):
        # 액션
        self.list_action_temp = {_: [] for _ in range(self.net_nub)}

        # 수정된 액션
        self.list_mod_action_temp = {_: [] for _ in range(self.net_nub)}

        # 액션 Proba
        self.list_porb_action_temp = {_: [] for _ in range(self.net_nub)}

        # 상태
        self.spy_list_temp = []

        self.scomp_list_temp = []

        # 보상
        self.list_reward_temp = {_: [] for _ in range(self.net_nub)}

        # 종료
        self.list_done_temp = {_: [] for _ in range(self.net_nub)}

class PTCureve:
    """
        0 : 만족, 1: 불만족
        PTCureve().Check(Temp=110, Pres=0)
    """
    def __init__(self):
        self.UpTemp = [0, 37.000000, 65.500000, 93.000000, 104.400000, 110.000000,
                       115.500000, 121.000000, 148.800000, 176.500000, 186.500000, 350.0]
        self.UpPres = [29.5, 29.500000, 30.500000, 36.500000, 42.000000, 45.600000,
                       49.000000, 54.200000, 105.000000, 176.000000, 200.000000, 592]
        self.BotTemp = [0, 37.000000, 149.000000, 159.000000, 169.000000, 179.000000,
                        204.000000, 232.000000, 260.000000, 287.700000, 350.000000]
        self.BotPres = [17.0, 17.000000, 17.000000, 17.300000, 17.600000, 20.000000,
                        31.600000, 44.300000, 58.000000, 71.000000, 100.000000]
        self.UpLineFunc = []
        self.BotLineFunc = []

        self._make_bound_UpLine()
        self._make_bound_BotLine()

    def _make_bound_func(self, Temp, Pres):
        """
        2점에 대한 1차원 함수 반환
        :param Temp: [a1, a2] == x
        :param Pres: [b1, b2] == y
        :return: func
        """
        # y1 = ax1 + b
        # y2 = ax2 + b
        # a = (y1-y2)/(x1-x2)
        # b = y1 - {(y1-y2)/(x1-x2) * x1}
        get_a = (Pres[0] - Pres[1]) / (Temp[0] - Temp[1])
        get_b = Pres[0] - get_a * Temp[0]
        return lambda temp: get_a * temp + get_b

    def _make_bound_UpLine(self):
        for i in range(len(self.UpTemp) - 1):
            self.UpLineFunc.append(self._make_bound_func(Temp=self.UpTemp[i:i+2], Pres=self.UpPres[i:i+2]))

    def _make_bound_BotLine(self):
        for i in range(len(self.BotTemp) - 1):
            self.BotLineFunc.append(self._make_bound_func(Temp=self.BotTemp[i:i+2], Pres=self.BotPres[i:i+2]))

    def _call_fun(self, Temp):
        UpF, BotF = 0, 0
        for i in range(len(self.UpTemp) - 1):
            if self.UpTemp[i] <= Temp < self.UpTemp[i + 1]:
                UpF = self.UpLineFunc[i]
        for i in range(len(self.BotTemp) - 1):
            if self.BotTemp[i] <= Temp < self.BotTemp[i + 1]:
                BotF = self.BotLineFunc[i]
        return UpF, BotF

    def _check_up_or_under(self, fun, Temp, Pres):
        Get_Pres = fun(Temp)
        if Get_Pres > Pres:
            return 0    # 입력된 Pres가 그래프보다 아래쪽에 존재
        elif Get_Pres == Pres:
            return 1    # 입력된 Pres가 그래프에 존재
        else:
            return 2    # 입력된 Pres가 그래프보다 위쪽에 존재

    def _check_in_or_out(self, Temp, Pres):
        UpF, BotF = self._call_fun(Temp=Temp)
        Upcond = self._check_up_or_under(UpF, Temp, Pres)
        Botcond = self._check_up_or_under(BotF, Temp, Pres)

        if Upcond == 2 or Botcond == 0:
            return 1    # PT커브 초과
        else:
            return 0    # PT커브에서 운전 중

    def Check(self, Temp, Pres):
        """
        PT curve에 운전 중인지 확인
        :param Temp: 현재 온도
        :param Pres: 현재 압력
        :return: 0 만족, 1 불만족
       """
        return self._check_in_or_out(Temp, Pres)

class CSFTree:
    @staticmethod
    def CSF1(TRIP, PR, IR, SR):
        """
        미임계 상태 추적도 만족 불만족
        :param TRIP: Trip 1: Trip 0: Operation
        :param PR: Power Range [100 ~ 0]
        :param IR: Intermediate Range [-3 ~ .. ]
        :param SR: Source Range [0.0 ~ ..]
        :return: {'L': 0 만족, 1: 노랑, 2: 주황, 3: 빨강, 'N': 탈출 단계, 'P': 절차서}
        """
        if TRIP == 1:
            if not PR < 5: # 5%
                return {'L': 3, 'N': 0, 'P': 'S1'}              # GOTO 회복 S.1
            else:
                if IR <= 0:
                    if IR < 1E-9:
                        if SR <= 0:
                            return {'L': 0, 'N': 1, 'P': 'Ok'}  # OK!
                        else:
                            return {'L': 1, 'N': 2, 'P': 'S2'}  # GOTO 회복 S.2
                    else:
                        if IR < -0.2:
                            return {'L': 0, 'N': 3, 'P': 'Ok'}  # OK!
                        else:
                            return {'L': 1, 'N': 4, 'P': 'S2'}  # GOTO 회복 S.2
                else:
                    return {'L': 2, 'N': 5, 'P': 'S1'}          # GOTO 회복 S.1
        else:
            return {'L': 0, 'N': 6, 'P': 'Ok'}                  # Ok!

    @staticmethod
    def CSF2(TRIP, CET, PT):
        """
        노심냉각 상태 추적도
        :param TRIP: Trip 1: Trip 0: Operation
        :param CET: CoreExitTemp [ .. ~ 326 ]
        :param PT: PTCurve [ 0 만족, 1 불만족 ]
        :return: {'L': 0 만족, 1: 노랑, 2: 주황, 3: 빨강, 'N': 탈출 단계, 'P': 절차서}
        """
        if TRIP == 1:
            if CET < 649:
                if PT == 0:
                    return {'L': 0, 'N': 0, 'P': 'Ok'}            # OK!
                else:
                    if CET < 371:
                        return {'L': 1, 'N': 1, 'P': 'C3'}        # GOTO 회복 C.3
                    else:
                        return {'L': 2, 'N': 2, 'P': 'C2'}        # GOTO 회복 C.2
            else:
                return {'L': 3, 'N': 3, 'P': 'C1'}                # GOTO 회복 C.1
        else:
            return {'L': 0, 'N': 4, 'P': 'Ok'}                    # Ok!

    @staticmethod
    def CSF3(TRIP, SG1N, SG2N, SG3N, SG1P, SG2P, SG3P, SG1F, SG2F, SG3F):
        """
        열제거원 상태 추적도
        :param TRIP: Trip 1: Trip 0: Operation
        :param SG1N: SG 1 Narrow Level [0 ~ 50]
        :param SG2N: SG 2 Narrow Level [0 ~ 50]
        :param SG3N: SG 3 Narrow Level [0 ~ 50]
        :param SG1P: SG 1 Pressrue [ 0 ~ 100 ]
        :param SG2P: SG 2 Pressrue [ 0 ~ 100 ]
        :param SG3P: SG 3 Pressrue [ 0 ~ 100 ]
        :param SG1F: SG 1 Feedwater [ 0 ~ 25 ] in emergency
        :param SG2F: SG 2 Feedwater [ 0 ~ 25 ] in emergency
        :param SG3F: SG 3 Feedwater [ 0 ~ 25 ] in emergency
        :return: {'L': 0 만족, 1: 노랑, 2: 주황, 3: 빨강, 'N': 탈출 단계, 'P': 절차서}
        """
        if TRIP == 1:
            if SG1N >= 6 or SG2N >= 6 or SG3N >= 6:
                pass
            else:
                if SG1F + SG2F + SG3F >= 33:
                    pass
                else:
                    return {'L': 3, 'N': 1, 'P': 'H1'}            # GOTO 회복 H.1
            # --
            if not SG1P < 88.6 and not SG2P < 88.6 and not SG3P < 88.6:
                return {'L': 1, 'N': 2, 'P': 'H2'}                # GOTO 회복 H.2
            else:
                if not SG1N < 78 and not SG2N < 78 and not SG3N < 78:
                    return {'L': 1, 'N': 3, 'P': 'H3'}            # GOTO 회복 H.3
                else:
                    if not SG1P < 83.3 and not SG2P < 83.3 and not SG3P < 83.3:
                        return {'L': 1, 'N': 4, 'P': 'H4'}        # GOTO 회복 H.4
                    else:
                        if not SG1N > 6 and not SG2N > 6 and not SG3N > 6:
                            return {'L': 1, 'N': 5, 'P': 'H5'}    # GOTO 회복 H.5
                        else:
                            return {'L': 0, 'N': 6, 'P': 'Ok'}    # OK!
        else:
            return {'L': 0, 'N': 7, 'P': 'Ok'}                    # Ok!

    @staticmethod
    def CSF4(TRIP, RC1, RC2, RC3, RP, PT, TIME):
        """
        RCS 건전성 상태 추적도
        :param TRIP: Trip 1: Trip 0: Operation
        :param RC1: RCS Cool LOOP 1 [List] [270 ..]
        :param RC2: RCS Cool LOOP 2 [List] [270 ..]
        :param RC3: RCS Cool LOOP 3 [List] [270 ..]
        :param RP: RCS pressure [160 ~ ..]
        :param PT: PTCurve [ 0 만족, 1 불만족 ]
        :param TIME: CNS TIME [5 tick ~ ..]
        :return: {'L': 0 만족, 1: 노랑, 2: 주황, 3: 빨강, 'N': 탈출 단계, 'P': 절차서}
        """
        if TRIP == 1:
            RC1AVG = sum(list(RC1)[:-1]) / len(list(RC1)[:-1])
            RC2AVG = sum(list(RC2)[:-1]) / len(list(RC2)[:-1])
            RC3AVG = sum(list(RC3)[:-1]) / len(list(RC3)[:-1])

            if not RC1[-1] < RC1AVG and not RC2[-1] < RC2AVG and not RC3[-1] < RC3AVG:
                if not PT == 0:
                    return {'L': 3, 'N': 0, 'P': 'P1'}            # GOTO 회복 P.1
                else:
                    if not RC1[-1] > 106 and not RC2[-1] > 106 and not RC3[-1] > 106:
                        return {'L': 2, 'N': 1, 'P': 'P1'}        # GOTO 회복 P.1
                    else:
                        if not RC1[-1] > 136 and not RC2[-1] > 136 and not RC3[-1] > 136:
                            return {'L': 1, 'N': 2, 'P': 'P2'}    # GOTO 회복 P.2
                        else:
                            return {'L': 0, 'N': 3, 'P': 'Ok'}    # Ok!
            else:
                if not RC1[-1] > 177 and not RC2[-1] > 177 and not RC3[-1] > 177:
                    if not PT == 0:
                        if not RC1[-1] > 106 and not RC2[-1] > 106 and not RC3[-1] > 106:
                            return {'L': 2, 'N': 4, 'P': 'P1'}    # GOTO 회복 P.1
                        else:
                            return {'L': 1, 'N': 5, 'P': 'P2'}    # GOTO 회복 P.2
                    else:
                        return {'L': 0, 'N': 6, 'P': 'Ok'}        # Ok!
                else:
                    return {'L': 0, 'N': 7, 'P': 'Ok'}            # Ok!
        else:
            return {'L': 0, 'N': 8, 'P': 'Ok'}                    # Ok!

    @staticmethod
    def CSF5(TRIP, CTP, CTS, CTR):
        """
        격납용기 건전상 상태 추적도
        :param TRIP: Trip 1: Trip 0: Operation
        :param CTP: CTMTPressre     [... ~ 0.2]
        :param CTS: CTMTSumpLevel   [0 ~ ... ]
        :param CTR: CTMTRad         [2.0 ~ ... ]
        :return: {'L': 0 만족, 1: 노랑, 2: 주황, 3: 빨강, 'N': 탈출 단계, 'P': 절차서}
        """
        if TRIP == 1:
            if not CTP < 4.2:
                    return {'L': 3, 'N': 0, 'P': 'Z1'}            # GOTO 회복 Z.1
            else:
                if not CTP < 1.55:
                    return {'L': 2, 'N': 1, 'P': 'Z1'}            # GOTO 회복 Z.1
                else:
                    if not CTS < 0.345:
                        return {'L': 2, 'N': 2, 'P': 'Z2'}        # GOTO 회복 Z.2
                    else:
                        if not CTR < 1E4:
                            return {'L': 1, 'N': 3, 'P': 'Z3'}    # GOTO 회복 Z.3
                        else:
                            return {'L': 0, 'N': 4, 'P': 'Ok'}    # Ok!
        else:
            return {'L': 0, 'N': 5, 'P': 'Ok'}                    # Ok!

    @staticmethod
    def CSF6(TRIP, PZRL):
        """
        RCS 재고량 상태 추적도
        :param TRIP: Trip 1: Trip 0: Operation
        :param PZRL: PZRLevel
        :return: {'L': 0 만족, 1: 노랑, 2: 주황, 3: 빨강, 'N': 탈출 단계, 'P': 절차서}
        """
        if TRIP == 1:
            if not PZRL < 92:
                return {'L': 1, 'N': 0, 'P': 'I1'}            # GOTO 회복 I.1
            else:
                if not PZRL > 17:
                    return {'L': 1, 'N': 1, 'P': 'I2'}        # GOTO 회복 I.2
                else:
                    if not 17 <= PZRL <= 92:
                        return {'L': 1, 'N': 2, 'P': 'I2'}    # GOTO 회복 I.2
                    else:
                        return {'L': 0, 'N': 3, 'P': 'Ok'}    # Ok!
        else:
            return {'L': 0, 'N': 4, 'P': 'Ok'}                # Ok.
