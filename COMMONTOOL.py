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

