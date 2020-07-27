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
    def log_add(file_name, ep, ep_iter, x, opt='Default'):   #x 받아서 텍스트에 저장
        if opt == 'Default':
            with open(file_name, 'a') as f:
                if ep_iter == 0:
                    f.write(f"{'=' * 30}\n{datetime.today().strftime('%Y/%m/%d %H:%M:%S')}\n{'=' * 30}\n")
                    f.write(f"{ep:5} | {ep_iter:5} | {x}\n")
                else:
                    f.write(f"{ep:5} | {ep_iter:5} | {x}\n")
        elif opt == 'Malinfo':
            with open(file_name, 'a') as f:
                f.write(f"\t{ep:5} | Size : {x[0]:10}\tMaltime : {x[1]:10}\n")
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
