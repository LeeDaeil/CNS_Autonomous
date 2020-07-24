import numpy as np
import os
from datetime import datetime

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