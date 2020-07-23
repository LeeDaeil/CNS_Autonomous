import numpy as np


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