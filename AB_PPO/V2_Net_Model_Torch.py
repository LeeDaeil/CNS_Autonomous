import torch
from torch import nn, optim, tensor
from COMMONTOOL import TOOL
from numpy.random import random

LSTMMODE = False

class NETBOX:
    """
    여러개의 네트워크를 호출하여 딕셔너리로 보관
    """
    def __init__(self):
        self.NET = {
            0: PPOModel(name="TEST", NubPhyPara=2, NubComPara=2, NubTimeSeq=10, ClipNetOut=[-1.0, 1.0]),
            1: PPOModel(name="TEST", NubPhyPara=2, NubComPara=2, NubTimeSeq=10, ClipNetOut=[-1.0, 1.0])
        }
        self.NubNET = len(self.NET)


class NETOPTBOX:
    """
    위의 NETBOX의 NET수에 맞게 OPT 생성
    """
    def __init__(self, NubNET, NET):
        """
        :param NubNET: NETBOX의 NubNet
        :param NETBOXinNET: NETBOX의 NET
        """
        self.learning_rate = 0.0002
        self.NETOPT = {_: optim.Adam(NET[_].parameters(), lr=self.learning_rate) for _ in range(NubNET)}


class PPOModel(nn.Module):
    """
    Fun: 1개 네트워크 모델 구성하여 반환
    """
    def __init__(self, name, NubPhyPara, NubComPara, NubTimeSeq, ClipNetOut):
        # 상속
        super(PPOModel, self).__init__()
        # 입력 정보
        self.ModelName = name           # str
        self.NubPhyPara = NubPhyPara    # int
        self.NubComPara = NubComPara    # int
        self.NubTimeSeq = NubTimeSeq    # int
        self.ClipNetOut = ClipNetOut    # [Min float, Max float]
        # 모델 정보

        # Physical
        self.PhyConV1 = nn.Conv1d(in_channels=self.NubPhyPara, out_channels=self.NubPhyPara,
                                  kernel_size=3, stride=1)
        # Component
        self.ComConV1 = nn.Conv1d(in_channels=self.NubComPara, out_channels=self.NubComPara,
                                  kernel_size=3, stride=1)
        # Model Out
        # x,10 -> 10-3+1 -> x,8 -> max(3, 1) 8-3+1 -> x,6 --> xz,6>6,zx --> LSTM(zx, zx*6) -->
        # z,10 -> 10-3+1 -> z,8 -> max(3, 1) 8-3+1 -> z,6 -|
        self.LSTM2 = nn.LSTM(self.NubComPara + self.NubPhyPara, 30, batch_first=True)
        if LSTMMODE:
            self.FC2_A = nn.Linear(30, 1)
            self.FC2_C = nn.Linear(30, 1)
        else:
            # self.FC2_A = nn.Linear(24, 1)
            # self.FC2_C = nn.Linear(24, 1)
            self.FC2_A = nn.Linear(32, 1)
            self.FC2_C = nn.Linear(32, 1)

    def _CommonPredictNet(self, x_py, x_comp):
        # Physical
        x_py = nn.functional.relu(self.PhyConV1(x_py))
        # x_py = nn.functional.max_pool1d(x_py, 3, 1)
        # TOOL.ALLP(x_py, comt='x_py')
        # Component
        x_comp = nn.functional.relu(self.ComConV1(x_comp))
        # x_comp = nn.functional.max_pool1d(x_comp, 3, 1)
        # TOOL.ALLP(x_comp, comt='x_comp')
        # Concat
        x = torch.cat([x_py, x_comp], dim=1)
        # TOOL.ALLP(x, comt='x_cat')
        if LSTMMODE:
            x = torch.transpose(x, 1, 2)
            # TOOL.ALLP(x, comt='x_tranpose')
            x, hid = self.LSTM2(x)
            # TOOL.ALLP(x, comt='x_LSTM')
            x = x[:, -1, :]  # Many to one Out
            # TOOL.ALLP(x, comt='x_te')
        else:
            # x = x.reshape(x.shape[0], 24)
            x = x.reshape(x.shape[0], 32)
        # TOOL.ALLP(x, comt='x_te')
        return x

    def GetPredictActorOut(self, x_py, x_comp):
        x = self._CommonPredictNet(x_py, x_comp)
        # x = nn.functional.hardtanh(self.FC2_A(x), self.ClipNetOut[0], self.ClipNetOut[1])
        # x = nn.functional.relu(self.FC2_A(x))
        # x = self.FC2_A(x)
        x = nn.functional.leaky_relu(self.FC2_A(x))
        # TOOL.ALLP(x, comt='x_Act')
        return x

    def GetPredictCrticOut(self, x_py, x_comp):
        x = self._CommonPredictNet(x_py, x_comp)
        # x = self.FC2_C(x)  # Activation 특성상 음수 ~ 양수 보상 제공
        # x = nn.functional.relu(self.FC2_C(x))  # Activation 특성상 음수 ~ 양수 보상 제공
        x = nn.functional.leaky_relu(self.FC2_C(x))
        # TOOL.ALLP(x, comt='x_Cri')
        return x

    def TestOut(self, batchtest=False):
        if not batchtest:
            print('Network Out Test Real-time')
            Batch = 1
        else:
            print('Network Out Test Batch')
            Batch = 3

        PhyTemp = tensor([[list(random(size=self.NubTimeSeq)) for _ in range(self.NubPhyPara)] for _ in range(Batch)],
                         dtype=torch.float)
        TOOL.ALLP(PhyTemp)
        CompTemp = tensor([[list(random(size=self.NubTimeSeq)) for _ in range(self.NubComPara)] for _ in range(Batch)],
                          dtype=torch.float)
        TOOL.ALLP(CompTemp)

        self.GetPredictActorOut(x_py=PhyTemp, x_comp=CompTemp)
        self.GetPredictCrticOut(x_py=PhyTemp, x_comp=CompTemp)


if __name__ == '__main__':
    TESTMODEL = PPOModel(name="TEST", NubPhyPara=2, NubComPara=2, NubTimeSeq=10, ClipNetOut=[-0.2, 0.2])
    TESTMODEL.TestOut(batchtest=False)
    TESTMODEL.TestOut(batchtest=True)
