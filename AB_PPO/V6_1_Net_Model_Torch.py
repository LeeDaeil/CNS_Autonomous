import torch
from torch import nn, optim, tensor
from AB_PPO.COMMONTOOL import TOOL
from numpy.random import random
import random as ra

LSTMMODE = False

class NETBOX:
    """
    여러개의 네트워크를 호출하여 딕셔너리로 보관
    """
    def __init__(self):
        self.NET = {
            0: PPOModel(name="TEST", NubPhyPara=3, NubComPara=2, NubTimeSeq=15, ClipNetOut=[-1.0, 1.0]),
            1: PPOModel(name="VCTLevel", NubPhyPara=3, NubComPara=2, NubTimeSeq=15, ClipNetOut=[-1.0, 1.0], ActCase=200),
            2: PPOModel(name="PZRLevel", NubPhyPara=3, NubComPara=2, NubTimeSeq=15, ClipNetOut=[-1.0, 1.0], ActCase=200),
            3: PPOModel(name="PZRpressure", NubPhyPara=3, NubComPara=2, NubTimeSeq=15, ClipNetOut=[-1.0, 1.0], ActCase=200),
            4: PPOModel(name="BFV122", NubPhyPara=3, NubComPara=2, NubTimeSeq=15, ClipNetOut=[-1.0, 1.0], ActCase=200),
            5: PPOModel(name="BPV145", NubPhyPara=3, NubComPara=2, NubTimeSeq=15, ClipNetOut=[-1.0, 1.0], ActCase=200),
            # Control
            6: PPOModel(name="BFV122", NubPhyPara=3, NubComPara=2, NubTimeSeq=15, ClipNetOut=[-1.0, 1.0], ActCase=3),
            7: PPOModel(name="BPV145", NubPhyPara=3, NubComPara=2, NubTimeSeq=15, ClipNetOut=[-1.0, 1.0], ActCase=3),

            # CNN
            # 6: CNNModel(name="Phy", NubPhyPara=3, NubComPara=2, NubTimeSeq=15, ClipNetOut=[-1.0, 1.0], ActCase=3),
            # 7: CNNModel(name="Comp", NubPhyPara=3, NubComPara=2, NubTimeSeq=15, ClipNetOut=[-1.0, 1.0], ActCase=2),
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
    def __init__(self, name, NubPhyPara, NubComPara, NubTimeSeq, ClipNetOut, ActCase=2):
        # 상속
        super(PPOModel, self).__init__()
        # 입력 정보
        self.ModelName = name           # str
        self.NubPhyPara = NubPhyPara    # int
        self.NubComPara = NubComPara    # int
        self.NubTimeSeq = NubTimeSeq    # int
        self.ClipNetOut = ClipNetOut    # [Min float, Max float]
        self.ActCase = ActCase
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
            # self.FC2_A = nn.Linear(8*(self.NubPhyPara + self.NubComPara), self.ActCase)    # default 2 out
            self.FC2_A = nn.Linear(65, self.ActCase)    # default 2 out
            self.FC2_C = nn.Linear(65, 1)

    def _CommonPredictNet(self, x_py, x_comp):
        # print(self.ModelName)
        # Physical
        # TOOL.ALLP(x_py, comt='x_py')
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
            # x = x.reshape(x.shape[0], 8*(self.NubPhyPara + self.NubComPara))
            x = x.reshape(x.shape[0], 65)
        # TOOL.ALLP(x, comt='x_te')
        return x

    def GetPredictActorOut(self, x_py, x_comp):
        x = self._CommonPredictNet(x_py, x_comp)
        # x = nn.functional.hardtanh(self.FC2_A(x), self.ClipNetOut[0], self.ClipNetOut[1])
        # x = nn.functional.relu(self.FC2_A(x))
        # x = self.FC2_A(x)
        # x = nn.functional.leaky_relu(self.FC2_A(x))
        # TOOL.ALLP(x, comt='x_Act Before')
        if self.ModelName == "Progno1":
            x = nn.functional.hardtanh(self.FC2_A(x), -1, 1)
            x = x * 100
            # TOOL.ALLP(x, comt='x_Act_before round')
            x = x.round()
            x = x / 100
            # TOOL.ALLP(x, comt='x_Act_after round')
        else:
            x = nn.functional.softmax(self.FC2_A(x), dim=1)
        # x = nn.functional.log_softmax(self.FC2_A(x), dim=1)
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

        # print(ra.randint(0, 10))


class CNNModel(nn.Module):
    """
    Fun: 1개 네트워크 모델 구성하여 반환
    """
    def __init__(self, name, NubPhyPara, NubComPara, NubTimeSeq, ClipNetOut, ActCase=2):
        # 상속
        super(CNNModel, self).__init__()
        # 입력 정보
        self.ModelName = name           # str
        self.NubPhyPara = NubPhyPara    # int
        self.NubComPara = NubComPara    # int
        self.NubTimeSeq = NubTimeSeq    # int
        self.ClipNetOut = ClipNetOut    # [Min float, Max float]
        self.ActCase = ActCase
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
            # self.FC2_A = nn.Linear(8*(self.NubPhyPara + self.NubComPara), self.ActCase)    # default 2 out
            self.FC2_A = nn.Linear(91, self.ActCase)    # default 2 out
            # self.FC2_C = nn.Linear(91, 1)

    def _CommonPredictNet(self, x_py, x_comp):
        # print(self.ModelName)
        # Physical
        # TOOL.ALLP(x_py, comt='x_py')
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
            # x = x.reshape(x.shape[0], 8*(self.NubPhyPara + self.NubComPara))
            x = x.reshape(x.shape[0], 91)
        # TOOL.ALLP(x, comt='x_te')
        return x

    def GetPredictActorOut(self, x_py, x_comp):
        x = self._CommonPredictNet(x_py, x_comp)
        # x = nn.functional.hardtanh(self.FC2_A(x), self.ClipNetOut[0], self.ClipNetOut[1])
        # x = nn.functional.relu(self.FC2_A(x))
        # x = self.FC2_A(x)
        # x = nn.functional.leaky_relu(self.FC2_A(x))
        # TOOL.ALLP(x, comt='x_Act Before')
        if self.ModelName == "Progno1":
            x = nn.functional.hardtanh(self.FC2_A(x), -1, 1)
            x = x * 100
            # TOOL.ALLP(x, comt='x_Act_before round')
            x = x.round()
            x = x / 100
            # TOOL.ALLP(x, comt='x_Act_after round')
        else:
            # x = nn.functional.softmax(self.FC2_A(x), dim=1)
            x = nn.functional.relu6(self.FC2_A(x))
        # x = nn.functional.log_softmax(self.FC2_A(x), dim=1)
        # TOOL.ALLP(x, comt='x_Act')
        return x

    # def GetPredictCrticOut(self, x_py, x_comp):
    #     x = self._CommonPredictNet(x_py, x_comp)
    #     # x = self.FC2_C(x)  # Activation 특성상 음수 ~ 양수 보상 제공
    #     # x = nn.functional.relu(self.FC2_C(x))  # Activation 특성상 음수 ~ 양수 보상 제공
    #     x = nn.functional.leaky_relu(self.FC2_C(x))
    #     # TOOL.ALLP(x, comt='x_Cri')
    #     return x

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
        # self.GetPredictCrticOut(x_py=PhyTemp, x_comp=CompTemp)

        # print(ra.randint(0, 10))


if __name__ == '__main__':
    for net_name in ["TEST", "Progno1"]:
        TESTMODEL = PPOModel(name=net_name, NubPhyPara=3, NubComPara=4, NubTimeSeq=15, ClipNetOut=[-0.2, 0.2], ActCase=1)
        TESTMODEL.TestOut(batchtest=False)
        TESTMODEL.TestOut(batchtest=True)

        TESTMODEL = CNNModel(name=net_name, NubPhyPara=3, NubComPara=4, NubTimeSeq=15, ClipNetOut=[-0.2, 0.2],
                             ActCase=1)
        TESTMODEL.TestOut(batchtest=False)
        TESTMODEL.TestOut(batchtest=True)
