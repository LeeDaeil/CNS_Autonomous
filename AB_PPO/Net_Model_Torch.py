import torch
from torch import nn, optim, tensor


class PPOModel(nn.Module):
    def __init__(self, nub_para, time_leg):
        super(PPOModel, self).__init__()
        self.val = 0
        if True:
            # ==============================================================================================================
            # +1 Tick 예측 모델
            # ==============================================================================================================
            # Physical
            self.Conv1 = nn.Conv1d(in_channels=nub_para, out_channels=nub_para,
                                   kernel_size=3, stride=1)
            self.Conv2 = nn.Conv1d(in_channels=nub_para, out_channels=nub_para,
                                   kernel_size=3, stride=1)
            # Component
            self.Conv3 = nn.Conv1d(in_channels=nub_para, out_channels=nub_para,
                                   kernel_size=3, stride=1)
            self.Conv4 = nn.Conv1d(in_channels=nub_para, out_channels=nub_para,
                                   kernel_size=3, stride=1)
            # Tick Model Output
            self.LSTM1 = nn.LSTM(16, 16, batch_first=True) # train 할때 batch 간 영향 미 고려
            self.FC1 = nn.Linear(32, 2)
            self.FC2 = nn.Linear(32, 1)
        if True:
            # ==============================================================================================================
            # Comp 제어 모델
            # ==============================================================================================================
            # Physical
            self.Conv5 = nn.Conv1d(in_channels=nub_para, out_channels=nub_para,
                                   kernel_size=3, stride=1)
            self.Conv6 = nn.Conv1d(in_channels=nub_para, out_channels=nub_para,
                                   kernel_size=3, stride=1)
            # Component
            self.Conv7 = nn.Conv1d(in_channels=nub_para, out_channels=nub_para,
                                   kernel_size=3, stride=1)
            self.Conv8 = nn.Conv1d(in_channels=nub_para, out_channels=nub_para,
                                   kernel_size=3, stride=1)
            self.MAX_FOOL = nn.MaxPool1d(2, stride=1)

            # Tick Model Output
            self.LSTM2 = nn.LSTM(13, 32, batch_first=True)  # train 할때 batch 간 영향 미 고려
            self.FC3_1 = nn.Linear(64, 3)
            self.FC3_2 = nn.Linear(64, 3)
            self.FC4 = nn.Linear(64, 1)

    def CommonPredictNet(self, x_py, x_comp):
        # Physical
        # x_py = nn.functional.relu(self.Conv1(x_py))
        x_py = nn.functional.relu6(self.Conv2(x_py))
        # print("A", x_py)
        # x_py = self.MAX_FOOL(x_py)
        # print("A", x_py)
        # Component
        # x_comp = nn.functional.relu(self.Conv3(x_comp))
        x_comp = nn.functional.relu(self.Conv4(x_comp))
        # Concat
        x = torch.cat([x_py, x_comp], dim=2)
        # Output
        # x = self.LSTM1(x)[0]  # Last value
        x = x.reshape(x.shape[0], 32)   # batch 사이즈에 따라서 자동적으로 조정
        # x = nn.functional.softsign(x)
        return x

    def GetPredictActorOut(self, x_py, x_comp):
        x = self.CommonPredictNet(x_py, x_comp)
        x = nn.functional.leaky_relu(self.FC1(x))#,negative_slope=0.01)
        # 출력 변수 Clamp
        # ZINST58 0.2 ~ 0   /   156.21      [kg/cm^2    ][0]
        # ZINST63 0.1 ~ 0   /   55.13       [%          ][1]
        # upper_clamp = torch.tensor([0.2, 0.1])
        # lower_clamp = torch.tensor([0.0, 0.0])
        # x = torch.max(torch.min(x, upper_clamp), lower_clamp)
        return x

    def GetPredictCrticOut(self, x_py, x_comp):
        x = self.CommonPredictNet(x_py, x_comp)
        x = nn.functional.leaky_relu(self.FC2(x))  # ,negative_slope=0.01)
        return x

    def CommonControlNet(self, x_py, x_comp):
        # Get t+1
        Predicted_Val = self.GetPredictActorOut(x_py, x_comp)   # Get t+1 tick val
        Predicted_Val = Predicted_Val.view(Predicted_Val.shape[0], 2, 1).detach() # batch 사이즈에 따라서 자동적으로 조정
        x_py = torch.cat([x_py, Predicted_Val], dim=2)
        # Physical
        x_py = nn.functional.relu(self.Conv5(x_py))
        x_py = nn.functional.relu(self.Conv6(x_py))
        # Component
        x_comp = nn.functional.relu(self.Conv7(x_comp))
        x_comp = nn.functional.relu(self.Conv8(x_comp))
        # Concat
        x = torch.cat([x_py, x_comp], dim=2)
        # Output
        x = self.LSTM2(x)[0]  # Last value
        x = x.reshape(x.shape[0], 64)   # batch 사이즈에 따라서 자동적으로 조정
        x = nn.functional.softsign(x)
        return x

    def GetControlActorOut(self, x_py, x_comp, softmax_dim=1):
        x = self.CommonControlNet(x_py, x_comp)
        comp_1_x = nn.functional.relu(self.FC3_1(x))
        comp_2_x = nn.functional.relu(self.FC3_2(x))
        comp_1_x = nn.functional.softmax(comp_1_x, dim=softmax_dim)
        comp_2_x = nn.functional.softmax(comp_2_x, dim=softmax_dim)
        return comp_1_x, comp_2_x

    def GetControlCrticOut(self, x_py, x_comp):
        x = self.CommonControlNet(x_py, x_comp)
        x = self.FC4(x)
        return x


if __name__ == '__main__':

    Test_db = tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=torch.float)
    Test_db = Test_db.unsqueeze(0).unsqueeze(0)
    Conv1 = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=3, stride=1)
    print(Conv1(Test_db))

    Model = PPOModel(nub_para=2, time_leg=10)

    Test_db = tensor([[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                       [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]], dtype=torch.float)
    print(Test_db.size()) # 2, 2, 10

    print(Model.GetPredictActorOut(x_py=Test_db/10, x_comp=Test_db/10))
    print(Model.GetPredictActorOut(x_py=Test_db/2, x_comp=Test_db/3))

    out = Model.GetPredictActorOut(x_py=Test_db/2, x_comp=Test_db/3).tolist()
    print(out)
    print(Model.GetPredictCrticOut(x_py=Test_db/2, x_comp=Test_db/3))
    print('='*100)
    print(Model.GetControlActorOut(x_py=Test_db/2, x_comp=Test_db/3))
    Test_db = tensor([[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                       [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]],
                      [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                       [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]
                      ], dtype=torch.float)

    print(Test_db.size()) # 2, 2, 10

    print(Model.GetPredictActorOut(x_py=Test_db/2, x_comp=Test_db/3))
    print(Model.GetPredictCrticOut(x_py=Test_db/2, x_comp=Test_db/3))
    print(Model.GetControlActorOut(x_py=Test_db/2, x_comp=Test_db/3))
