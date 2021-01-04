"""
Builder: Daeil Lee 2021-01-02

Ref-Code:
    - https://github.com/ku2482/sac-discrete.pytorch
    -
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


class BaseNetwork(nn.Module):
    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))


class ActorNet(BaseNetwork):
    def __init__(self, nub_a=3, nub_s=2, net_type='DNN'):
        super(ActorNet, self).__init__()
        self.net_type = net_type

        if self.net_type == 'DNN':
            self.L1 = nn.Linear(nub_s, 256)
            self.L2 = nn.Linear(256, 512)
            self.L3 = nn.Linear(512, 512)
            self.L_a_prob = nn.Linear(512, nub_a)

        if self.net_type == 'LSTM':
            self.L1 = nn.LSTM(nub_s, 256, 1)
            self.L2 = nn.Linear(256, 512)
            self.L3 = nn.Linear(512, 512)
            self.L_a_prob = nn.Linear(512, nub_a)

        if self.net_type == 'C_LSTM':
            self.L1 = nn.Conv1d(in_channels=nub_s, out_channels=256, kernel_size=2)
            self.L1_av = nn.AvgPool1d(2)
            self.L1_LSTM = nn.LSTM(256, 256, 1)
            self.L2 = nn.Linear(256, 512)
            self.L3 = nn.Linear(512, 512)
            self.L_a_prob = nn.Linear(512, nub_a)

    def forward(self, s):
        """
        s = [batch, seq_len, input_size] is given to CLSTM and LSTM
        """
        if self.net_type == 'DNN':
            x = F.relu(self.L1(s))
            x = F.relu(self.L2(x))
            x = F.relu(self.L3(x))

        if self.net_type == 'LSTM':
            s = s.transpose(0, 1)                   # [batch, seq_len, input_size] -> [seq_len, batch, input_size]
            x, h = self.L1(s)
            x = F.relu(self.L2(x[-1, :, :]))
            x = F.relu(self.L3(x))

        if self.net_type == 'C_LSTM':
            s = s.transpose(1, 2)                   # [batch, seq_len, input_size] -> [batch, input_size, seq_len]
            x = self.L1(s)
            x = self.L1_av(x)
            x = x.transpose(1, 2).transpose(0, 1)   # [batch, input_size, seq_len] -> [seq_len, batch, input_size]
            x, h = self.L1_LSTM(x)

            x = F.relu(self.L2(x[-1, :, :]))
            x = F.relu(self.L3(x))

        action_probs = F.softmax(self.L_a_prob(x), dim=1)
        return action_probs

    def get_act(self, s):
        s = torch.FloatTensor(s) #.unsqueeze(0)
        action_probs = self.forward(s)
        action = torch.argmax(action_probs, dim=1, keepdim=True)        # a_t

        action_distribution = Categorical(action_probs)
        action_ = action_distribution.sample().view(-1, 1)              # pi_theta(s_t)

        return action_.detach().cpu().numpy()

    def sample(self, s):
        action_probs = self.forward(s)
        action = torch.argmax(action_probs, dim=1, keepdim=True)        # a_t
        action_distribution = Categorical(action_probs)
        action_ = action_distribution.sample().view(-1, 1)              # pi_theta(s_t)

        # 0 값 방지
        z = (action_probs == 0.0).float() * 1e-8
        log_probs = torch.log(action_probs + z)                         # log(pi_theta(s_t))

        return action.detach().cpu().numpy()[0], action_, action_probs, log_probs


class CriticNet(BaseNetwork):
    def __init__(self, nub_a=3, nub_s=2, net_type='C_LSTM'):
        super(CriticNet, self).__init__()
        self.net_type = net_type

        if self.net_type == 'DNN':
            self.L1 = nn.Linear(nub_s, 256)
            self.L2 = nn.Linear(256, 512)
            self.L3 = nn.Linear(512, 512)
            self.L_q = nn.Linear(512, 1)

        if self.net_type == 'LSTM':
            self.L1 = nn.LSTM(nub_s, 256, 1)
            self.L2 = nn.Linear(256, 512)
            self.L3 = nn.Linear(512, 512)
            self.L_q = nn.Linear(512, 1)

        if self.net_type == 'C_LSTM':
            self.L1 = nn.Conv1d(in_channels=nub_s, out_channels=256, kernel_size=2)
            self.L1_av = nn.AvgPool1d(2)
            self.L1_LSTM = nn.LSTM(256, 256, 1)
            self.L2 = nn.Linear(256, 512)
            self.L3 = nn.Linear(512, 512)
            self.L_q = nn.Linear(512, 1)

    def forward(self, s):
        if self.net_type == 'DNN':
            x = F.relu(self.L1(s))
            x = F.relu(self.L2(x))
            x = F.relu(self.L3(x))

        if self.net_type == 'LSTM':
            s = s.transpose(0, 1)                   # [batch, seq_len, input_size] -> [seq_len, batch, input_size]
            x, h = self.L1(s)
            x = F.relu(self.L2(x[-1, :, :]))
            x = F.relu(self.L3(x))

        if self.net_type == 'C_LSTM':
            s = s.transpose(1, 2)                   # [batch, seq_len, input_size] -> [batch, input_size, seq_len]
            x = self.L1(s)
            x = self.L1_av(x)
            x = x.transpose(1, 2).transpose(0, 1)   # [batch, input_size, seq_len] -> [seq_len, batch, input_size]
            x, h = self.L1_LSTM(x)

            x = F.relu(self.L2(x[-1, :, :]))
            x = F.relu(self.L3(x))

        q = F.relu(self.L_q(x))
        return q


if __name__ == '__main__':
    # Test
    s = torch.Tensor([[0, 0.1, 0.2, 0.3]])
    s2 = torch.Tensor([[0, 0.1, 0.2, 0.3], [0, 0.1, 0.2, 0.4]])

    s3 = torch.Tensor(
        [
            [[1, 0.1], [1.1, 0.2], [1.2, 0.3]],
            [[1.1, 0.2], [1.2, 0.3], [1.3, 0.4]],
            [[1.2, 0.3], [1.3, 0.4], [1.4, 0.5]],
            [[1.3, 0.4], [1.4, 0.5], [10.5, 0.6]],
        ]
    )
    # print(s3.size())            # [batch, time_step, feature_dim]

    A_net = ActorNet()
    # print(A_net.sample(s))
    # print(A_net.sample(s2))
    # print(A_net.sample(s3))

    Q_net = CriticNet()
    # print(Q_net(s))
    # print(Q_net(s2))
    # print(Q_net(s3))