import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


class Agent_network(nn.Module):
    def __init__(self, agent_name, state_dim, state_time, action_dim):
        super(Agent_network, self).__init__()
        self.agent_name = agent_name
        self.state_dim = state_dim
        self.state_time = state_time
        self.action_dim = action_dim

        self.distribution = torch.distributions.Normal

        if self.state_time != 0:
            self.test_data = torch.empty((3, state_time, state_dim))
            self.test_a_data = torch.empty((3, action_dim))
            self.test_r_data = torch.empty((3, action_dim))
        else:
            self.test_data = torch.empty((state_dim))

        # Build_net_structure
        # 1) build actor, critic net
        if state_time != 0:
            self.a_1 = nn.Conv1d(in_channels=state_dim, out_channels=state_dim, kernel_size=2, stride=1)
            self.a_2 = nn.Conv1d(in_channels=state_dim, out_channels=state_dim, kernel_size=2, stride=1)
            self.a_3 = nn.LSTM(input_size=state_dim, hidden_size=3, num_layers=1)

            # print(self.a_3.all_weights)

            self.a_m = nn.Linear(in_features=3, out_features=action_dim)
            self.a_s = nn.Linear(in_features=3, out_features=action_dim)

            self.c_1 = nn.Conv1d(in_channels=state_dim, out_channels=state_dim, kernel_size=2, stride=1)
            self.c_2 = nn.Conv1d(in_channels=state_dim, out_channels=state_dim, kernel_size=2, stride=1)
            self.c_3 = nn.LSTM(input_size=state_dim, hidden_size=3, num_layers=1)
            self.c_v = nn.Linear(in_features=3, out_features=1)
        else:
            self.a_1 = nn.Linear(state_dim, 2)
            self.a_m = nn.Linear(2, action_dim)
            self.a_s = nn.Linear(2, action_dim)

            self.c_1 = nn.Linear(state_dim, 2)
            self.c_v = nn.Linear(2, 1)

    def forward(self, x):
        # x = torch.Tensor [time, val]
        if self.state_time != 0:
            # input processing
            x = x.transpose(1, 2)

            # Actor ========================================
            # Conv1d
            out = F.max_pool1d(self.a_1(x), 2)
            out = F.max_pool1d(self.a_2(out), 2)

            # LSTM
            out = out.transpose(1, 2)
            out = self.a_3(out)
            out = out[0][:, -1, :]

            # Linear mu, sigma
            mu = self.a_m(out)
            sigma = F.softplus(self.a_s(out)) + 0.001

            # Critic ========================================
            # Conv1d
            out = F.max_pool1d(self.c_1(x), 2)
            out = F.max_pool1d(self.c_2(out), 2)

            # LSTM
            out = out.transpose(1, 2)
            out = self.c_3(out)
            out = out[0][:, -1, :]

            # Linear value
            value = self.c_v(out)

        else:
            a = self.a_1(x)
            a = self.c_1(x)

        return mu, sigma, value

    def choose_action_val(self, s):
        # s: torch.Tensor [batch ,time, val]
        self.training = False
        mu, sigma, _ = self.forward(s)
        # print(mu, mu.shape)
        get_dis = self.distribution(mu.view(-1, self.action_dim), sigma.view(-1, self.action_dim))
        return get_dis.sample()

    def choose_action_fin(self, s):
        # s: torch.Tensor [batch, time, val]
        sample_action = self.choose_action_val(s)
        # 샘플린된 값에서 action을 정수로 변환
        return sample_action.argmax(dim=1)[0].item(), sample_action.tolist()

    def loss_fun(self, s, a, r_t):
        # s: torch.Tensor [batch, time, val]
        # a: torch.Tensor [batch, val]
        # r: torch.Tensor [batch, val]
        # print(s.shape, a.shape, r_t.shape)
        self.train()

        mu, sigma, values = self.forward(s)
        td = r_t - values
        c_loss = td.pow(2)

        m = self.distribution(mu, sigma)
        log_prob = m.log_prob(a)
        entropy = 0.5 + 0.5 * math.log(2 * math.pi) + torch.log(m.scale)
        exp_v = log_prob * td.detach() + 0.005 * entropy
        a_loss = -exp_v
        return (c_loss + a_loss).mean()

    # ===========================================================
    def test_model(self):
        # Test Part
        print(f"{'='*50}\n{self.agent_name}_Net Test\n{'-'*50}")
        print(self)
        print(f"Net_In \t\t\t {self.test_data.shape}\n{'-'*50}")
        print(f"Net_Out \t\t {self.forward(self.test_data)}\n{'-'*50}")
        print(f"Net_Act \t\t {self.choose_action_val(self.test_data)}\n{'-'*50}")
        print(f"Net_Act \t\t {self.choose_action_fin(self.test_data)}\n{'-'*50}")
        print(f"Net_Loss \t\t {self.loss_fun(self.test_data,self.test_a_data,self.test_r_data)}"
              f"\n{'-'*50}")
        print(f"{'='*50}")

    def show_model(self):
        return print(self)


if __name__ == '__main__':
    Agent = Agent_network('Main', state_dim=5, state_time=12, action_dim=2)
    Agent.test_model()