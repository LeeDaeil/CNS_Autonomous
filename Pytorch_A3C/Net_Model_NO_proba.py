import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


class Agent_network(nn.Module):
    def __init__(self, agent_name, state_dim, state_time, action_dim, comp_dim):
        super(Agent_network, self).__init__()
        self.agent_name = agent_name
        self.state_dim = state_dim
        self.state_time = state_time
        self.action_dim = action_dim
        self.comp_dim = comp_dim

        self.distribution = torch.distributions.Categorical

        if self.state_time != 0:
            self.test_data = torch.empty((4, state_time, state_dim)).random_(1)
            self.test_one_data = torch.empty((1, state_time, state_dim)).random_(1)
            self.test_a_data = torch.empty((4, 1), dtype=int).random_(1)
            self.test_comp_a_data = torch.empty((4, 1), dtype=int).random_(1)
            # self.test_a_data = torch.empty((3, 1, action_dim)).random_(1)
            # self.test_comp_a_data = torch.empty((3, 1, comp_dim)).random_(1)
            self.test_r_data = torch.empty((4, 1)).random_(1)
        else:
            self.test_data = torch.empty((state_dim))

        # Build_net_structure
        # 1) build actor, critic net
        if state_time != 0:
            self.a_1 = nn.Conv1d(in_channels=state_dim, out_channels=state_dim, kernel_size=2, stride=1)
            self.a_2 = nn.Conv1d(in_channels=state_dim, out_channels=state_dim, kernel_size=2, stride=1)
            self.a_3 = nn.LSTM(input_size=state_dim, hidden_size=20, num_layers=1)

            # print(self.a_3.all_weights)
            self.comp_a = nn.Linear(in_features=20, out_features=comp_dim)

            self.cont_a = nn.Linear(in_features=20+comp_dim, out_features=action_dim)

            # self.c_1 = nn.Conv1d(in_channels=state_dim, out_channels=state_dim, kernel_size=2, stride=1)
            # self.c_2 = nn.Conv1d(in_channels=state_dim, out_channels=state_dim, kernel_size=2, stride=1)
            # self.c_3 = nn.LSTM(input_size=state_dim, hidden_size=3, num_layers=1)
            self.c_v = nn.Linear(in_features=20, out_features=1)
        else:
            self.a_1 = nn.Linear(state_dim, 2)
            self.a_m = nn.Linear(2, action_dim)
            self.a_s = nn.Linear(2, action_dim)

            self.c_1 = nn.Linear(state_dim, 2)
            self.c_v = nn.Linear(2, 1)

    def forward(self, x, softmax_dim=0):
        # x = torch.Tensor [time, val]
        # input processing
        x = x.transpose(1, 2)

        # Actor ========================================
        # Conv1d
        out = F.max_pool1d(self.a_1(x), 2)
        out = F.max_pool1d(self.a_2(out), 2)

        # LSTM
        out = out.transpose(1, 2)
        out = self.a_3(out)
        L_out = out[0][:, -1, :]

        # Linear mu, sigma
        comp_a = self.comp_a(L_out)
        comp_a = F.softmax(comp_a, softmax_dim)

        comp_m = self.distribution(comp_a)
        comp_a_fin = comp_m.sample()

        cont_a = torch.cat((L_out, comp_a), dim=1)

        cont_a = self.cont_a(cont_a)
        cont_a = F.softmax(cont_a, softmax_dim)
        cont_m = self.distribution(cont_a)
        cont_a_fin = cont_m.sample()

        value = self.c_v(L_out)

        return cont_a, cont_a_fin, comp_a, comp_a_fin, value


    def get_one_act(self, s):
        # s: torch.Tensor [batch ,time, val]
        self.training = False
        cont_a, cont_a_fin, comp_a, comp_a_fin, value = self.forward(s, softmax_dim=1)
        return cont_a, cont_a_fin, comp_a, comp_a_fin, value

    def get_acts(self, s):
        # s: torch.Tensor [batch ,time, val]
        self.training = False
        cont_a, cont_a_fin, comp_a, comp_a_fin, value = self.forward(s, softmax_dim=0)
        return cont_a, cont_a_fin, comp_a, comp_a_fin, value

    def loss_fun(self, s, a, c_a, td_target):
        # s: torch.Tensor [batch, time, val]
        # a: torch.Tensor [batch, val]
        # c_a: torch.Tensor [batch, val]
        # r: torch.Tensor [batch, val]
        self.train()

        cont_a, cont_a_fin, comp_a, comp_a_fin, value = self.get_acts(s)
        advantage = td_target - value

        prob_cont_a = cont_a.gather(1, a)                               # a is must torch.int64
        loss_cont_a = - torch.log(prob_cont_a) * advantage.detach()

        prob_comp_a = comp_a.gather(1, c_a)
        loss_comp_a = - torch.log(prob_comp_a) * advantage.detach()

        loss_value = F.smooth_l1_loss(value, td_target.detach())

        fin_loss = - loss_cont_a - loss_comp_a + loss_value
        return fin_loss.mean()

    # ===========================================================
    def test_model(self):
        # Test Part
        print(f"{'='*50}\n{self.agent_name}_Net Test\n{'-'*50}")
        print(self)
        print(f"Net_In \t\t\t {self.test_data.shape}\n{'-'*50}")
        print(f"Net_Out \t\t {self.forward(self.test_data, softmax_dim=0)}\n{'-'*50}")
        print(f"Net_Act \t\t {self.get_one_act(self.test_one_data)}\n{'-'*50}")
        print(f"Net_Act \t\t {self.get_acts(self.test_data)}\n{'-'*50}")
        print(f"Net_Loss \t\t {self.loss_fun(self.test_data,self.test_a_data,self.test_comp_a_data,self.test_r_data)}"
              f"\n{'-'*50}")
        print(f"{'='*50}")

    def show_model(self):
        return print(self)


if __name__ == '__main__':
    Agent = Agent_network('Main', state_dim=5, state_time=12, action_dim=3, comp_dim=8)
    Agent.test_model()