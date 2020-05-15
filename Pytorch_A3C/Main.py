from Pytorch_A3C.Net_Model import Agent_network
from Pytorch_A3C.CNS_UDP_FAST import CNS
import torch
import torch.multiprocessing as mp
import time
import datetime
import numpy as np
from collections import deque

STATE_D = 5
STATE_T = 12
ACT_D = 2


class Worker(mp.Process):
    def __init__(self, Remote_ip, Remote_port, CNS_ip, CNS_port, Shared_info,
                 G_net, G_OPT, L_net_name):
        super(Worker, self).__init__()
        self.CNS = CNS(self.name, CNS_ip, CNS_port, Remote_ip, Remote_port)
        self.L_net = Agent_network(agent_name=L_net_name, state_dim=STATE_D, state_time=STATE_T, action_dim=ACT_D)
        self.G_net, self.G_opt = G_net, G_OPT
        self.Shared_info = Shared_info

    def run(self):

        while True:
            self.CNS.init_cns(initial_nub=17)
            buffer_s, buffer_a, buffer_r = [], [], []
            ep_r = 0
            set_s = deque(maxlen=STATE_T)

            for i in range(STATE_T):
                self.CNS.run_freeze_CNS()
                # [M] monitoring part
                self.Shared_info.value = self.CNS.mem['KCNTOMS']['Val']
                s = [self.CNS.mem['KCNTOMS']['Val'], self.CNS.mem['UAVLEGM']['Val'], self.CNS.mem['UAVLEGS']['Val'],
                     self.CNS.mem['ZINST63']['Val'], self.CNS.mem['ZVCT']['Val']]
                set_s.append(s)

            for i in range(1000):
                # input-> action -> act_val : 입력 넣고 네트워크에서 나온 값 int로 치환
                a, prob_a = self.L_net.choose_action_fin(torch.FloatTensor(set_s).view(1, STATE_T, STATE_D))

                # 액션을 보내기
                self.send_action(action=a)

                # 1회 Run
                self.CNS.run_freeze_CNS()
                # [M] monitoring part
                self.Shared_info.value = self.CNS.mem['KCNTOMS']['Val']

                # 평가
                done, r = self.estimate_state()

                # 값 저장
                buffer_s.append(set_s)
                buffer_a.append(prob_a)
                buffer_r.append(r)

                if self.CNS.mem['KCNTOMS']['Val'] > 120:
                    print('DONE')
                    done = True

                s = [self.CNS.mem['KCNTOMS']['Val'], self.CNS.mem['UAVLEGM']['Val'], self.CNS.mem['UAVLEGS']['Val'],
                     self.CNS.mem['ZINST63']['Val'], self.CNS.mem['ZVCT']['Val']]
                set_s.append(s)     # now set_s is s_

                if np.shape(buffer_s)[0] >= 5:
                    print('Train!')
                    self.push_and_pull(self.G_opt, self.L_net, self.G_net, done, set_s, buffer_s, buffer_a, buffer_r)
                    buffer_s, buffer_a, buffer_r = [], [], []

                    if done:
                        break
                #

    def send_action_append(self, pa, va):
        for _ in range(len(pa)):
            self.para.append(pa[_])
            self.val.append(va[_])

    def send_action(self, action):
        # 전송될 변수와 값 저장하는 리스트
        self.para = []
        self.val = []

        if action == 0:
            self.send_action_append(['KSWO33', 'KSWO32'], [0, 0])  # Stay
        elif action == 1:
            self.send_action_append(['KSWO33', 'KSWO32'], [1, 0])  # Out
        elif action == 2:
            self.send_action_append(['KSWO33', 'KSWO32'], [0, 1])  # In

        # 최종 파라메터 전송
        self.CNS._send_control_signal(self.para, self.val)

    def estimate_state(self):
        done = False
        r = self.CNS.mem['QPROREL']['Val']

        print(r)
        return done, r

    def push_and_pull(self, opt, lnet, gnet, done, s_, bs, ba, br):
        gamma = 0.001
        if done:
            v_s_ = 0.0      # terminal
        else:
            v_s_ = lnet.forward(torch.FloatTensor(s_).view(1, STATE_T, STATE_D))[-1].data.numpy()[0, 0]

        buffer_v_target = []
        for r in br[::-1]:
            v_s_ = r + gamma * v_s_
            buffer_v_target.append(v_s_)
        buffer_v_target.reverse()

        loss = lnet.loss_fun(torch.FloatTensor(bs), torch.FloatTensor(ba), v_s_)
        print("LOSS", loss)
        opt.zero_grad()
        loss.backward()
        for lp, gp in zip(lnet.parameters(), gnet.parameters()):
            gp._grad = lp._grad
        opt.step()
        lnet.load_state_dict(gnet.state_dict())
        

class Shared_OPT(torch.optim.Adam):
    def __init__(self, Net_para):
        super(Shared_OPT, self).__init__(Net_para, lr=1e-3, betas=(0.95, 0.99), eps=1e-8,
                                         weight_decay=0)
        print(self.param_groups)


if __name__ == '__main__':
    GlobalNet = Agent_network(agent_name="Main", state_dim=STATE_D, state_time=STATE_T, action_dim=ACT_D)
    GlobalNet.share_memory()
    Opt = Shared_OPT(GlobalNet.parameters())

    Shared_info, Shared_info_iter = [], 0

    workers = []
    for cnsip, com_port, max_iter in zip(['192.168.0.9', '192.168.0.7', '192.168.0.4'],
                                         [7100, 7200, 7300], [1, 0, 0]):
        if max_iter != 0:
            for i in range(1, max_iter + 1):
                Shared_info.append(mp.Value('i', 0))
                workers.append(Worker(Remote_ip='192.168.0.10', Remote_port=com_port + i,
                                      CNS_ip=cnsip, CNS_port=com_port + i,
                                      Shared_info=Shared_info[Shared_info_iter],
                                      G_net=GlobalNet, G_OPT=Opt, L_net_name=f"L_net_{i}"
                                      )
                               )
                Shared_info_iter += 1

    for __ in workers:
        __.start()
        # time.sleep(1)
    # LOOP MONITORING
    while True:
        Fin_out = ''
        for _ in Shared_info:
            Fin_out += f"{_.value} | "
        print(Fin_out)
        time.sleep(1)
    [w.join() for w in workers]