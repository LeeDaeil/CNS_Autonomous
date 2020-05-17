from Pytorch_A3C.Net_Model import Agent_network
from Pytorch_A3C.CNS_UDP_FAST import CNS
import torch
import torch.multiprocessing as mp
import time
import datetime
import numpy as np
from collections import deque


class Work_info:  # 데이터 저장 및 초기 입력 변수 선정함.
    def __init__(self):
        self.Act_D = 3
        self.State_D = len(self.state_para_list())
        self.State_t = 12

        self.CURNET_COM_IP = '192.168.0.10'
        self.CNS_IP_LIST = ['192.168.0.9', '192.168.0.7', '192.168.0.4']
        self.CNS_PORT_LIST = [7100, 7200, 7300]
        self.CNS_NUMBERS = [2, 0, 0]

    def state_para_list(self):  # 네트워크 입력 변수의 리스트를 반환
        para_list = [
            'KCNTOMS',   # 시간
            'UAVLEGM',   # 평균온도
            'UAVLEGS',   # Ref 온도
            'ZINST63',   # VCT level
            'ZVCT',      # VCT Level
        ]
        return para_list

    def make_s(self, mem):      # 네트워크 입력 변수의 리스트에 상응하는 S 반환
        para_list = self.state_para_list()
        return [mem[para]['Val'] for para in para_list]


class Worker(mp.Process):
    def __init__(self, Remote_ip, Remote_port, CNS_ip, CNS_port, Shared_info,
                 G_net, G_OPT, L_net_name):
        super(Worker, self).__init__()
        self.CNS = CNS(self.name, CNS_ip, CNS_port, Remote_ip, Remote_port)
        self.W_info = Work_info()
        self.L_net = Agent_network(agent_name=L_net_name, state_dim=self.W_info.State_D,
                                   state_time=self.W_info.State_t, action_dim=self.W_info.Act_D)
        self.G_net, self.G_opt = G_net, G_OPT
        self.Shared_info = Shared_info

    def run(self):
        while True:
            self.CNS.init_cns(initial_nub=17)
            buffer_s, buffer_a, buffer_r = [], [], []
            ep_r = 0
            set_s = deque(maxlen=self.W_info.State_t)

            for i in range(self.W_info.State_t):
                self.CNS.run_freeze_CNS()
                # [M] monitoring part
                self.Shared_info.value = self.CNS.mem['KCNTOMS']['Val']
                set_s.append(self.W_info.make_s(self.CNS.mem))  # S 만들고 저장

            for i in range(1000):
                # input-> action -> act_val : 입력 넣고 네트워크에서 나온 값 int로 치환
                a, prob_a = self.L_net.choose_action_fin(torch.FloatTensor(set_s).view(1, self.W_info.State_t, self.W_info.State_D))

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

                set_s.append(self.W_info.make_s(self.CNS.mem))  # S 만들고 저장

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
            v_s_ = lnet.forward(torch.FloatTensor(s_).view(1, self.W_info.State_t,
                                                           self.W_info.State_D))[-1].data.numpy()[0, 0]

        buffer_v_target = []
        for r in br[::-1]:
            v_s_ = r + gamma * v_s_
            buffer_v_target.append(v_s_)
        buffer_v_target.reverse()

        loss = lnet.loss_fun(torch.FloatTensor(bs), torch.FloatTensor(ba), v_s_)
        # print("LOSS", loss)
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

    def show_para(self):
        # 가중치 업데이트 여부 테스트.
        want_show = self.param_groups[0]['params'][0][0]
        return print(want_show, type(want_show), len(want_show))


if __name__ == '__main__':
    W_info = Work_info()
    GlobalNet = Agent_network(agent_name="Main", state_dim=W_info.State_D,
                              state_time=W_info.State_t, action_dim=W_info.Act_D)
    GlobalNet.share_memory()
    Opt = Shared_OPT(GlobalNet.parameters())

    Shared_info, Shared_info_iter = [], 0

    workers = []
    for cnsip, com_port, max_iter in zip(W_info.CNS_IP_LIST,
                                         W_info.CNS_PORT_LIST,
                                         W_info.CNS_NUMBERS):
        if max_iter != 0:
            for i in range(1, max_iter + 1):
                Shared_info.append(mp.Value('i', 0))
                workers.append(Worker(Remote_ip=W_info.CURNET_COM_IP, Remote_port=com_port + i,
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
        # print(Opt.show_para())    # 가중치 업데이트 여부 테스트
        time.sleep(1)
    [w.join() for w in workers]