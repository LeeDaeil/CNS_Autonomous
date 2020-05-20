from Pytorch_A3C.Net_Model import Agent_network
from Pytorch_A3C.CNS_UDP_FAST import CNS
import torch
import torch.multiprocessing as mp
import time

import pandas as pd
from collections import deque
import numpy as np

class Work_info:  # 데이터 저장 및 초기 입력 변수 선정
    def __init__(self):
        self.CURNET_COM_IP = '192.168.0.10'
        self.CNS_IP_LIST = ['192.168.0.9', '192.168.0.7', '192.168.0.4']
        self.CNS_PORT_LIST = [7100, 7200, 7300]
        self.CNS_NUMBERS = [5, 0, 0]

        self.L_net_name = 'TEMP'
        self.Act_D = 3
        self.Comp_D = 3

        self.DB_dict = {
            'Buf': {
                'S': [], 'R': [], 'A': [], 'CA': []
            },
            'Act': 0, 'Act_p': [0 for _ in range(self.Act_D)], 'Comp': 0,
            'Comp_p': [0 for _ in range(self.Comp_D)], 'Reward': 0,
            'Net_S': {
                # phy
                'Time': {'key': 'KCNTOMS', 'v': 0},
                'Critical': {'key': 'CRETIV', 'v': 0},
                'MWe_power': {'key': 'ZINST124', 'v': 0},

                # comp
                'Char_pump_2': {'key': 'KLAMPO70', 'v': 0},
                'BHV22': {'key': 'BHV22', 'v': 0},
                'RHR_pump': {'key': 'KLAMPO55', 'v': 0},
            },
            'DB': {
                'Act': [], 'Act_p': [], 'Comp': [], 'Comp_p': [], 'Reward': [], 'Test_val': [],
                # + 'Net_s'
            }
        }
        # copy
        for _ in self.DB_dict['Net_S'].keys():
            self.DB_dict['DB'][_] = []

        self.State_D = len(self.DB_dict['Net_S'].keys())
        self.State_t = 12
        #

    def make_s(self, mem):      # 네트워크 입력 변수의 리스트에 상응하는 S 반환
        # 1) mem 에서 업데이트 및 저장
        for val in self.DB_dict['Net_S'].keys():
            if self.DB_dict['Net_S'][val]['key'] in mem.keys():
                self.DB_dict['Net_S'][val]['v'] = mem[self.DB_dict['Net_S'][val]['key']]['Val']/100

        # 2) Overwrite 및 변수 가공
        self.DB_dict['Net_S']['MWe_power']['v'] = self.DB_dict['Net_S']['MWe_power']['v']/1000

        # 3) Save DB
        self.DB_dict['DB']['Act'].append(self.DB_dict['Act'])
        self.DB_dict['DB']['Act_p'].append(self.DB_dict['Act_p'])
        self.DB_dict['DB']['Comp'].append(self.DB_dict['Comp'])
        self.DB_dict['DB']['Comp_p'].append(self.DB_dict['Comp_p'])
        self.DB_dict['DB']['Reward'].append(self.DB_dict['Reward'])
        self.DB_dict['DB']['Test_val'].append(mem['KCNTOMS']['Val'])
        for val in self.DB_dict['Net_S'].keys():
            self.DB_dict['DB'][val].append(self.DB_dict['Net_S'][val]['v'])

        return [self.DB_dict['Net_S'][_]['v'] for _ in self.DB_dict['Net_S'].keys()]

    # SAVE 모듈
    def save_val(self, para, val):
        self.DB_dict[para] = val

    def dump_save_val(self):            # 저장된 데이터 CSV 저장
        temp = pd.DataFrame(self.DB_dict['DB'])
        temp.to_csv(f'{self.L_net_name}.csv')

    def init_save_db(self):            # 저장용 변수 초기화
        self.dump_save_val()
        for _ in self.DB_dict['DB'].keys():
            self.DB_dict['DB'][_].clear()
        self.DB_dict['Act'] = 0
        self.DB_dict['Act_p'] = [0 for _ in range(self.Act_D)]
        self.DB_dict['Reward'] = 0
        self.DB_dict['Comp'] = 0
        self.DB_dict['Comp_p'] = [0 for _ in range(self.Comp_D)]
        return 0

    # 버퍼
    def init_buf(self):                 # 버퍼용 변수 초기화
        self.DB_dict['Buf'] = {'S': [], 'R': [], 'A': [], 'CA': []}
        return 0

    def append_buf(self, s, r, a, ca): # 값 저장후 각 버퍼의 길이 반환
        self.DB_dict['Buf']['S'].append(s)
        self.DB_dict['Buf']['R'].append(r)
        self.DB_dict['Buf']['A'].append(a)
        self.DB_dict['Buf']['CA'].append(ca)
        return [len(self.DB_dict['Buf'][key]) for key in self.DB_dict['Buf'].keys()]


class Worker(mp.Process):
    def __init__(self, Remote_ip, Remote_port, CNS_ip, CNS_port, Shared_info,
                 G_net, G_OPT, L_net_name):
        super(Worker, self).__init__()
        self.CNS = CNS(self.name, CNS_ip, CNS_port, Remote_ip, Remote_port)
        self.W_info = Work_info()
        self.W_info.L_net_name = L_net_name
        self.L_net = Agent_network(agent_name=L_net_name, state_dim=self.W_info.State_D,
                                   state_time=self.W_info.State_t, action_dim=self.W_info.Act_D,
                                   comp_dim=self.W_info.Comp_D)
        self.G_net, self.G_opt = G_net, G_OPT
        self.Shared_info = Shared_info[0]
        self.Shared_info_iter = Shared_info[1]

    def run(self):
        while True:
            self.CNS.init_cns(initial_nub=1)
            time.sleep(1)
            self.CNS._send_malfunction_signal(12, 100100, 15)
            time.sleep(1)
            ep_r = 0
            ep_loss = []
            self.Shared_info_iter.value += 1
            set_s = deque(maxlen=self.W_info.State_t)

            for i in range(self.W_info.State_t):
                self.CNS.run_freeze_CNS()
                # [M] monitoring part
                self.Shared_info.value = self.CNS.mem['KCNTOMS']['Val']
                # self.Shared_info.value = ep_r
                set_s.append(self.W_info.make_s(self.CNS.mem))  # S 만들고 저장

            for i in range(1000):
                # input-> action -> act_val : 입력 넣고 네트워크에서 나온 값 int로 치환
                _input_state = torch.FloatTensor(set_s).view(1, self.W_info.State_t, self.W_info.State_D)
                a, prob_a, ca, prob_ca = self.L_net.choose_action_fin(_input_state)

                # 액션을 보내기 및 저장
                self.send_action(comp=ca, action=a)
                self.W_info.save_val('Act', a)
                self.W_info.save_val('Act_p', prob_a[0])
                self.W_info.save_val('Comp', ca)
                self.W_info.save_val('Comp_p', prob_ca[0])

                # 1회 Run
                self.CNS.run_freeze_CNS()
                # [M] monitoring part
                self.Shared_info.value = self.CNS.mem['KCNTOMS']['Val']

                # 평가
                done, r, suc = self.estimate_state(a, ca)
                # ep_r += int(r*100)
                # self.Shared_info.value = ep_r

                self.W_info.save_val('Reward', r)

                # 값 저장후 각 버퍼의 길이 리스트 반환
                [Leg_s, Leg_r, Leg_a, Leg_ca] = self.W_info.append_buf(s=set_s, r=r, a=prob_a, ca=prob_ca)

                if self.CNS.mem['KCNTOMS']['Val'] > 100:
                    # print('DONE')
                    done = True

                set_s.append(self.W_info.make_s(self.CNS.mem))  # S 만들고 저장

                if Leg_s >= 5 or done:
                    # print('Train!')
                    loss_ = self.push_and_pull(self.G_opt, self.L_net, self.G_net,
                                               done, set_s, self.W_info.DB_dict['Buf'])
                    ep_loss.append(loss_)
                    self.W_info.init_buf()

                    if done:
                        print(self.Shared_info_iter.value, '|', suc, sum(self.W_info.DB_dict['DB']['Reward']),
                              '|', sum(ep_loss)/len(ep_loss))
                        self.W_info.init_save_db()
                        # print('DONE - DB initial')
                        break
                #

    def send_action_append(self, pa, va):
        for _ in range(len(pa)):
            self.para.append(pa[_])
            self.val.append(va[_])

    def send_action(self, comp, action):
        # 전송될 변수와 값 저장하는 리스트
        self.para = []
        self.val = []

        def comp_act_updown(action, up, down):
            if action == 0:
                 self.send_action_append([up, down], [0, 0])  # Stay
            elif action == 1:
                 self.send_action_append([up, down], [1, 0])  # Out
            elif action == 2:
                 self.send_action_append([up, down], [0, 1])  # In

        def comp_act_onoff(action, para):
            if action == 0:
                 pass
            elif action == 1:
                 self.send_action_append([para], [1])  # On
            elif action == 2:
                 self.send_action_append([para], [0])  # Off

        if comp == 0: comp_act_onoff(action, 'KSWO70')      # charging
        if comp == 1: comp_act_onoff(action, 'KSWO81')      # HV22
        if comp == 2: comp_act_onoff(action, 'KSWO53')      # HV22

        # if action == 0:
        #     self.send_action_append(['KSWO33', 'KSWO32'], [0, 0])  # Stay
        # elif action == 1:
        #     self.send_action_append(['KSWO33', 'KSWO32'], [1, 0])  # Out
        # elif action == 2:
        #     self.send_action_append(['KSWO33', 'KSWO32'], [0, 1])  # In

        # 최종 파라메터 전송
        self.CNS._send_control_signal(self.para, self.val)

    def estimate_state(self, a, ca):
        # Safety function 1: = reactivity_control
        Reactivity_control = [0, 0, 0]
        if True:
            # 1) Reactivity
            if self.CNS.mem['CRETIV']['Val'] < 0:
                Reactivity_control[0] = 1
            else:
                Reactivity_control[0] = 0
            # 2) Stabilize or reduce reactor power
            if 0 <= self.CNS.mem['QPROREL']['Val'] < 0.02:
                Reactivity_control[1] += 0.5
            else:
                Reactivity_control[1] += 0
            if self.CNS.mem['ZINST124']['Val'] < 1:
                Reactivity_control[1] += 0.5
            else:
                Reactivity_control[1] += 0
            # 3) Boration addition rate
            if True:
                # 3-1) Charging Line Flow
                if self.CNS.mem['KLAMPO70']['Val'] == 1 and self.CNS.mem['BHV22']['Val'] == 1:
                    Reactivity_control[2] += 0.5
                else:
                    Reactivity_control[2] += 0
                # 3-2) IRWST->HV8->RHR->HV603 Flow
                if self.CNS.mem['KLAMPO55']['Val'] == 1 and self.CNS.mem['ZRWST']['Val'] > 0 \
                        and self.CNS.mem['BHV8']['Val'] == 1 and self.CNS.mem['BHV603']['Val'] >= 1:
                    Reactivity_control[2] += 0.5
                else:
                    Reactivity_control[2] += 0

        done = False
        r = sum(Reactivity_control)
        # print(Reactivity_control, r)
        # r = self.CNS.mem['QPROREL']['Val']

        # if ca == 3:
        #     if a == 0:
        #         r = 0.1
        #     else:
        #         r = 0.05
        # else:
        #     r = 0
        #
        # if r == 0:
        #     done = True

        success = False

        if r == 2.5:
            done = True
            success = True
            r = r/20 + 0.5
        else:
            r = r/20
        return done, r, success

    def push_and_pull(self, opt, lnet, gnet, done, s_, buf):
        bs, ba, br, bca = buf['S'], buf['A'], buf['R'], buf['CA']
        gamma = 0.001
        if done:
            v_s_ = 0.0      # terminal
        else:
            # 2 -> value
            v_s_ = lnet.forward(torch.FloatTensor(s_).view(1, self.W_info.State_t,
                                                           self.W_info.State_D))[2].data.numpy()[0, 0]

        buffer_v_target = []
        for r in br[::-1]:
            v_s_ = r + gamma * v_s_
            buffer_v_target.append(v_s_)
        buffer_v_target.reverse()

        loss = lnet.loss_fun(torch.FloatTensor(bs), torch.FloatTensor(ba), torch.FloatTensor(bca),
                             torch.FloatTensor(buffer_v_target).view(len(bs), 1))
        # print("LOSS", loss)
        opt.zero_grad()
        loss.backward()
        for lp, gp in zip(lnet.parameters(), gnet.parameters()):
            gp._grad = lp._grad
        opt.step()
        lnet.load_state_dict(gnet.state_dict())
        return loss.item()


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
                              state_time=W_info.State_t, action_dim=W_info.Act_D,
                              comp_dim=W_info.Comp_D)
    GlobalNet.share_memory()
    Opt = Shared_OPT(GlobalNet.parameters())

    Shared_info, Shared_info_iter = [], 0
    Shared_info_ep = mp.Value('i', 0)

    workers = []
    for cnsip, com_port, max_iter in zip(W_info.CNS_IP_LIST,
                                         W_info.CNS_PORT_LIST,
                                         W_info.CNS_NUMBERS):
        if max_iter != 0:
            for i in range(1, max_iter + 1):
                Shared_info.append(mp.Value('i', 0))
                workers.append(Worker(Remote_ip=W_info.CURNET_COM_IP, Remote_port=com_port + i,
                                      CNS_ip=cnsip, CNS_port=com_port + i,
                                      Shared_info=(Shared_info[Shared_info_iter], Shared_info_ep),
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