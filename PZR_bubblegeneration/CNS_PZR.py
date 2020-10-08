from COMMONTOOL import TOOL, PTCureve, CSFTree, NBPlot3D
from CNS_UDP_FAST import CNS
from PZR_bubblegeneration.PID_Na import PID
import numpy as np
import time
import random
import copy
import matplotlib.pylab as plt


class ENVCNS(CNS):
    def __init__(self, Name, IP, PORT, Monitoring_ENV=None):
        super(ENVCNS, self).__init__(threrad_name=Name,
                                     CNS_IP=IP, CNS_Port=PORT,
                                     Remote_IP='192.168.0.29', Remote_Port=PORT, Max_len=10)
        self.Monitoring_ENV = Monitoring_ENV
        self.Name = Name  # = id
        self.AcumulatedReward = 0
        self.ENVStep = 0
        self.LoggerPath = 'DB'
        self.want_tick = 5  # 20sec

        self.Loger_txt = ''

        self.input_info = [
            # (para, x_round, x_min, x_max)
            ('ZINST98',  1, 0,   100),     # SteamDumpPos
            ('ZINST87',  1, 0,   50),      # Steam Flow 1
            ('ZINST86',  1, 0,   50),      # Steam Flow 2
            ('ZINST85',  1, 0,   50),      # Steam Flow 3
            ('KLAMPO70', 1, 0,   0),       # Charging Pump2 State
            ('BHV22',    1, 0,   0),       # SI Valve State
            ('ZINST66',  1, 0,   25),      # PZRSprayPos
            ('UAVLEG2',  1, 150, 320),     # PTTemp
            ('ZINST65',  1, 0,   160),     # PTPressure
            ('ZINST78',  1, 0,   70),      # SG1Nar
            ('ZINST77',  1, 0,   70),      # SG2Nar
            ('ZINST76',  1, 0,   70),      # SG3Nar
            ('ZINST75',  1, 0,   80),      # SG1Pres
            ('ZINST74',  1, 0,   80),      # SG2Pres
            ('ZINST73',  1, 0,   80),      # SG3Pres
            ('ZINST72',  1, 0,   100),     # SG1Wid
            ('ZINST71',  1, 0,   100),     # SG2Wid
            ('ZINST70',  1, 0,   100),     # SG3Wid
            ('UUPPPL',   1, 100, 350),     # CoreExitTemp
            ('WFWLN1',   1, 0,   25),      # SG1Feed
            ('WFWLN2',   1, 0,   25),      # SG2Feed
            ('WFWLN3',   1, 0,   25),      # SG3Feed
            ('UCOLEG1',  1, 100, 0),       # RCSColdLoop1
            ('UCOLEG2',  1, 100, 0),       # RCSColdLoop2
            ('UCOLEG3',  1, 100, 0),       # RCSColdLoop3
            ('ZINST65',  1, 0,   160),     # RCSPressure
            ('ZINST63',  1, 0,   100),     # PZRLevel
        ]

        self.action_space = 2       # TODO "HV142" , "FV122"
        self.observation_space = len(self.input_info)

        self.PID_NA = PID(kp=0.03, ki=0.001, kd=1.0)
        self.PID_NA.SetPoint_pres = 24.0

        # GP
        # self.pl = NBPlot3D()

    # ENV Logger
    def ENVlogging(self, s):
        cr_time = time.strftime('%c', time.localtime(time.time()))
        if self.ENVStep == 0:
            with open(f'{self.Name}.txt', 'a') as f:
                f.write('==' * 20 + '\n')
                f.write(f'[{cr_time}]\n')
                f.write('==' * 20 + '\n')
        else:
            with open(f'{self.Name}.txt', 'a') as f:
                f.write(f'[{cr_time}] {self.Loger_txt}\n')

    def normalize(self, x, x_round, x_min, x_max):
        if x_max == 0 and x_min == 0:
            # It means X value is not normalized.
            x = x / x_round
        else:
            x = x_max if x >= x_max else x
            x = x_min if x <= x_min else x
            x = (x - x_min) / (x_max - x_min)
        return x

    def get_state(self):
        state = [self.normalize(self.mem[para]['Val'], x_round, x_min, x_max) for
                 para, x_round, x_min, x_max in self.input_info]
        # state = [self.mem[para]['Val'] / Round_val for para, Round_val in self.input_info]
        # self.Loger_txt += f'{state}\t'
        return np.array(state)

    def get_reward(self):
        r = 0
        return r

    def get_done(self, r):
        d = False
        return d, r

    def _send_control_save(self, zipParaVal):
        super(ENVCNS, self)._send_control_save(para=zipParaVal[0], val=zipParaVal[1])

    def send_act(self, A):
        """
        A 에 해당하는 액션을 보내고 나머지는 자동
        E.x)
            self._send_control_save(['KSWO115'], [0])
            ...
            self._send_control_to_cns()
        :param A: A 액션 [0, 0, 0] <- act space에 따라서
        :return: AMod: 수정된 액션
        """
        AMod = A
        V = {
            'CNSTime': self.mem['KCNTOMS']['Val'],
        }
        ActOrderBook = {
            'ChargingValveOpen': (['KSWO101', 'KSWO102'], [0, 1]),
            'ChargingValveClase': (['KSWO101', 'KSWO102'], [1, 0]),

            'LetdownValveOpen': (['KSWO231', 'KSWO232'], [0, 1]),
            'LetdownValveClose': (['KSWO231', 'KSWO232'], [1, 0]),

            'PZRBackHeaterOff': (['KSWO125'], [0]), 'PZRBackHeaterOn': (['KSWO125'], [1]),

            'PZRProHeaterMan': (['KSWO120'], [1]), 'PZRProHeaterAuto': (['KSWO120'], [0]),
            'PZRProHeaterDown': (['KSWO121', 'KSWO122'], [1, 0]),
            'PZRProHeaterUp': (['KSWO121', 'KSWO122'], [0, 1]),
        }
        self._send_control_save(ActOrderBook['PZRBackHeaterOn'])
        self._send_control_save(ActOrderBook['PZRProHeaterUp'])

        if V['CNSTime'] % (self.want_tick * 3) == 0:
            err_ = self.PID_NA.SetPoint_pres - self.mem['ZINST65']['Val']
            PID_out = self.PID_NA.update(err_, 1)

            if PID_out >= 0.005:
                self._send_control_save(ActOrderBook['LetdownValveClose'])
                print(f'Close {PID_out}')
            elif -0.005 < PID_out < 0.005:
                print(f'Stay {PID_out}')
            else:
                print(f'Open {PID_out}')
                self._send_control_save(ActOrderBook['LetdownValveOpen'])

        # Done Act
        self._send_control_to_cns()
        return AMod

    def step(self, A):
        AMod = self.send_act(A)
        self.want_tick = int(5)

        # New Data (time t+1) -------------------------------------
        super(ENVCNS, self).step()
        self._append_val_to_list()
        self.ENVStep += 1

        reward = self.get_reward()
        done, reward = self.get_done(reward)
        next_state = self.get_state()
        # ----------------------------------------------------------
        self.ENVlogging(s=self.Loger_txt)
        # self.Loger_txt = f'{next_state}\t'
        self.Loger_txt = ''
        return next_state, reward, done, AMod

    def reset(self, file_name):
        # 1] CNS 상태 초기화 및 초기화된 정보 메모리에 업데이트
        super(ENVCNS, self).reset(initial_nub=19, mal=False, mal_case=0, mal_opt=0, mal_time=0, file_name=file_name)
        # 2] 업데이트된 'Val'를 'List'에 추가 및 ENVLogging 초기화
        self._append_val_to_list()
        self.ENVlogging('')
        # 3] 'Val'을 상태로 제작후 반환
        state = self.get_state()
        # 4] 보상 누적치 및 ENVStep 초기화
        self.AcumulatedReward = 0
        self.ENVStep = 0
        # 5] FIX RADVAL
        self.FixedRad = random.randint(0, 20) * 5
        self.FixedTime = 0
        self.FixedTemp = 0

        return state

if __name__ == '__main__':
    # ENVCNS TEST
    env = ENVCNS(Name='Env1', IP='192.168.0.103', PORT=int(f'7101'))
    # Run
    for _ in range(1, 4):
        env.reset(file_name=f'Ep{_}')
        for __ in range(500):
            A = 0
            env.step(A)