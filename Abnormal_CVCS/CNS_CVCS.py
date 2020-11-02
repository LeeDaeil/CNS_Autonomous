from COMMONTOOL import TOOL, PTCureve, CSFTree, NBPlot3D
from CNS_UDP_FAST import CNS
import numpy as np
import time
import random


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
        self.want_tick = 5  # 1sec

        self.Loger_txt = ''

        self.input_info = [
            # (para, x_round, x_min, x_max), (x_min=0, x_max=0 is not normalized.)
            ('PVCT',     1, 0,   100),        # VCT Press
            ('ZVCT',     1, 0,   100),        # VCT Level
            ('BLV616',   1, 0,   0),          # LV616, VCT->ChargingPump 유로 밸브
            ('KLAMPO71', 1, 0,   0),          # Charging Pump 1
            ('KLAMPO70', 1, 0,   0),          # Charging Pump 2

            ('KLAMPO69', 1, 0,   0),          # Charging Pump 3
            ('BHV50',    1, 0,   0),          # HV50 ChargingPump->RCP Seal 유로 밸브
            ('WCHGNO',   1, 0,   100),        # ChargingFlow 지시기
            ('BFV122',   1, 0,   1),          # Charging Valve Pos
            ('UCHGUT',   1, 0,   300),        # Charging Valve->RCS 온도 지시기

            ('BHV41',    1, 0,   1),          # Letdown HV41, RCS->HV43->VCT
            ('KHV43',    1, 0,   1),          # Letdown HV43, RCS->HV43->VCT
            ('BLV459',   1, 0,   1),          # Letdown LV459, RCS->VCT
            ('URHXUT',   1, 0,   300),        # Letdown RCS->VCT 온도 지시기
            ('BHV1',     1, 0,   1),          # HV1 Pos

            ('BHV2',     1, 0,   1),          # HV2 Pos
            ('BHV3',     1, 0,   1),          # HV3 Pos
            ('WNETLD',   1, 0,   100),        # Letdown HX Flow
            ('UNRHXUT',  1, 0,   300),        # Letdown HX Temp
            ('ZINST36',  1, 0,   100),        # Letdown HX Press = PV145 Pos

            # Boric acid Tank과 Makeup 고려가 필요한지 고민해야됨.

            # NewValue

        ]

        self.action_space = 1       # TODO "
        self.observation_space = len(self.input_info)

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
        state = []
        for para, x_round, x_min, x_max in self.input_info:
            if para in self.mem.keys():
                state.append(self.normalize(self.mem[para]['Val'], x_round, x_min, x_max))
            else:
                # ADD logic ----- 계산된 값을 사용하고 싶을 때
                pass

        # state = [self.mem[para]['Val'] / Round_val for para, Round_val in self.input_info]
        self.Loger_txt += f'{state}\t'
        return np.array(state)

    def get_reward(self):
        """
        R => _
        :return:
        """
        r = 0
        V = {
            'WLETDNO': self.mem['WLETDNO']['Val'],
            'CWHV1': 2.839
        }

        self.Loger_txt += f'R:{r}\t'
        return r

    def get_done(self, r):
        V = {}
        d = False

        self.Loger_txt += f'{d}\t'
        return d, self.normalize(r, 1, 0, 2)

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
        V = {}
        ActOrderBook = {}

        # Done Act
        self._send_control_to_cns()
        return AMod

    def step(self, A, mean_, std_):
        """
        A를 받고 1 step 전진
        :param A: [Act], numpy.ndarry, Act는 numpy.float32
        :return: 최신 state와 reward done 반환
        """
        # Old Data (time t) ---------------------------------------
        AMod = self.send_act(A)
        self.want_tick = int(5)

        self.Monitoring_ENV.push_ENV_val(i=self.Name,
                                         Dict_val={f'{Para}': self.mem[f'{Para}']['Val'] for Para in
                                                   ['PVCT', 'ZVCT', 'ZINST58', 'ZINST63', 'BFV122', 'BPV145']}
                                         )

        self.Monitoring_ENV.push_ENV_ActDis(i=self.Name,
                                            Dict_val={'Mean': 0, 'Std': 0.5}
                                            )

        # New Data (time t+1) -------------------------------------
        super(ENVCNS, self).step()
        self._append_val_to_list()
        self.ENVStep += 1

        reward = self.get_reward()
        done, reward = self.get_done(reward)
        self.Monitoring_ENV.push_ENV_reward(i=self.Name,
                                            Dict_val={'R': reward, 'AcuR': self.AcumulatedReward, 'Done': done})
        next_state = self.get_state()
        # ----------------------------------------------------------
        self.ENVlogging(s=self.Loger_txt)
        # self.Loger_txt = f'{next_state}\t'
        self.Loger_txt = ''
        return next_state, reward, done, AMod

    def reset(self, file_name):
        # 1] CNS 상태 초기화 및 초기화된 정보 메모리에 업데이트
        super(ENVCNS, self).reset(initial_nub=1, mal=False, mal_case=38, mal_opt=10, mal_time=50, file_name=file_name)
        # 2] 업데이트된 'Val'를 'List'에 추가 및 ENVLogging 초기화
        self._append_val_to_list()
        self.ENVlogging('')
        # 3] 'Val'을 상태로 제작후 반환
        state = self.get_state()
        # 4] 보상 누적치 및 ENVStep 초기화
        self.AcumulatedReward = 0
        self.ENVStep = 0
        # self.Monitoring_ENV.init_ENV_val(self.Name)
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
            env.step(A, 0, 0)