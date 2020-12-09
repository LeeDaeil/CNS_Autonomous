from COMMONTOOL import *
from CNS_UDP_FAST import CNS
from PZR_bubblegeneration.PID_Na import PID
import numpy as np
import time
import random
import copy

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


class CMem:
    def __init__(self, mem):
        self.m = mem  # Line CNSmem -> getmem
        self.update()

    def update(self):
        self.CTIME = self.m['KCNTOMS']['Val']       # CNS Time
        self.CDelt = self.m['TDELTA']['Val']
        self.PZRPres = self.m['ZINST65']['Val']
        self.PZRLevl = self.m['ZINST63']['Val']
        self.PZRTemp = self.m['UPRZ']['Val']
        self.ExitCoreT = self.m['UUPPPL']['Val']

        self.FV122 = self.m['BFV122']['Val']
        self.FV122M = self.m['KLAMPO95']['Val']

        self.HV142 = self.m['BHV142']['Val']
        self.HV142Flow = self.m['WRHRCVC']['Val']

        self.PZRSprayPos = self.m['ZINST66']['Val']

        self.LetdownSet = self.m['ZINST36']['Val']  # Letdown setpoint
        self.LetdownSetM = self.m['KLAMPO89']['Val']  # Letdown setpoint Man0/Auto1


class ENVCNS(CNS):
    def __init__(self, Name, IP, PORT):
        super(ENVCNS, self).__init__(threrad_name=Name,
                                     CNS_IP=IP, CNS_Port=PORT,
                                     Remote_IP='192.168.0.29', Remote_Port=PORT, Max_len=10)

        # Plot --------------------------------------------------------------------------------
        # -------------------------------------------------------------------------------------
        # Initial and Memory
        self.Name = Name  # = id
        self.AcumulatedReward = 0
        self.ENVStep = 0
        self.LoggerPath = 'DB'
        self.want_tick = 5  # 1sec
        self.Loger_txt = ''
        self.CMem = CMem(self.mem)
        # RL ----------------------------------------------------------------------------------
        self.input_info = [
            # (para, x_round, x_min, x_max), (x_min=0, x_max=0 is not normalized.)
            ('BHV142',     1, 0,   0),       # Letdown(HV142)
            ('WRHRCVC',    1, 0,   0),       # RHR to CVCS Flow
            ('WNETLD',     1, 0,   10),      # Total Letdown Flow
            ('BFV122',     1, 0,   0),       # ChargingValve(FV122)
            ('WNETCH',     1, 0,   10),      # Total Charging Flow
            ('ZINST66',    1, 0,   30),      # PZR spray
            ('ZINST65',    1, 0,   160),     # RCSPressure
            ('ZINST63',    1, 0,   100),     # PZRLevel
            ('UUPPPL',     1, 0,   200),     # Core Exit Temperature
            ('UPRZ',       1, 0,   300),     # PZR Temperature

            # ('ZINST36',  1, 0,   0),      # Letdown Pressrue

            ('SetPres',    1, 0,   30),      # Pres-Setpoint
            ('SetLevel',   1, 0,   30),      # Level-Setpoint
            ('ErrPres',    1, 0,   100),     # RCSPressure - setpoint
            ('UpPres',     1, 0,   100),     # RCSPressure - Up
            ('DownPres',   1, 0,   100),     # RCSPressure - Down
            ('ErrLevel',   1, 0,   100),     # PZRLevel - setpoint
            ('UpLevel',    1, 0,   100),     # PZRLevel - Up
            ('DownLevel',  1, 0,   100),     # PZRLevel - Down
        ]

        self.action_space = 3       # TODO HV142 [0], Spray [1], FV122 [2]
        self.observation_space = len(self.input_info)
        # -------------------------------------------------------------------------------------
        # PID Part
        self.PID_Mode = False
        self.PID_Prs = PID(kp=0.03, ki=0.001, kd=1.0)
        self.PID_Prs_S = PID(kp=0.03, ki=0.001, kd=1.0)
        self.PID_Lev = PID(kp=0.03, ki=0.001, kd=1.0)
        self.PID_Prs.SetPoint = 27.0           # Press set-point
        self.PID_Prs_S.SetPoint = 27.0         # Press set-point
        self.PID_Lev.SetPoint = 30.0           # Level set-point
        # -------------------------------------------------------------------------------------

    # ENV TOOLs =======================================================================================================
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

    # ENV RL TOOLs ====================================================================================================
    def get_state(self):
        state = []
        for para, x_round, x_min, x_max in self.input_info:
            if para in self.mem.keys():
                state.append(self.normalize(self.mem[para]['Val'], x_round, x_min, x_max))
            else:
                # ADD logic ----- 계산된 값을 사용하고 싶을 때
                if para == 'SetPres':
                    v_ = self.PID_Prs.SetPoint
                    state.append(self.normalize(v_, x_round, x_min, x_max))
                if para == 'SetLevel':
                    v_ = self.PID_Lev.SetPoint
                    state.append(self.normalize(v_, x_round, x_min, x_max))
                if para == 'ErrPres':
                    v_ = - abs(self.PID_Prs.SetPoint - self.CMem.PZRPres)
                    # setpoint - 현재 압력 한뒤에 - abs
                    state.append(self.normalize(v_, x_round, x_min, x_max))
                if para == 'UpPres':
                    v_ = self.PID_Prs.SetPoint + 2
                    v_ = v_ - self.CMem.PZRPres
                    state.append(self.normalize(v_, x_round, x_min, x_max))
                if para == 'DownPres':
                    v_ = self.PID_Prs.SetPoint - 2
                    v_ = self.CMem.PZRPres - v_
                    state.append(self.normalize(v_, x_round, x_min, x_max))
                if para == 'ErrLevel':
                    v_ = - abs(self.PID_Lev.SetPoint - self.CMem.PZRLevl)
                    # setpoint - 현재 압력 한뒤에 - abs
                    state.append(self.normalize(v_, x_round, x_min, x_max))
                if para == 'UpLevel':
                    v_ = self.PID_Lev.SetPoint + 2
                    v_ = v_ - self.CMem.PZRLevl
                    state.append(self.normalize(v_, x_round, x_min, x_max))
                if para == 'DownLevel':
                    v_ = self.PID_Lev.SetPoint - 2
                    v_ = self.CMem.PZRLevl - v_
                    state.append(self.normalize(v_, x_round, x_min, x_max))
                pass

        # state = [self.mem[para]['Val'] / Round_val for para, Round_val in self.input_info]
        self.Loger_txt += f'{state}\t'
        return np.array(state)

    def get_reward(self, A):
        """

        :param A: tanh (-1 ~ 1) 사이 값
        :return:
        """
        r = 0
        if self.CMem.PZRLevl >= 95:                 # 기포 생성 이전
            r1, r2, c = 0, 0, 0
            # 압력
            if abs(self.CMem.PZRPres - self.PID_Prs.SetPoint) < 0.25:
                r1 += 0.1                                                       # 압력 저정 범위 안에 존재 + 조작 x
            else:
                r1 += - abs(self.CMem.PZRPres - self.PID_Prs.SetPoint)/100
            r1 = np.clip(r1, 0, 1)
            # 수위
            # 제어
            if abs(A[0]) < 0.6: c += 0.01
            if abs(A[1]) < 0.6: c += 0.01
            if abs(A[2]) < 0.6: c += 0.01
        else:                                       # 기포 생성 이후
            r1, r2, c = 0, 0, 0
            # 압력
            if abs(self.CMem.PZRPres - self.PID_Prs.SetPoint) < 0.25:
                r1 += 0.1  # 압력 저정 범위 안에 존재 + 조작 x
            else:
                r1 += - abs(self.CMem.PZRPres - self.PID_Prs.SetPoint)/100
            r1 = np.clip(r1, 0, 1)
            # 수위
            if abs(self.CMem.PZRLevl - self.PID_Lev.SetPoint) < 0.25:
                r2 += 0.1  # 압력 저정 범위 안에 존재 + 조작 x
            else:
                r2 += - abs(self.CMem.PZRLevl - self.PID_Lev.SetPoint)
            # 제어
            if abs(A[0]) < 0.6: c += 0.01
            if abs(A[1]) < 0.6: c += 0.01
            if abs(A[2]) < 0.6: c += 0.01

        r = r1 + r2 + c
        self.Loger_txt += f'R:{r}|{r1}|{r2}|{c}\t'
        return r

    def get_done(self, r):
        # r = self.normalize(r, 1, 0, 2)
        d = False
        if self.CMem.ExitCoreT > 176:
            d = True

        self.Loger_txt += f'{d}\t'
        return d, r #self.normalize(r, 1, 0, 2)

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
        ActOrderBook = {
            'ChargingValveOpen': (['KSWO101', 'KSWO102'], [0, 1]),
            'ChargingValveStay': (['KSWO101', 'KSWO102'], [0, 0]),
            'ChargingValveClose': (['KSWO101', 'KSWO102'], [1, 0]),

            'ChargingEdit': (['BFV122'], [0.12]),

            'LetdownValveOpen': (['KSWO231', 'KSWO232'], [0, 1]),
            'LetdownValveStay': (['KSWO231', 'KSWO232'], [0, 0]),
            'LetdownValveClose': (['KSWO231', 'KSWO232'], [1, 0]),

            'LetdownPresSetUp':(['KSWO90', 'KSWO91'], [0, 1]),
            'LetdownPresSetStay': (['KSWO90', 'KSWO91'], [0, 1]),
            'LetdownPresSetDown': (['KSWO90', 'KSWO91'], [1, 0]),
            'LetdownPresSetA': (['KSWO89'], [0]),

            'PZRBackHeaterOff': (['KSWO125'], [0]), 'PZRBackHeaterOn': (['KSWO125'], [1]),

            'PZRProHeaterMan': (['KSWO120'], [1]), 'PZRProHeaterAuto': (['KSWO120'], [0]),

            'PZRProHeaterDown': (['KSWO121', 'KSWO122'], [1, 0]),
            'PZRProHeaterStay': (['KSWO121', 'KSWO122'], [0, 0]),
            'PZRProHeaterUp': (['KSWO121', 'KSWO122'], [0, 1]),

            'PZRSprayOpen': (['KSWO126', 'KSWO127'], [0, 1]),
            'PZRSprayStay': (['KSWO126', 'KSWO127'], [0, 0]),
            'PZRSprayClose': (['KSWO126', 'KSWO127'], [1, 0]),

            'LetDownSetDown': (['KSWO90', 'KSWO91'], [1, 0]),
            'LetDownSetStay': (['KSWO90', 'KSWO91'], [0, 0]),
            'LetDownSetUP': (['KSWO90', 'KSWO91'], [0, 1]),

            'ChangeDelta': (['TDELTA'], [1.0]),
            'ChargingAuto': (['KSWO100'], [0])
        }
        self._send_control_save(ActOrderBook['PZRBackHeaterOn'])
        self._send_control_save(ActOrderBook['PZRProHeaterUp'])
        # =========================================================================================
        #  CORE!!!
        # =========================================================================================
        if self.CMem.PZRLevl >= 95:                                         # 가압기 기포 생성 이전
            self.PID_Prs.SetPoint = 27
            # ----------------------------- PRESS -------------------------------------------------
            if self.PID_Mode:
                PID_out = self.PID_Prs.update(self.CMem.PZRPres, 1)
                if PID_out >= 0.005:
                    self._send_control_save(ActOrderBook['LetdownValveClose'])
                elif -0.005 < PID_out < 0.005:
                    self._send_control_save(ActOrderBook['LetdownValveStay'])
                else:
                    self._send_control_save(ActOrderBook['LetdownValveOpen'])
            else:
                # A[0] HV142
                if abs(A[0]) < 0.6:
                    self._send_control_save(ActOrderBook['LetdownValveStay'])
                else:
                    if A[0] < 0: self._send_control_save(ActOrderBook['LetdownValveClose'])
                    else: self._send_control_save(ActOrderBook['LetdownValveOpen'])
                # A[1] PZR spray
                AMod[1] = 0
            # ----------------------------- Level -------------------------------------------------
            if self.PID_Mode:
                PID_out = self.PID_Lev.update(self.CMem.PZRLevl, 1)
                # if PID_out >= 0.005:
                #     self._send_control_save(ActOrderBook['ChargingValveOpen'])
                # elif -0.005 < PID_out < 0.005:
                #     self._send_control_save(ActOrderBook['ChargingValveStay'])
                # else:
                #     self._send_control_save(ActOrderBook['ChargingValveClose'])
            else:
                # A[2] FV122
                AMod[2] = 0
            # ----------------------------- ----- -------------------------------------------------
        else:                                                               # 가압기 기포 생성 이후
            self.PID_Prs.SetPoint = 30
            self.PID_Prs_S.SetPoint = 30
            self.PID_Lev.SetPoint = 30
            # ----------------------------- PRESS -------------------------------------------------
            if self.PID_Mode:
                # HV142 ----------------------------------------------------------
                PID_out = self.PID_Prs.update(self.CMem.PZRPres, 1)
                if self.CMem.HV142 != 0:
                    self._send_control_save(ActOrderBook['LetdownValveClose'])
                # if PID_out >= 0.005:
                #     self._send_control_save(ActOrderBook['LetdownValveClose'])
                # elif -0.005 < PID_out < 0.005:
                #     self._send_control_save(ActOrderBook['LetdownValveStay'])
                # else:
                #     self._send_control_save(ActOrderBook['LetdownValveOpen'])

                # Spray ----------------------------------------------------------
                PID_out = self.PID_Prs_S.update(self.CMem.PZRPres, 1)
                if PID_out >= 0.005:
                    self._send_control_save(ActOrderBook['PZRSprayClose'])
                elif -0.005 < PID_out < 0.005:
                    self._send_control_save(ActOrderBook['PZRSprayStay'])
                else:
                    self._send_control_save(ActOrderBook['PZRSprayOpen'])
                # LetPress ----------------------------------------------------------
                if self.CMem.LetdownSetM == 1:
                    self._send_control_save(ActOrderBook['LetdownPresSetA'])

                # print(f'GetPoint|{self.CMem.PZRPres}, {self.CMem.PZRPres}|\n'
                #       f'LetdownPos:{self.CMem.HV142}:{self.CMem.HV142Flow}|'
                #       f'PZRSpray:{self.CMem.PZRSprayPos}|{PID_out}')
            else:
                # A[0] HV142
                AMod[0] = -1
                if self.CMem.HV142 != 0:
                    self._send_control_save(ActOrderBook['LetdownValveClose'])
                # A[1] PZR spray
                if abs(A[1]) < 0.6:
                    self._send_control_save(ActOrderBook['PZRSprayStay'])
                else:
                    if A[1] < 0:
                        self._send_control_save(ActOrderBook['PZRSprayClose'])
                    else:
                        self._send_control_save(ActOrderBook['PZRSprayOpen'])
            # ----------------------------- Level -------------------------------------------------
            if self.PID_Mode:
                PID_out = self.PID_Lev.update(self.CMem.PZRLevl, 1)
                if PID_out >= 0.005:
                    self._send_control_save(ActOrderBook['ChargingValveOpen'])
                elif -0.005 < PID_out < 0.005:
                    self._send_control_save(ActOrderBook['ChargingValveStay'])
                else:
                    self._send_control_save(ActOrderBook['ChargingValveClose'])
            else:
                # A[2] FV122
                if abs(A[2]) < 0.6:
                    self._send_control_save(ActOrderBook['ChargingValveStay'])
                else:
                    if A[2] < 0: self._send_control_save(ActOrderBook['ChargingValveClose'])
                    else: self._send_control_save(ActOrderBook['ChargingValveOpen'])
            # ----------------------------- ----- -------------------------------------------------
        # =========================================================================================
        # Delta
        if self.CMem.CDelt != 1: self._send_control_save(ActOrderBook['ChangeDelta'])
        # if self.CMem.FV122M != 0: self._send_control_save(ActOrderBook['ChargingAuto'])
        # Done Act
        self._send_control_to_cns()
        return AMod

    def SkipAct(self):
        ActOrderBook = {
            'ChargingValveOpen': (['KSWO101', 'KSWO102'], [0, 1]),
            'ChargingValveStay': (['KSWO101', 'KSWO102'], [0, 0]),
            'ChargingValveClase': (['KSWO101', 'KSWO102'], [1, 0]),

            'LetdownValveOpen': (['KSWO231', 'KSWO232'], [0, 1]),
            'LetdownValveStay': (['KSWO231', 'KSWO232'], [0, 0]),
            'LetdownValveClose': (['KSWO231', 'KSWO232'], [1, 0]),

            'PZRBackHeaterOff': (['KSWO125'], [0]), 'PZRBackHeaterOn': (['KSWO125'], [1]),

            'PZRProHeaterMan': (['KSWO120'], [1]), 'PZRProHeaterAuto': (['KSWO120'], [0]),
            'PZRProHeaterDown': (['KSWO121', 'KSWO122'], [1, 0]),
            'PZRProHeaterUp': (['KSWO121', 'KSWO122'], [0, 1]),
        }
        # Skip or Reset Act
        self._send_control_save(ActOrderBook['LetdownValveStay'])
        # Done Act
        self._send_control_to_cns()
        return 0

    # ENV Main TOOLs ==================================================================================================
    def step(self, A):
        """
        A를 받고 1 step 전진
        :param A: [Act], numpy.ndarry, Act는 numpy.float32
        :return: 최신 state와 reward done 반환
        """
        # Old Data (time t) ---------------------------------------
        AMod = self.send_act(A)
        self.want_tick = int(25)

        # New Data (time t+1) -------------------------------------
        super(ENVCNS, self).step()                  # 전체 CNS mem run-Freeze 하고 mem 업데이트
        self.CMem.update()                          # 선택 변수 mem 업데이트

        self._append_val_to_list()
        self.ENVStep += 1

        reward = self.get_reward(AMod)
        done, reward = self.get_done(reward)
        next_state = self.get_state()
        # ----------------------------------------------------------
        self.ENVlogging(s=self.Loger_txt)
        # self.Loger_txt = f'{next_state}\t'
        self.Loger_txt = ''
        return next_state, reward, done, AMod

    def reset(self, file_name):
        # 1] CNS 상태 초기화 및 초기화된 정보 메모리에 업데이트
        super(ENVCNS, self).reset(initial_nub=21, mal=False, mal_case=0, mal_opt=0, mal_time=0, file_name=file_name)
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


class CNSTestEnv:
    def __init__(self):
        self.env = ENVCNS(Name='Env1', IP='192.168.0.101', PORT=int(f'7101'))

    def run_(self, iter_=1):
        for i in range(1, iter_+1):  # iter = 1 이면 1번 동작
            self.env.reset(file_name=f'Ep{i}')
            start = time.time()
            max_iter = 0
            while True:
                A = 0
                max_iter += 1
                next_state, reward, done, AMod = self.env.step(A, std_=1, mean_=0)
                print(f'Doo--{start}->{time.time()} [{time.time() - start}]')
                if done or max_iter >= 2000:
                    print(f'END--{start}->{time.time()} [{time.time() - start}]')
                    break


if __name__ == '__main__':
    Model = CNSTestEnv()
    Model.run_()