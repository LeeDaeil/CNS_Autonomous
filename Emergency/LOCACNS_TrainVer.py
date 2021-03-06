from COMMONTOOL import TOOL, PTCureve, CSFTree, NBPlot3D
from CNS_UDP_FAST import CNS
import numpy as np
import time
import random
import copy
import matplotlib.pylab as plt


class ENVCNS(CNS):
    def __init__(self, Name, IP, PORT, Monitoring_ENV = None):
        super(ENVCNS, self).__init__(threrad_name=Name,
                                     CNS_IP=IP, CNS_Port=PORT, Remote_IP='192.168.0.29', Remote_Port=PORT, Max_len=10)
        self.Monitoring_ENV = Monitoring_ENV
        self.Name = Name # = id
        self.AcumulatedReward = 0
        self.ENVStep = 0
        self.LoggerPath = 'DB'
        self.want_tick = 100  # 20sec

        self.accident_name = ['LOCA', 'SGTR', 'MSLB'][1]

        self.Loger_txt = ''

        self.input_info = [
            # (para, x_round, x_min, x_max), (x_min=0, x_max=0 is not normalized.)
            ('ZINST98',  1, 0,   100),     # SteamDumpPos
            ('ZINST87',  1, 0,   50),      # Steam Flow 1
            ('ZINST86',  1, 0,   50),      # Steam Flow 2
            ('ZINST85',  1, 0,   50),      # Steam Flow 3
            ('KLAMPO70', 1, 0,   1),       # Charging Pump2 State
            ('BHV22',    1, 0,   1),       # SI Valve State
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
            ('UCOLEG1',  1, 0,   100),     # RCSColdLoop1
            ('UCOLEG2',  1, 0,   100),     # RCSColdLoop2
            ('UCOLEG3',  1, 0,   100),     # RCSColdLoop3
            ('ZINST65',  1, 0,   160),     # RCSPressure
            ('ZINST63',  1, 0,   100),     # PZRLevel
        ]

        self.action_space = 4       # TODO 1개에서 > 4개 "가압기 제어", "보조급수 제어", "SI 유량 제어", "덤프밸브"
        self.observation_space = len(self.input_info)

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
        state = []
        for para, x_round, x_min, x_max in self.input_info:
            if para in self.mem.keys():
                state.append(self.normalize(self.mem[para]['Val'], x_round, x_min, x_max))
            else:
                # ADD logic ----- 계산된 값을 사용하고 싶을 때
                print('Error_get_state function')
                pass
        # state = [self.mem[para]['Val'] / Round_val for para, Round_val in self.input_info]
        # self.Loger_txt += f'{state}\t'
        return np.array(state)

    def get_reward(self):
        """
        R => -20 ~ 0
        :return:
        """
        # Ver list
        self.Verlist = {
            '1': False,
            '2': False,
            '3': False,
            '4': False,
            '5': True,
        }
        # --------------------------------- NEW ----
        r = 0
        if self.ENVGetSIReset:
            V = {
                'CoolRateTemp': self.DRateFun(self.mem['KCNTOMS']['Val']),
                'CurrentTemp': self.mem['UAVLEG2']['Val'],
                'CurrentPres': self.mem['ZINST65']['Val'],
                'Dis': abs(self.DRateFun(self.mem['KCNTOMS']['Val']) - self.mem['UAVLEG2']['Val']),
                'PZRLevel': self.mem['ZINST63']['Val'],
                'SG1Nar': self.mem['ZINST78']['Val'], 'SG2Nar': self.mem['ZINST77']['Val'],
                'SG3Nar': self.mem['ZINST76']['Val'],
                'SG1Wid': self.mem['ZINST72']['Val'], 'SG2Wid': self.mem['ZINST71']['Val'],
                'SG3Wid': self.mem['ZINST70']['Val'],
                'SG1Pres': self.mem['ZINST75']['Val'], 'SG2Pres': self.mem['ZINST74']['Val'],
                'SG3Pres': self.mem['ZINST73']['Val'],
            }
            if self.Verlist['1']:
                # Cooling rate에 따라서 온도 감소
                r -= V['Dis'] / 100
                # 가압기 수위 10 아래 종료
                # if V['PZRLevel'] <= 10:  r -= 100
            if self.Verlist['2']:
                # 목표치까지 도달
                r += (29.5 - V['CurrentPres']) / 100
                r += (170 - V['CurrentTemp']) / 100
            if self.Verlist['3']:
                # Cooling rate에 따라서 온도 감소
                dis_reward = - V['Dis'] / 100 # [0.0 ~ -0.2] 동향을 보임
                # Pressure and Temp Dis
                curp = 29.5 if V['CurrentPres'] <= 29.5 else V['CurrentPres']
                curt = 170 if V['CurrentTemp'] <= 170 else V['CurrentTemp']
                dis_pres = (29.5 - V['CurrentPres']) / 100
                dis_temp = (170 - V['CurrentTemp']) / 100

                # r += (dis_pres * 0.1) + (dis_temp * 0.1) + (dis_reward * 10) # 감압 X
                r += (dis_pres * 0.1) + (dis_reward * 5)
            if self.Verlist['4']:
                # Cooling rate에 따라서 온도 감소
                dis_reward = - V['Dis'] / 100 # [0.0 ~ -0.2] 동향을 보임
                # Pressure and Temp Dis
                curp = 29.5 if V['CurrentPres'] <= 29.5 else V['CurrentPres']
                dis_pres = (29.5 - V['CurrentPres']) / 100
                PT_reward = - PTCureve().Check(Temp=V['CurrentTemp'], Pres=V['CurrentPres'])
                r += (dis_pres * 0.1) + (dis_reward * 5) + (PT_reward * 0.1)
            if self.Verlist['5']:
                r = 0
                # 1] Cooling rate에 따라서 온도 감소
                coolrate_r = - V['Dis']
                # 2] 가압기 수위 20~76% 구간 초과시 패널티
                pzrlevel_r = 0
                if 20 <= V['PZRLevel'] <= 76:
                    pass
                else:
                    if 20 > V['PZRLevel']:
                        pzrlevel_r -= (20 - V['PZRLevel'])
                    else:
                        pzrlevel_r -= (V['PZRLevel'] - 76)
                # 3] 증기 발생기 6% ~ 50% 이상 초과 시 패널티
                sg_r = 0
                for _ in range(1, 4):
                    if 6 <= V[f'SG{_}Nar'] <= 50:
                        pass
                    else:
                        if 6 > V[f'SG{_}Nar']:
                            sg_r -= (6 - V[f'SG{_}Nar'])
                        else:
                            sg_r -= (V[f'SG{_}Nar'] - 50)
                # 4] PT 커브에서 벗어나면 거리만큼 패널티
                PT_reward = - PTCureve().Check_Dis(Temp=V['CurrentTemp'], Pres=V['CurrentPres'])
                # 5] 목표치와 가까워 질 수록 +
                pres_r, temp_r = 0, 0
                pres_r = (29.5 - V['CurrentPres'])
                temp_r = (170 - V['CurrentTemp'])
                # 6] S/G 압력
                Avg_pres = (V['SG1Pres'] + V['SG2Pres'] + V['SG3Pres'])/3
                SGpres_r = 9 - Avg_pres if Avg_pres > 9 else 0
                # --------------------------------------------------------------
                w = {
                    'coolrate_r': [coolrate_r, 2],
                    'pzrlevel_r': [pzrlevel_r, 1],
                    'sg_r': [sg_r, 1.5],
                    'PT_reward': [PT_reward, 3],
                    'pres_r': [pres_r, 1],
                    'temp_r': [temp_r, 0.5],
                    'SGpres_r': [SGpres_r, 0.5]
                }

                log_txt_temp = ''
                for key in w.keys():
                    r += w[key][0] * w[key][1]
                    log_txt_temp += f'[{round(w[key][0]*w[key][1], 1)}:{w[key][0]}*{w[key][1]}]_'
                log_txt_temp = f'R:{r} = ' + log_txt_temp

                self.Loger_txt += log_txt_temp

            # self.Loger_txt += f"R:{r} = {dis_pres * 0.1}+{dis_temp * 0.1}+({dis_reward * 10})\t"
            # self.Loger_txt += f"R:{r} = {dis_pres * 0.1}+({dis_reward * 5})\t" #Verlist['3']
            # self.Loger_txt += f"R:{r} = {dis_pres * 0.1}+({dis_reward * 5})+({PT_reward * 0.1})\t"

            # --------------------------------- Send R ----
            self.AcumulatedReward += r
        # self.Loger_txt += f'{r}\t'
        self.DIS_CSF_Info += f'[R: {r}]\t'
        return r

    def get_done(self, r):
        V = {
            'CurrentTemp': self.mem['UAVLEG2']['Val'],
            'CurrentPres': self.mem['ZINST65']['Val'],
            'PZRLevel': self.mem['ZINST63']['Val'],
            'SG1Nar': self.mem['ZINST78']['Val'], 'SG2Nar': self.mem['ZINST77']['Val'],
            'SG3Nar': self.mem['ZINST76']['Val'],
            'SG1Pres': self.mem['ZINST75']['Val'], 'SG2Pres': self.mem['ZINST74']['Val'],
            'SG3Pres': self.mem['ZINST73']['Val'],
        }
        if self.Verlist['1']:
            pass
            # if self.AcumulatedReward < -100:
            #     d = True
            # elif self.mem['KCNTOMS']['Val'] == 38000:
            #     d = True
            #     r = 1
            # else:
            #     d = False
        if self.Verlist['2'] or self.Verlist['1']:
            d = False
            # 압력이 너무 아래까지 가는 것을 방지 아래가면 종료
            if V['CurrentPres'] <= 17:
                self.Loger_txt += f'PZR Pres Done\t'
                d = True
                r = -1
            # 증기발생기 수위 78 이상 시 종료
            if max(V['SG1Nar'], V['SG2Nar'], V['SG3Nar']) > 78:
                self.Loger_txt += f'SG Level Done\t'
                d = True
                r= -1
            # PT 커브 범위 초과 시 종료
            if PTCureve().Check(Temp=V['CurrentTemp'], Pres=V['CurrentPres']) == 1: # 불만족
                self.Loger_txt += f'PT Curve\t'
                d = True
                r = -1
            if self.mem['KCNTOMS']['Val'] == 35000:
                d = True
                if 17 < V['CurrentPres'] < 29.5 or V['CurrentTemp'] <= 170:
                    r = 0
                    if 17 < V['CurrentPres'] < 29.5: r += 0.5
                    if V['CurrentTemp'] <= 170: r += 0.5
                else:
                    r = -1
        if self.Verlist['3']:
            d = False
            # 압력이 너무 아래까지 가는 것을 방지 아래가면 종료
            if V['CurrentPres'] <= 17:
                self.Loger_txt += f'PZR Pres Done\t'
                d = True
                r = -1

            # PT 커브 범위 초과 시 종료
            if PTCureve().Check(Temp=V['CurrentTemp'], Pres=V['CurrentPres']) == 1:  # 불만족
                self.Loger_txt += f'PT Curve\t'
                d = True
                r = -1
            if self.mem['KCNTOMS']['Val'] == 35000:
                d = True
                if 17 < V['CurrentPres'] < 29.5 and V['CurrentTemp'] <= 170:
                    r = 1
                else:
                    r = -1
        if self.Verlist['4'] or self.Verlist['5']:
            d = False
            # 압력이 너무 아래까지 가는 것을 방지 아래가면 종료
            if self.mem['KCNTOMS']['Val'] >= 35000:
                d = True
                if 17 < V['CurrentPres'] < 29.5 and V['CurrentTemp'] <= 170:
                    r = 1
                else:
                    r = r
        # self.Loger_txt += f'{d}\t'
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
        # -------------------------------------------------------------------------------------------------------
        # AMod = A[0]  # A는 리스트 값으로 0번째 값 추출
        AMod = A
        # -------------------------------------------------------------------------------------------------------
        # Order Book
        # -------------------------------------------------------------------------------------------------------
        # def check_CSFTree()
        V = {
            # ETC
            'Trip': self.mem['KLAMPO9']['Val'],
            'SIS': self.mem['KLAMPO6']['Val'],
            'MSI': self.mem['KLAMPO3']['Val'],

            'RCP1': self.mem['KLAMPO124']['Val'], 'RCP2': self.mem['KLAMPO125']['Val'],
            'RCP3': self.mem['KLAMPO126']['Val'], 'NetBRK': self.mem['KLAMPO224']['Val'],
            'CNSTime': self.mem['KCNTOMS']['Val'],

            'SteamDumpManAuto': self.mem['KLAMPO150']['Val'],  'SteamDumpPos': self.mem['ZINST98']['Val'],
            'SteamLine1': self.mem['BHV108']['Val'],
            'SteamLine2': self.mem['BHV208']['Val'],
            'SteamLine3': self.mem['BHV308']['Val'],

            'ChargingManAUto': self.mem['KLAMPO95']['Val'],
            'ChargingValvePos': self.mem['BFV122']['Val'],
            'ChargingPump2State': self.mem['KLAMPO70']['Val'],
            'SIValve': self.mem['BHV22']['Val'],

            'TurningGear': self.mem['KLAMPO165']['Val'],
            'OilSys': self.mem['KLAMPO164']['Val'],
            'BHV311': self.mem['BHV311']['Val'],

            # 강화학습을 위한 감시 변수
            'PZRSprayManAuto': self.mem['KLAMPO119']['Val'],
            'PZRSprayPos': self.mem['ZINST66']['Val'],
            'PZRBackHeaterOnOff': self.mem['KLAMPO118']['Val'],
            'PZRProHeaterManAuto': self.mem['KLAMPO117']['Val'],
            'PZRProHeaterPos': self.mem['BHV22']['Val'],

            # CSF 1 Value 미임계 상태 추적도
            'PowerRange': self.mem['ZINST1']['Val'], 'IntermediateRange': self.mem['ZINST2']['Val'],
            'SourceRange': self.mem['ZINST3']['Val'],
            # CSF 2 Value 노심냉각 상태 추적도
            'CoreExitTemp': self.mem['UUPPPL']['Val'],
            'PTCurve': PTCureve().Check(Temp=self.mem['UAVLEG2']['Val'], Pres=self.mem['ZINST65']['Val']),
            # CSF 3 Value 열제거원 상태 추적도
            'SG1Nar': self.mem['ZINST78']['Val'], 'SG2Nar': self.mem['ZINST77']['Val'],
            'SG3Nar': self.mem['ZINST76']['Val'],
            'SG1Pres': self.mem['ZINST75']['Val'], 'SG2Pres': self.mem['ZINST74']['Val'],
            'SG3Pres': self.mem['ZINST73']['Val'],
            'SG1Feed': self.mem['WFWLN1']['Val'], 'SG2Feed': self.mem['WFWLN2']['Val'],
            'SG3Feed': self.mem['WFWLN3']['Val'],

            'AllSGFeed': self.mem['WFWLN1']['Val'] +
                         self.mem['WFWLN2']['Val'] +
                         self.mem['WFWLN3']['Val'],
            'SG1Wid': self.mem['ZINST72']['Val'], 'SG2Wid': self.mem['ZINST71']['Val'],
            'SG3Wid': self.mem['ZINST70']['Val'],
            'SG123Wid': [self.mem['ZINST72']['Val'], self.mem['ZINST71']['Val'], self.mem['ZINST70']['Val']],

            # CSF 4 Value RCS 건전성 상태 추적도
            'RCSColdLoop1': self.mem['UCOLEG1']['List'], 'RCSColdLoop2': self.mem['UCOLEG2']['List'],
            'RCSColdLoop3': self.mem['UCOLEG3']['List'], 'RCSPressure': self.mem['ZINST65']['Val'],
            'CNSTimeL': self.mem['KCNTOMS']['List'],  # PTCurve: ...
            # CSF 5 Value 격납용기 건전성 상태 추적도
            'CTMTPressre': self.mem['ZINST26']['Val'], 'CTMTSumpLevel': self.mem['ZSUMP']['Val'],
            'CTMTRad': self.mem['ZINST22']['Val'],
            # CSF 6 Value RCS 재고량 상태 추적도
            'PZRLevel': self.mem['ZINST63']['Val']
        }
        CSF = {
            'CSF1': CSFTree.CSF1(V['Trip'], V['PowerRange'], V['IntermediateRange'], V['SourceRange']),
            'CSF2': CSFTree.CSF2(V['Trip'], V['CoreExitTemp'], V['PTCurve']),
            'CSF3': CSFTree.CSF3(V['Trip'], V['SG1Nar'], V['SG2Nar'], V['SG3Nar'],
                                 V['SG1Pres'], V['SG2Pres'], V['SG3Pres'],
                                 V['SG1Feed'], V['SG2Feed'], V['SG3Feed']),
            'CSF4': CSFTree.CSF4(V['Trip'], V['RCSColdLoop1'], V['RCSColdLoop2'], V['RCSColdLoop3'],
                                 V['RCSPressure'], V['PTCurve'], V['CNSTimeL']),
            'CSF5': CSFTree.CSF5(V['Trip'], V['CTMTPressre'], V['CTMTSumpLevel'], V['CTMTRad']),
            'CSF6': CSFTree.CSF6(V['Trip'], V['PZRLevel'])
        }
        ActOrderBook = {
            'StopAllRCP': (['KSWO132', 'KSWO133', 'KSWO134'], [0, 0, 0]),
            'StopRCP1': (['KSWO132'], [0]),
            'StopRCP2': (['KSWO133'], [0]),
            'StopRCP3': (['KSWO134'], [0]),
            'NetBRKOpen': (['KSWO244'], [0]),
            'OilSysOff': (['KSWO190'], [0]),
            'TurningGearOff': (['KSWO191'], [0]),
            'CutBHV311': (['BHV311', 'FKAFWPI'], [0, 0]),

            # 강화학습을 위한 제어 변수
            'PZRSprayMan': (['KSWO128'], [1]), 'PZRSprayAuto': (['KSWO128'], [0]),

            'PZRSprayClose': (['BPRZSP'], [self.mem['BPRZSP']['Val'] + 0.015 * -1]),
            'PZRSprayOpen': (['BPRZSP'], [self.mem['BPRZSP']['Val'] + 0.015 * 1]),

            'PZRBackHeaterOff': (['KSWO125'], [0]), 'PZRBackHeaterOn': (['KSWO125'], [1]),

            'PZRProHeaterMan': (['KSWO120'], [1]), 'PZRProHeaterAuto': (['KSWO120'], [0]),
            'PZRProHeaterDown': (['KSWO121', 'KSWO122'], [1, 0]),
            'PZRProHeaterUp': (['KSWO121', 'KSWO122'], [0, 1]),

            'DecreaseAux1Flow': (['KSWO142', 'KSWO143'], [1, 0]),
            'IncreaseAux1Flow': (['KSWO142', 'KSWO143'], [0, 1]),
            'DecreaseAux2Flow': (['KSWO151', 'KSWO152'], [1, 0]),
            'IncreaseAux2Flow': (['KSWO151', 'KSWO152'], [0, 1]),
            'DecreaseAux3Flow': (['KSWO154', 'KSWO155'], [1, 0]),
            'IncreaseAux3Flow': (['KSWO154', 'KSWO155'], [0, 1]),

            'UpAllAux': (['KSWO142', 'KSWO143', 'KSWO151', 'KSWO152', 'KSWO154', 'KSWO155'], [0, 1, 0, 1, 0, 1]),
            'DownAllAux': (['KSWO142', 'KSWO143', 'KSWO151', 'KSWO152', 'KSWO154', 'KSWO155'], [1, 0, 1, 0, 1, 0]),

            # WAFWS1
            'RL_IncreaseAux1Flow': (['WAFWS1'], [self.mem['WAFWS1']['Val'] + 0.04 * 1]),
            'RL_DecreaseAux1Flow': (['WAFWS1'], [self.mem['WAFWS1']['Val'] + 0.04 * (-1)]),
            'RL_IncreaseAux2Flow': (['WAFWS2'], [self.mem['WAFWS2']['Val'] + 0.04 * 1]),
            'RL_DecreaseAux2Flow': (['WAFWS2'], [self.mem['WAFWS2']['Val'] + 0.04 * (-1)]),
            'RL_IncreaseAux3Flow': (['WAFWS3'], [self.mem['WAFWS3']['Val'] + 0.04 * 1]),
            'RL_DecreaseAux3Flow': (['WAFWS3'], [self.mem['WAFWS3']['Val'] + 0.04 * (-1)]),

            'SteamDumpMan': (['KSWO176'], [1]), 'SteamDumpAuto': (['KSWO176'], [0]),
            'SteamDumpUp': (['PMSS'], [self.mem['PMSS']['Val'] + 2.0E5 * 1 * 0.2]),
            'SteamDumpDown': (['PMSS'], [self.mem['PMSS']['Val'] + 2.0E5 * (-1) * 0.2]),

            'IFLOGIC_SteamDumpUp': (['PMSS'], [self.mem['PMSS']['Val'] + 2.0E5 * 3 * 0.2]),
            'IFLOGIC_SteamDumpDown': (['PMSS'], [self.mem['PMSS']['Val'] + 2.0E5 * (-3) * 0.2]),

            'SteamLine1Open': (['KSWO148', 'KSWO149'], [1, 0]),
            'SteamLine2Open': (['KSWO146', 'KSWO147'], [1, 0]),
            'SteamLine3Open': (['KSWO144', 'KSWO145'], [1, 0]),

            'ChargingValveMan': (['KSWO100'], [1]), 'ChargingValveAUto': (['KSWO100'], [0]),
            'ChargingValveDown': (['KSWO101', 'KSWO102'], [1, 0]),
            'ChargingValveUp': (['KSWO101', 'KSWO102'], [0, 1]),

            'RunRCP2': (['KSWO130', 'KSWO133'], [1, 1]),
            'RunCHP2': (['KSWO70'], [1]), 'StopCHP2': (['KSWO70'], [0]),
            'OpenSI': (['KSWO81', 'KSWO82'], [1, 0]), 'CloseSI': (['KSWO81', 'KSWO82'], [0, 1]),

            'ResetSI': (['KSWO7', 'KSWO5'], [1, 1]),
        }
        CSF_level = [CSF[_]['L'] for _ in CSF.keys()]
        # self.Loger_txt += f'{CSF}\t'
        # self.Loger_txt += f'{CSF_level}\t'
        self.DIS_CSF_Info = f"[{V['CNSTime']}]\t"
        # -------------------------------------------------------------------------------------------------------
        # 자동 액션
        if V['Trip'] == 1:
            # 1] RCP 97 압력 이하에서 정지
            if V['RCSPressure'] < 97 and V['CNSTime'] < 15 * 60 * 5:
                if V['RCP1'] == 1: self._send_control_save(ActOrderBook['StopRCP1'])
                if V['RCP2'] == 1: self._send_control_save(ActOrderBook['StopRCP2'])
                if V['RCP3'] == 1: self._send_control_save(ActOrderBook['StopRCP3'])
            if V['NetBRK'] == 1: self._send_control_save(ActOrderBook['NetBRKOpen'])
            if V['TurningGear'] == 1: self._send_control_save(ActOrderBook['TurningGearOff'])
            if V['OilSys'] == 1: self._send_control_save(ActOrderBook['OilSysOff'])
            if V['BHV311'] > 0: self._send_control_save(ActOrderBook['CutBHV311'])

            # 1.1] Setup 현재 최대 압력 기준으로 세팅.
            if V['SIS'] != 0 and V['MSI'] != 0:
                if max(V['SG1Pres'], V['SG2Pres'], V['SG3Pres']) < V['SteamDumpPos']:
                    self._send_control_save(ActOrderBook['IFLOGIC_SteamDumpDown'])
            # 1.2] SI reset 전에 Aux 평균화 [검증 완료 20200903]
                if V['SG1Feed'] == V['SG2Feed'] and V['SG1Feed'] == V['SG3Feed'] and \
                        V['SG2Feed'] == V['SG1Feed'] and V['SG2Feed'] == V['SG3Feed'] and \
                        V['SG3Feed'] == V['SG1Feed'] and V['SG3Feed'] == V['SG2Feed']:
                    pass
                else:
                    # 1.2.1] 급수 일정화 수행
                    # 1.2.1.1] 가장 큰 급수 찾기
                    SGFeedList = [V['SG1Feed'], V['SG2Feed'], V['SG3Feed']]
                    MaxSGFeed = SGFeedList.index(max(SGFeedList))  # 0, 1, 2
                    MinSGFeed = SGFeedList.index(min(SGFeedList))  # 0, 1, 2
                    self._send_control_save(ActOrderBook[f'DecreaseAux{MaxSGFeed + 1}Flow'])
                    self._send_control_save(ActOrderBook[f'IncreaseAux{MinSGFeed + 1}Flow'])

            # 1.3] 2000부터 SI reset
            if V['CNSTime'] == 3000:                self._send_control_save(ActOrderBook['ResetSI'])

            # 2] SI reset 발생 시 (냉각 운전 시작)
            if V['SIS'] == 0 and V['MSI'] == 0 and V['CNSTime'] > 5 * 60 * 5:
                # 2.0] Build Cooling rate function
                if self.ENVGetSIReset == False:
                    rate = - 50 / (60 * 60 * 5) # 55
                    self.FixedTemp = copy.deepcopy(self.mem['UAVLEG2']['Val'])  # 변수의 Follow up 방지
                    self.FixedTime = copy.deepcopy(self.mem['KCNTOMS']['Val'])  # 변수의 Follow up 방지
                    self.DRateFun = lambda t: rate * (t - self.FixedTime) + self.FixedTemp
                    self.ENVGetSIReset = True

                # 2.0] Press set-point 를 현재 최대 압력 기준까지 조절
                # Steam dump Auto
                if V['SteamDumpManAuto'] == 0:      self._send_control_save(ActOrderBook['SteamDumpMan'])
                # 변수 업!
                # if max(V['SG1Pres'], V['SG2Pres'], V['SG3Pres']) < V['SteamDumpPos']:
                #     self._send_control_save(ActOrderBook['SteamDumpDown'])

                # Steam Line up
                if V['SteamLine1'] == 0:            self._send_control_save(ActOrderBook['SteamLine1Open'])
                if V['SteamLine2'] == 0:            self._send_control_save(ActOrderBook['SteamLine2Open'])
                if V['SteamLine3'] == 0:            self._send_control_save(ActOrderBook['SteamLine3Open'])

                # 2.1] Charging flow 최소화
                if V['ChargingManAUto'] == 0:       self._send_control_save(ActOrderBook['ChargingValveMan'])
                if V['ChargingValvePos'] != 0:      self._send_control_save(ActOrderBook['ChargingValveDown'])
                # 2.2] PZR spray 동작 [감압]
                if V['PZRSprayManAuto'] == 0:       self._send_control_save(ActOrderBook['PZRSprayMan'])
                if V['RCP2'] == 0:                  self._send_control_save(ActOrderBook['RunRCP2'])
                # 2.3] PZR 감압을 위한 Heater 종료
                if V['PZRProHeaterManAuto'] == 0:   self._send_control_save(ActOrderBook['PZRProHeaterMan'])
                if V['PZRProHeaterPos'] >= 0:       self._send_control_save(ActOrderBook['PZRProHeaterDown'])
                if V['PZRBackHeaterOnOff'] == 1:    self._send_control_save(ActOrderBook['PZRBackHeaterOff'])

                # -------------------------------------------------------------------------------------------
                # 강화학습 제어 파트
                # 2.4] TEST 모듈
                TEST = False
                if TEST:
                    # Check ACT input
                    while True:
                        ACT = input("Want act:")
                        if ACT == '' or ACT == '0':
                            # print("[SEND] No Act")
                            ACT = int(0)
                            break
                        if ACT.isdigit():
                            # print(f"[SEND] Act {int(ACT)}")
                            ACT = int(ACT)
                            break
                    # Done ACT input
                    # --------------
                    # Start ACT Code
                    if ACT == 0: pass
                    if ACT == 1: self._send_control_save(ActOrderBook['PZRSprayOpen'])
                    if ACT == 2: self._send_control_save(ActOrderBook['PZRSprayClose'])
                    if ACT == 3: self._send_control_save(ActOrderBook['UpAllAux'])
                    if ACT == 4: self._send_control_save(ActOrderBook['DownAllAux'])
                    if ACT == 5: self._send_control_save(ActOrderBook['RunCHP2'])
                    if ACT == 6: self._send_control_save(ActOrderBook['StopCHP2'])
        # 강화학습 제어 파트 ---------------------------------------------------------------------------------------
        V6_2_1 = False
        V6_2_2 = False
        V6_2_3 = False
        V6_2_4 = True
        if V6_2_1:  # TODO 돌리려면 AMod = Act[0] 하고 마지막 return [AMod] 써야함.
            if V['Trip'] == 1 and V['SIS'] == 0 and V['MSI'] == 0 and V['CNSTime'] > 5 * 60 * 5:
                # 1] 선택한 액션의 적합성을 판단하고 램덤으로 재 샘플링
                if AMod == 0:
                    pass
                else:
                    while True:
                        AMod = random.randint(0, 6)/10      # [0 ... 0.6]
                        if AMod == 0: break
                        if AMod == 0.1 and V['PZRSprayPos'] < 25.0: break
                        if AMod == 0.2 and V['PZRSprayPos'] != 0: break
                        if AMod == 0.3 and V['AllSGFeed'] < 73: break
                        if AMod == 0.4:
                            if not V['AllSGFeed'] == 0:
                                if V['SG1Nar'] > 6 and V['SG2Nar'] > 6 and V['SG3Nar'] > 6:
                                    break
                        if AMod == 0.5:
                            if V['PZRLevel'] > 40: pass
                            else:
                                if V['ChargingPump2State'] == 1 and V['SIValve'] == 1: pass
                                elif V['ChargingPump2State'] == 1 and V['SIValve'] == 0: break
                                elif V['ChargingPump2State'] == 0 and V['SIValve'] == 0: break
                        if AMod == 0.6:
                            if V['PZRLevel'] < 18: pass
                            else:
                                if V['ChargingPump2State'] == 0 and V['SIValve'] == 0: pass
                                elif V['ChargingPump2State'] == 0 and V['SIValve'] == 1: break
                                elif V['ChargingPump2State'] == 1 and V['SIValve'] == 1: break
                # 2] 최종적으로 제어 보내기
                if AMod == 0: pass
                if AMod == 0.1: self._send_control_save(ActOrderBook['PZRSprayOpen'])
                if AMod == 0.2: self._send_control_save(ActOrderBook['PZRSprayClose'])
                if AMod == 0.3: self._send_control_save(ActOrderBook['UpAllAux'])
                if AMod == 0.4: self._send_control_save(ActOrderBook['DownAllAux'])
                if AMod == 0.5:
                    if V['PZRLevel'] > 40:
                        AMod = 0
                    else:
                        if V['ChargingPump2State'] == 1 and V['SIValve'] == 1:
                            AMod = 0
                        elif V['ChargingPump2State'] == 1 and V['SIValve'] == 0:
                            self._send_control_save(ActOrderBook['RunCHP2'])
                        elif V['ChargingPump2State'] == 0 and V['SIValve'] == 0:
                            self._send_control_save(ActOrderBook['OpenSI'])
                if AMod == 0.6:
                    if V['PZRLevel'] < 18:
                        AMod = 0
                    else:
                        if V['ChargingPump2State'] == 0 and V['SIValve'] == 0:
                            AMod = 0
                        elif V['ChargingPump2State'] == 0 and V['SIValve'] == 1:
                            self._send_control_save(ActOrderBook['CloseSI'])
                        elif V['ChargingPump2State'] == 1 and V['SIValve'] == 1:
                            self._send_control_save(ActOrderBook['StopCHP2'])
            else:
                AMod = 0
        if V6_2_2:
            if V['Trip'] == 1 and V['SIS'] == 0 and V['MSI'] == 0 and V['CNSTime'] > 5 * 60 * 5:
                # 1] Continuous 값을 Discrete 값으로 변경
                AMod_ = [] # [1, 0, 0] 형식으로 저장됨
                for A in AMod:
                    if A >= 0.4:
                        AMod_.append(1)
                    elif -0.4 <= A < 0.4:
                        AMod_.append(0)
                    else:
                        AMod_.append(-1)
                # AMod = AMod_
                # 1.1] 가압기 10 아래면 무조건 CHP 2 on, HV Open
                if V['PZRLevel'] <= 10:
                    AMod[2] = 1

                # 2] 액션 스페이스는 줄이지 않고 수행해보기
                if AMod[0] == 1:
                    if V['PZRSprayPos'] >= 35:
                        AMod[0] = 0
                    else:
                        self._send_control_save(ActOrderBook['PZRSprayOpen'])
                if AMod[0] == 0: pass
                if AMod[0] == -1:
                    if V['PZRSprayPos'] == 0:
                        AMod[0] = 0
                    else:
                        self._send_control_save(ActOrderBook['PZRSprayClose'])

                if AMod[1] == 1:
                    if max(V['SG1Nar'], V['SG2Nar'], V['SG3Nar']) > 50:
                        AMod[1] = -1
                    else:
                        self._send_control_save(ActOrderBook['UpAllAux'])
                if AMod[1] == 0: pass
                if AMod[1] == -1:
                    if V['AllSGFeed'] == 0:
                        AMod[1] = 0
                    else:
                        self._send_control_save(ActOrderBook['DownAllAux'])

                if AMod[2] == 1:
                    if V['PZRLevel'] > 76:  ## 가압기 만수위 방지용.
                        AMod[2] = -1
                    else:
                        if V['ChargingPump2State'] == 1 and V['SIValve'] == 1:
                            AMod[2] = 0
                        elif V['ChargingPump2State'] == 0 and V['SIValve'] == 1:
                            self._send_control_save(ActOrderBook['RunCHP2'])
                        elif V['ChargingPump2State'] == 0 and V['SIValve'] == 0:
                            self._send_control_save(ActOrderBook['OpenSI'])
                if AMod[2] == 0: pass
                if AMod[2] == -1:
                    if V['ChargingPump2State'] == 0 and V['SIValve'] == 0:
                        AMod[2] = 0
                    elif V['ChargingPump2State'] == 0 and V['SIValve'] == 1:
                        self._send_control_save(ActOrderBook['CloseSI'])
                    elif V['ChargingPump2State'] == 1 and V['SIValve'] == 1:
                        self._send_control_save(ActOrderBook['StopCHP2'])

                if AMod[3] == 1:
                    if max(V['SG1Pres'], V['SG2Pres'], V['SG3Pres']) < V['SteamDumpPos']:
                        self._send_control_save(ActOrderBook['SteamDumpDown'])
                        AMod[3] == -1
                    else:
                        self._send_control_save(ActOrderBook['SteamDumpUp'])
                if AMod[3] == 0: pass
                if AMod[3] == -1: self._send_control_save(ActOrderBook['SteamDumpDown'])
            else:
                AMod = [0, 0, 0, 0]
        if V6_2_3:
            if V['Trip'] == 1 and V['SIS'] == 0 and V['MSI'] == 0 and V['CNSTime'] > 5 * 60 * 5:
                # 1] Continuous 값을 Discrete 값으로 변경
                # AMod_ = []  # [1, 0, 0] 형식으로 저장됨
                # for A in AMod:
                #     if A >= 0.4:
                #         AMod_.append(1)
                #     elif -0.4 <= A < 0.4:
                #         AMod_.append(0)
                #     else:
                #         AMod_.append(-1)
                # AMod = AMod_
                # 1.1] 가압기 10 아래면 무조건 CHP 2 on, HV Open
                if V['PZRLevel'] <= 10:
                    if AMod[2] < 0.4:  AMod[2] = 0.8
                if V['PZRLevel'] > 76:
                    if -0.4 <= AMod[2]: AMod[2] = -0.8

                # 2] 액션 스페이스는 줄이지 않고 수행해보기
                if 0.4 < AMod[0]:
                    if V['PZRSprayPos'] >= 35:
                        AMod[0] = 0
                    else:
                        self._send_control_save(ActOrderBook['PZRSprayOpen'])
                if -0.4 <= AMod[0] <= 0.4: pass
                if AMod[0] < -0.4:
                    if V['PZRSprayPos'] == 0:
                        AMod[0] = 0
                    else:
                        self._send_control_save(ActOrderBook['PZRSprayClose'])

                if 0.4 < AMod[1]:
                    if max(V['SG1Nar'], V['SG2Nar'], V['SG3Nar']) > 50:
                        AMod[1] = -0.8
                    else:
                        self._send_control_save(ActOrderBook['UpAllAux'])
                if -0.4 <= AMod[1] <= 0.4: pass
                if AMod[1] < -0.4:
                    if V['AllSGFeed'] == 0:
                        AMod[1] = 0
                    else:
                        self._send_control_save(ActOrderBook['DownAllAux'])

                if 0.4 < AMod[2]:
                    if V['ChargingPump2State'] == 1 and V['SIValve'] == 1:
                        AMod[2] = 0
                    elif V['ChargingPump2State'] == 0 and V['SIValve'] == 1:
                        self._send_control_save(ActOrderBook['RunCHP2'])
                    elif V['ChargingPump2State'] == 0 and V['SIValve'] == 0:
                        self._send_control_save(ActOrderBook['OpenSI'])
                if -0.4 <= AMod[2] <= 0.4: pass
                if AMod[2] < -0.4:
                    if V['ChargingPump2State'] == 0 and V['SIValve'] == 0:
                        AMod[2] = 0
                    elif V['ChargingPump2State'] == 0 and V['SIValve'] == 1:
                        self._send_control_save(ActOrderBook['CloseSI'])
                    elif V['ChargingPump2State'] == 1 and V['SIValve'] == 1:
                        self._send_control_save(ActOrderBook['StopCHP2'])

                if 0.4 < AMod[3]:
                    self._send_control_save(ActOrderBook['SteamDumpUp'])
                if -0.4 <= AMod[3] <= 0.4: pass
                if AMod[3] < -0.4:
                    self._send_control_save(ActOrderBook['SteamDumpDown'])
            else:
                AMod = [0, 0, 0, 0]
        if V6_2_4:
            if V['Trip'] == 1 and V['SIS'] == 0 and V['MSI'] == 0 and V['CNSTime'] > 5 * 60 * 5:

                # 1] Spray Control
                pos = self.mem['BPRZSP']['Val'] + 0.015 * np.clip(AMod[0] * 2, -2, 2)
                zip_spray_pos = (['BPRZSP'], [pos])
                self._send_control_save(zip_spray_pos)

                # 2] Aux Feed
                if AMod[1] < -0.8:
                    self._send_control_save(ActOrderBook['RL_DecreaseAux1Flow'])
                elif -0.8 <= AMod[1] < -0.6:
                    self._send_control_save(ActOrderBook['RL_DecreaseAux2Flow'])
                elif -0.6 <= AMod[1] < -0.4:
                    self._send_control_save(ActOrderBook['RL_DecreaseAux3Flow'])
                elif -0.4 <= AMod[1] < 0.4:
                    pass
                elif 0.4 <= AMod[1] < 0.6:
                    self._send_control_save(ActOrderBook['RL_IncreaseAux3Flow'])
                elif 0.6 <= AMod[1] < 0.8:
                    self._send_control_save(ActOrderBook['RL_IncreaseAux2Flow'])
                elif 0.8 <= AMod[1]:
                    self._send_control_save(ActOrderBook['RL_IncreaseAux1Flow'])

                # 3] SI Supply water
                if AMod[2] < -0.8:
                    self._send_control_save(ActOrderBook['CloseSI'])
                elif -0.8 <= AMod[2] < -0.6:
                    self._send_control_save(ActOrderBook['StopCHP2'])
                elif -0.6 <= AMod[2] < 0.6:
                    pass
                elif 0.6 <= AMod[2] < 0.8:
                    self._send_control_save(ActOrderBook['RunCHP2'])
                elif 0.8 <= AMod[2]:
                    self._send_control_save(ActOrderBook['OpenSI'])

                # 4] Steam Dump
                SteamDumpRate = 4
                DumpPos = self.mem['PMSS']['Val'] + 2.0E5 * np.clip(AMod[3] * SteamDumpRate,
                                                                    - SteamDumpRate, SteamDumpRate) * 0.2
                zip_Dump_pos = (['PMSS'],[DumpPos])
                self._send_control_save(zip_Dump_pos)

            else:
                pass #AMod = [0, 0, 0, 0]
        self.DIS_CSF_Info += f'[A:{AMod}]\t'
        self.Loger_txt += f'{AMod}\t'
        # -------------------------------------------------------------------------------------------------------
        # CSF 1 Act
        if CSF_level[0] != 0: self.DIS_CSF_Info += f'1: {CSF_level[0]} \t'
        # -------------------------------------------------------------------------------------------------------
        # CSF 2 Act
        if CSF_level[1] != 0: self.DIS_CSF_Info += f'2: {CSF_level[1]} \t'
        # -------------------------------------------------------------------------------------------------------
        # CSF 3 Act
        if CSF_level[2] != 0:
            self.DIS_CSF_Info += f'3: {CSF_level[2]} \t'
            if CSF_level[2] == 3:  # All Aux <= 33
                if V['AllSGFeed'] <= 33:
                    # 1] Find width low SG nub
                    LowSGNub = V['SG123Wid'].index(min(V['SG123Wid']))  # 0: SG1, 1:SG2, 2:SG3
                    # 2] Supply water to the low SG
                    if LowSGNub == 0: self._send_control_save(ActOrderBook['IncreaseAux1Flow'])
                    if LowSGNub == 1: self._send_control_save(ActOrderBook['IncreaseAux2Flow'])
                    if LowSGNub == 2: self._send_control_save(ActOrderBook['IncreaseAux3Flow'])
        # -------------------------------------------------------------------------------------------------------
        # CSF 4 Act
        if CSF_level[3] != 0: self.DIS_CSF_Info += f'4: {CSF_level[3]} \t'
        # -------------------------------------------------------------------------------------------------------
        # CSF 5 Act
        if CSF_level[4] != 0: self.DIS_CSF_Info += f'5: {CSF_level[4]} \t'
        # -------------------------------------------------------------------------------------------------------
        # CSF 6 Act
        if CSF_level[5] != 0: self.DIS_CSF_Info += f'6: {CSF_level[5]} \t'
        # -------------------------------------------------------------------------------------------------------
        # CSF info DIS
        # print(self.DIS_CSF_Info)
        # -------------------------------------------------------------------------------------------------------
        # Done Act
        self._send_control_to_cns()
        return AMod

    def step(self, A):
        """
        A를 받고 1 step 전진
        :param A: [Act], numpy.ndarry, Act는 numpy.float32
        :return: 최신 state와 reward done 반환
        """
        # Old Data (time t) ---------------------------------------
        # self.check_CSFTree()
        AMod = self.send_act(A)
        V = {
            # ETC
            'Trip': self.mem['KLAMPO9']['Val'],
            'SIS': self.mem['KLAMPO6']['Val'],
            'MSI': self.mem['KLAMPO3']['Val'],

            'RCP1': self.mem['KLAMPO124']['Val'], 'RCP2': self.mem['KLAMPO125']['Val'],
            'RCP3': self.mem['KLAMPO126']['Val'], 'NetBRK': self.mem['KLAMPO224']['Val'],
            'CNSTime': self.mem['KCNTOMS']['Val'],

            'SteamDumpManAuto': self.mem['KLAMPO150']['Val'], 'SteamDumpPos': self.mem['ZINST98']['Val'],
            'SteamLine1': self.mem['BHV108']['Val'],
            'SteamLine2': self.mem['BHV208']['Val'],
            'SteamLine3': self.mem['BHV308']['Val'],

            'ChargingManAUto': self.mem['KLAMPO95']['Val'],
            'ChargingValvePos': self.mem['BFV122']['Val'],
            'ChargingPump2State': self.mem['KLAMPO70']['Val'],
            'SIValve': self.mem['BHV22']['Val'],

            # 강화학습을 위한 감시 변수
            'PZRSprayManAuto': self.mem['KLAMPO119']['Val'],
            'PZRSprayPos': self.mem['ZINST66']['Val'],
            'PZRBackHeaterOnOff': self.mem['KLAMPO118']['Val'],
            'PZRProHeaterManAuto': self.mem['KLAMPO117']['Val'],
            'PZRProHeaterPos': self.mem['BHV22']['Val'],

            # CSF 1 Value 미임계 상태 추적도
            'PowerRange': self.mem['ZINST1']['Val'], 'IntermediateRange': self.mem['ZINST2']['Val'],
            'SourceRange': self.mem['ZINST3']['Val'],
            # CSF 2 Value 노심냉각 상태 추적도
            'CoreExitTemp': self.mem['UUPPPL']['Val'],
            'PTCurve': PTCureve().Check(Temp=self.mem['UAVLEG2']['Val'], Pres=self.mem['ZINST65']['Val']),
            # CSF 3 Value 열제거원 상태 추적도
            'SG1Nar': self.mem['ZINST78']['Val'], 'SG2Nar': self.mem['ZINST77']['Val'],
            'SG3Nar': self.mem['ZINST76']['Val'],
            'SG1Pres': self.mem['ZINST75']['Val'], 'SG2Pres': self.mem['ZINST74']['Val'],
            'SG3Pres': self.mem['ZINST73']['Val'],
            'SG1Feed': self.mem['WFWLN1']['Val'], 'SG2Feed': self.mem['WFWLN2']['Val'],
            'SG3Feed': self.mem['WFWLN3']['Val'],

            'AllSGFeed': self.mem['WFWLN1']['Val'] +
                         self.mem['WFWLN2']['Val'] +
                         self.mem['WFWLN3']['Val'],
            'SG1Wid': self.mem['ZINST72']['Val'], 'SG2Wid': self.mem['ZINST71']['Val'],
            'SG3Wid': self.mem['ZINST70']['Val'],
            'SG123Wid': [self.mem['ZINST72']['Val'], self.mem['ZINST71']['Val'], self.mem['ZINST70']['Val']],

            # CSF 4 Value RCS 건전성 상태 추적도
            'RCSColdLoop1': self.mem['UCOLEG1']['List'], 'RCSColdLoop2': self.mem['UCOLEG2']['List'],
            'RCSColdLoop3': self.mem['UCOLEG3']['List'], 'RCSPressure': self.mem['ZINST65']['Val'],
            'CNSTimeL': self.mem['KCNTOMS']['List'],  # PTCurve: ...
            # CSF 5 Value 격납용기 건전성 상태 추적도
            'CTMTPressre': self.mem['ZINST26']['Val'], 'CTMTSumpLevel': self.mem['ZSUMP']['Val'],
            'CTMTRad': self.mem['ZINST22']['Val'],
            # CSF 6 Value RCS 재고량 상태 추적도
            'PZRLevel': self.mem['ZINST63']['Val']
        }
        if V['Trip'] == 1 and V['SIS'] == 0 and V['MSI'] == 0 and V['CNSTime'] > 5 * 60 * 5:

        # GOTICK = input("Want Tick : ")
            self.want_tick = int(200)

        self.Monitoring_ENV.push_ENV_val(i=self.Name,
                                         Dict_val={f'{Para}': self.mem[f'{Para}']['Val'] for Para in
                                                   ['UAVLEG2', 'KCNTOMS', 'ZINST65', 'KLAMPO6', 'KLAMPO9',
                                                    'ZINST63', 'WFWLN1', 'WFWLN2', 'WFWLN3']}
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
        # -------------------------------------------------------------------------------------------------------
        print(self.DIS_CSF_Info)
        # -------------------------------------------------------------------------------------------------------
        return next_state, reward, done, AMod

    def reset(self, file_name):
        # 1] CNS 상태 초기화 및 초기화된 정보 메모리에 업데이트
        super(ENVCNS, self).reset(initial_nub=1, mal=True, mal_case=12, mal_opt=10005, mal_time=30, file_name=file_name)
        # 2] 업데이트된 'Val'를 'List'에 추가 및 ENVLogging 초기화
        self._append_val_to_list()
        self.ENVlogging('')
        # 3] 'Val'을 상태로 제작후 반환
        state = self.get_state()
        # 4] 보상 누적치 및 ENVStep 초기화
        self.AcumulatedReward = 0
        self.ENVStep = 0
        self.ENVGetSIReset = False
        self.Monitoring_ENV.init_ENV_val(self.Name)
        # 5] FIX RADVAL
        self.FixedRad = random.randint(0, 20) * 5
        self.FixedTime = 0
        self.FixedTemp = 0
        # 6] 초반에 제어의 의미가 없어 보임. 실제 냉각이 필요한 제어부분까지 진행
        while True:
            if self.mem['KLAMPO9']['Val'] == 1 and self.mem['KLAMPO6']['Val'] == 0 \
                    and self.mem['KLAMPO3']['Val'] == 0 and self.mem['KCNTOMS']['Val'] > 5 * 60 * 5:
                break
            else:
                state, _, _, _ = self.step([0, 0, 0])
        return state


if __name__ == '__main__':
    # ENVCNS TEST
    env = ENVCNS(Name='Env1', IP='192.168.0.103', PORT=int(f'7101'))
    # Modify input
    env.input_info = [
        ('ZINST78', 1000),
        ('ZINST77', 200),
        ('ZINST76', 100),
    ]
    env.observation_space = 3

    for _ in range(1, 4):
        env.reset(file_name=f'Ep{_}')
        for __ in range(3):
            A = 0
            env.step(int(A))

    # while True:
    #     # A = input(f'{env.ENVStep}A:')
    #     A = 0
    #     env.step(int(A))
