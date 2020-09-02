from COMMONTOOL import TOOL, PTCureve, CSFTree, NBPlot3D
from CNS_UDP_FAST import CNS
import numpy as np
import time
import random
import matplotlib.pylab as plt


class ENVCNS(CNS):
    def __init__(self, Name, IP, PORT):
        super(ENVCNS, self).__init__(threrad_name=Name,
                                     CNS_IP=IP, CNS_Port=PORT, Remote_IP='192.168.0.29', Remote_Port=PORT, Max_len=10)
        self.Name = Name
        self.AcumulatedReward = 0
        self.ENVStep = 0
        self.LoggerPath = 'DB'
        self.want_tick = 15     # 3sec

        self.accident_name = ['LOCA', 'SGTR', 'MSLB'][1]

        self.Loger_txt = ''

        self.input_info = [
            ('ZINST78', 1000)
        ]

        self.action_space = 1
        self.observation_space = len(self.input_info)

        # GP
        self.pl = NBPlot3D()

    # ENV Logger
    def ENVlogging(self, s):
        cr_time = time.strftime('%c', time.localtime(time.time()))
        if self.ENVStep == 0:
            with open(f'{self.Name}.txt', 'a') as f:
                f.write('=='*20 + '\n')
                f.write(f'[{cr_time}]\n')
                f.write('=='*20 + '\n')
        else:
            with open(f'{self.Name}.txt', 'a') as f:
                f.write(f'[{cr_time}] {self.Loger_txt}\n')

    def get_state(self):
        state = [self.mem[para]['Val']/Round_val for para, Round_val in self.input_info]
        # self.Loger_txt += f'{np.array(state)}\t'
        return np.array(state)

    def get_reward(self):
        New_Aux_Flow = self.mem['WAFWS1']['List'][-1]
        Old_Aux_Flow = self.mem['WAFWS1']['List'][-2]

        r = 0
        if self.mem['KLAMPO9']['List'][-1] == 1 and self.mem['KLAMPO134']['List'][-1] == 1:
            if New_Aux_Flow >= 20:
                r += 20
            else:
                r += New_Aux_Flow - 20

        else:
            r -= New_Aux_Flow
        r = r / 25

        self.AcumulatedReward += r

        if self.AcumulatedReward > 400:
            r += 100
        else:
            pass

        # self.Loger_txt += f'{r}\t'
        return r

    def get_done(self):
        if self.AcumulatedReward < -500 or self.AcumulatedReward > 400:
            d = True
        else:
            d = False
        # self.Loger_txt += f'{d}\t'
        return d

    def _send_control_save(self, zipParaVal):
        super(ENVCNS, self)._send_control_save(para=zipParaVal[0], val=zipParaVal[1])

    def send_act(self, A):
        """
        A 에 해당하는 액션을 보내고 나머지는 자동
        E.x)
            self._send_control_save(['KSWO115'], [0])
            ...
            self._send_control_to_cns()
        :param A: A 액션
        :return: AMod: 수정된 액션
        """
        ActOrderBook = {
            'StopAllRCP':       (['KSWO132', 'KSWO133', 'KSWO134'], [0, 0, 0]),
            'StopRCP1':         (['KSWO132'], [0]),
            'StopRCP2':         (['KSWO133'], [0]),
            'StopRCP3':         (['KSWO134'], [0]),
            'NetBRKOpen':       (['KSWO244'], [0]),

            # 강화학습을 위한 제어 변수
            'PZRSprayMan':      (['KSWO128'], [1]), 'PZRSprayAuto':         (['KSWO128'], [0]),
            'PZRSprayClose':    (['KSWO126', 'KSWO127'], [1, 0]),
            'PZRSprayOpen':     (['KSWO126', 'KSWO127'], [0, 1]),
            'PZRBackHeaterOff': (['KSWO125'], [0]), 'PZRBackHeaterOn':      (['KSWO125'], [1]),

            'PZRProHeaterMan':  (['KSWO120'], [1]), 'PZRProHeaterAuto':     (['KSWO120'], [0]),
            'PZRProHeaterDown': (['KSWO121', 'KSWO122'], [1, 0]),
            'PZRProHeaterUp':   (['KSWO121', 'KSWO122'], [0, 1]),

            'DecreaseAux1Flow': (['KSWO142', 'KSWO143'], [1, 0]),
            'IncreaseAux1Flow': (['KSWO142', 'KSWO143'], [0, 1]),
            'DecreaseAux2Flow': (['KSWO151', 'KSWO152'], [1, 0]),
            'IncreaseAux2Flow': (['KSWO151', 'KSWO152'], [0, 1]),
            'DecreaseAux3Flow': (['KSWO154', 'KSWO155'], [1, 0]),
            'IncreaseAux3Flow': (['KSWO154', 'KSWO155'], [0, 1]),
        }
        AMod = A
        # Order Book
        # -------------------------------------------------------------------------------------------------------
        # def check_CSFTree()
        V = {
            # ETC
            'Trip': self.mem['KLAMPO9']['Val'],
            'RCP1': self.mem['KLAMPO124']['Val'],           'RCP2': self.mem['KLAMPO125']['Val'],
            'RCP3': self.mem['KLAMPO126']['Val'],           'NetBRK': self.mem['KLAMPO224']['Val'],
            'CNSTime': self.mem['KCNTOMS']['Val'],

            # 강화학습을 위한 감시 변수
            'PZRSprayManAuto': self.mem['KLAMPO119']['Val'],
            'PZRSprayPos': self.mem['ZINST66']['Val'],
            'PZRBackHeaterManAuto': self.mem['KLAMPO118']['Val'],
            'PZRProHeaterManAuto': self.mem['KLAMPO117']['Val'],
            'PZRProHeaterPos': self.mem['QPRZH']['Val'],

            # CSF 1 Value 미임계 상태 추적도
            'PowerRange': self.mem['ZINST1']['Val'],        'IntermediateRange': self.mem['ZINST2']['Val'],
            'SourceRange': self.mem['ZINST3']['Val'],
            # CSF 2 Value 노심냉각 상태 추적도
            'CoreExitTemp': self.mem['UUPPPL']['Val'],
            'PTCurve': PTCureve().Check(Temp=self.mem['UAVLEG2']['Val'], Pres=self.mem['ZINST65']['Val']),
            # CSF 3 Value 열제거원 상태 추적도
            'SG1Nar': self.mem['ZINST78']['Val'],           'SG2Nar': self.mem['ZINST77']['Val'],
            'SG3Nar': self.mem['ZINST76']['Val'],
            'SG1Pres': self.mem['ZINST75']['Val'],          'SG2Pres': self.mem['ZINST74']['Val'],
            'SG3Pres': self.mem['ZINST73']['Val'],
            'SG1Feed': self.mem['WFWLN1']['Val'],           'SG2Feed': self.mem['WFWLN2']['Val'],
            'SG3Feed': self.mem['WFWLN3']['Val'],

            'AllSGFeed': self.mem['WFWLN1']['Val'] +
                         self.mem['WFWLN2']['Val'] +
                         self.mem['WFWLN3']['Val'],
            'SG1Wid': self.mem['ZINST72']['Val'],           'SG2Wid': self.mem['ZINST71']['Val'],
            'SG3Wid': self.mem['ZINST70']['Val'],
            'SG123Wid': [self.mem['ZINST72']['Val'], self.mem['ZINST71']['Val'], self.mem['ZINST70']['Val']],

            # CSF 4 Value RCS 건전성 상태 추적도
            'RCSColdLoop1': self.mem['UCOLEG1']['List'],    'RCSColdLoop2': self.mem['UCOLEG2']['List'],
            'RCSColdLoop3': self.mem['UCOLEG3']['List'],    'RCSPressure': self.mem['ZINST65']['Val'],
            'CNSTimeL': self.mem['KCNTOMS']['List'],         # PTCurve: ...
            # CSF 5 Value 격납용기 건전성 상태 추적도
            'CTMTPressre': self.mem['ZINST26']['Val'],      'CTMTSumpLevel': self.mem['ZSUMP']['Val'],
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
        CSF_level = [CSF[_]['L'] for _ in CSF.keys()]
        self.Loger_txt += f'{CSF}\t'
        self.Loger_txt += f'{CSF_level}\t'
        DIS_CSF_Info = f"[{V['CNSTime']}] \t"
        # -------------------------------------------------------------------------------------------------------
        # RCP Stop
        if V['Trip'] == 1:
            if V['RCSPressure'] < 97 and V['CNSTime'] < 15 * 60 * 5:
                if V['RCP1'] == 1: self._send_control_save(ActOrderBook['StopRCP1'])
                if V['RCP2'] == 1: self._send_control_save(ActOrderBook['StopRCP2'])
                if V['RCP3'] == 1: self._send_control_save(ActOrderBook['StopRCP3'])
            if V['NetBRK'] == 1: self._send_control_save(ActOrderBook['NetBRKOpen'])

        # -------------------------------------------------------------------------------------------------------
        # CSF 1 Act
        if CSF_level[0] != 0:
            DIS_CSF_Info += f'1: {CSF_level[0]} \t'

        # -------------------------------------------------------------------------------------------------------
        # CSF 2 Act
        if CSF_level[1] != 0:
            DIS_CSF_Info += f'2: {CSF_level[1]} \t'

        # -------------------------------------------------------------------------------------------------------
        # CSF 3 Act
        if CSF_level[2] != 0:
            DIS_CSF_Info += f'3: {CSF_level[2]} \t'

            if CSF_level[2] == 3:   # All Aux <= 33
                if V['AllSGFeed'] <= 33:
                    # 1] Find width low SG nub
                    LowSGNub = V['SG123Wid'].index(min(V['SG123Wid']))   # 0: SG1, 1:SG2, 2:SG3
                    # 2] Supply water to the low SG
                    if LowSGNub == 0: self._send_control_save(ActOrderBook['IncreaseAux1Flow'])
                    if LowSGNub == 1: self._send_control_save(ActOrderBook['IncreaseAux2Flow'])
                    if LowSGNub == 2: self._send_control_save(ActOrderBook['IncreaseAux3Flow'])

        # -------------------------------------------------------------------------------------------------------
        # CSF 4 Act
        if CSF_level[3] != 0:
            DIS_CSF_Info += f'4: {CSF_level[3]} \t'

        # -------------------------------------------------------------------------------------------------------
        # CSF 5 Act
        if CSF_level[4] != 0:
            DIS_CSF_Info += f'5: {CSF_level[4]} \t'

        # -------------------------------------------------------------------------------------------------------
        # CSF 6 Act
        if CSF_level[5] != 0:
            DIS_CSF_Info += f'6: {CSF_level[5]} \t'

        # CSF info DIS
        print(DIS_CSF_Info)

        # -------------------------------------------------------------------------------------------------------
        # Cool Act

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
        self.send_act(A)

        self.pl.plot([self.mem['UAVLEG2']['Val'], self.mem['KCNTOMS']['Val'], self.mem['ZINST65']['Val']])
        # self.pl2.plot([self.mem['KCNTOMS']['Val'], self.mem['KCNTOMS']['Val']])
        # New Data (time t+1) -------------------------------------
        super(ENVCNS, self).step()
        self._append_val_to_list()
        self.ENVStep += 1

        reward = self.get_reward()
        done = self.get_done()
        next_state = self.get_state()

        self.ENVlogging(s=self.Loger_txt)
        # self.Loger_txt = f'{next_state}\t'
        return next_state, reward, done, A

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
        # 5 FIX RADVAL
        self.FixedRad = random.randint(0, 20) * 5
        return state

    # CSF Tree
    def check_CSFTree(self):
        V = {
            # TRIP?
            'Trip': self.mem['KLAMPO9']['Val'],
            # CSF 1 Value 미임계 상태 추적도
            'PowerRange': self.mem['ZINST1']['Val'],        'IntermediateRange': self.mem['ZINST2']['Val'],
            'SourceRange': self.mem['ZINST3']['Val'],
            # CSF 2 Value 노심냉각 상태 추적도
            'CoreExitTemp': self.mem['UUPPPL']['Val'],
            'PTCurve': PTCureve().Check(Temp=self.mem['UAVLEG2']['Val'], Pres=self.mem['ZINST65']['Val']),
            # CSF 3 Value 열제거원 상태 추적도
            'SG1Nar': self.mem['ZINST78']['Val'],           'SG2Nar': self.mem['ZINST77']['Val'],
            'SG3Nar': self.mem['ZINST76']['Val'],
            'SG1Pres': self.mem['ZINST75']['Val'],          'SG2Pres': self.mem['ZINST74']['Val'],
            'SG3Pres': self.mem['ZINST73']['Val'],
            'SG1Feed': self.mem['WFWLN1']['Val'],           'SG2Feed': self.mem['WFWLN2']['Val'],
            'SG3Feed': self.mem['WFWLN3']['Val'],
            # CSF 4 Value RCS 건전성 상태 추적도
            'RCSColdLoop1': self.mem['UCOLEG1']['List'],    'RCSColdLoop2': self.mem['UCOLEG2']['List'],
            'RCSColdLoop3': self.mem['UCOLEG3']['List'],    'RCSPressure': self.mem['ZINST65']['Val'],
            'CNSTime': self.mem['KCNTOMS']['List'],         # PTCurve: ...
            # CSF 5 Value 격납용기 건전성 상태 추적도
            'CTMTPressre': self.mem['ZINST26']['Val'],      'CTMTSumpLevel': self.mem['ZSUMP']['Val'],
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
                                 V['RCSPressure'], V['PTCurve'], V['CNSTime']),
            'CSF5': CSFTree.CSF5(V['Trip'], V['CTMTPressre'], V['CTMTSumpLevel'], V['CTMTRad']),
            'CSF6': CSFTree.CSF6(V['Trip'], V['PZRLevel'])
        }
        self.Loger_txt += f'{CSF}\t'

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

    env.reset(file_name='Ep1')
    while True:
        # A = input(f'{env.ENVStep}A:')
        A = 0
        env.step(int(A))