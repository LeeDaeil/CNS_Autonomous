from COMMONTOOL import TOOL
from CNS_UDP_FAST import CNS
import numpy as np
import time
import random


class ENVCNS(CNS):
    def __init__(self, Name, IP, PORT):
        super(ENVCNS, self).__init__(threrad_name=Name,
                                     CNS_IP=IP, CNS_Port=PORT, Remote_IP='192.168.0.10', Remote_Port=PORT, Max_len=10)
        self.Name = Name
        self.AcumulatedReward = 0
        self.ENVStep = 0

        self.accident_name = ['LOCA', 'SGTR', 'MSLB'][0]

        self.Loger_txt = ''

        self.input_info = [
            ('ZINST78', 1000)
        ]

        self.action_space = 1
        self.observation_space = len(self.input_info)

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
        self.Loger_txt += f'{np.array(state)}\t'
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

        self.Loger_txt += f'{r}\t'
        return r

    def get_done(self):
        if self.AcumulatedReward < -500 or self.AcumulatedReward > 400:
            d = True
        else:
            d = False
        self.Loger_txt += f'{d}\t'
        return d

    def s_val(self, pa, va):
        for _ in range(len(pa)):
            self.logicpara.append(pa[_])
            self.logicval.append(va[_])

    def send_act(self, A):
        # A = A.clip(min=0, max=0.25)
        # modified_action = [_ * 100 for _ in A.tolist()]
        # self._send_control_signal(['WAFWS1'], modified_action)
        modified_action = A

        # Logic Line
        self.logicpara = []
        self.logicval = []

        if self.accident_name == 'LOCA':
            # 16.0
            for _tar, _val in zip(['WAFWS1', 'WAFWS2', 'WAFWS3'], ['KSWO143', 'KSWO152', 'KSWO155']):
                if self.mem[_tar]['Val'] < 20:
                    if self.get_CNS_time() >= self.FixedRad + 1325: self.s_val([_val], [1])
            # 17.2
            if self.get_CNS_time() == self.FixedRad + 1750: self.s_val(['KSWO208'], [1])

            # 20.4
            if self.get_CNS_time() == self.FixedRad + 2000: self.s_val(['KSWO115'], [1])
            if self.get_CNS_time() == self.FixedRad + 2300: self.s_val(['KSWO123'], [1])

            # 21.3
            if self.get_CNS_time() == self.FixedRad + 2600: self.s_val(['KSWO129'], [1])
            if self.get_CNS_time() == self.FixedRad + 2650: self.s_val(['KSWO130'], [1])
            if self.get_CNS_time() == self.FixedRad + 2700: self.s_val(['KSWO131'], [1])

        elif self.accident_name == 'SGTR':
            print(self.mem['cMALC']['Val'])
            # 16.0
            for _tar, _val in zip(['WAFWS1', 'WAFWS2', 'WAFWS3'], ['KSWO143', 'KSWO152', 'KSWO155']):
                if self.mem[_tar]['Val'] < 20:
                    if self.get_CNS_time() >= self.FixedRad + 900: self.s_val([_val], [1])
            # 17.2
            if self.get_CNS_time() == self.FixedRad + 1400: self.s_val(['KSWO208'], [1])

            # 20.4
            if self.get_CNS_time() == self.FixedRad + 2000: self.s_val(['KSWO115'], [1])
            if self.get_CNS_time() == self.FixedRad + 2300: self.s_val(['KSWO123'], [1])

            # 21.3
            if self.get_CNS_time() == self.FixedRad + 2800: self.s_val(['KSWO129'], [1])
            if self.get_CNS_time() == self.FixedRad + 2900: self.s_val(['KSWO130'], [1])
            if self.get_CNS_time() == self.FixedRad + 3000: self.s_val(['KSWO131'], [1])

            # 비상03 4.2
            if str(self.mem['cMALC']['Val'])[0] == 1:   # SG1번 고장
                if self.mem['WAFWS1']['Val'] != 0:
                    if self.get_CNS_time() >= self.FixedRad + 3200: self.s_val(['KSW0142'], [1])
            if str(self.mem['cMALC']['Val'])[0] == 2:   # SG2번 고장
                if self.mem['WAFWS2']['Val'] != 0:
                    if self.get_CNS_time() >= self.FixedRad + 3200: self.s_val(['KSWO151'], [1])
            if str(self.mem['cMALC']['Val'])[0] == 3:   # SG3번 고장
                if self.mem['WAFWS3']['Val'] != 0:
                    if self.get_CNS_time() >= self.FixedRad + 3200: self.s_val(['KSWO154'], [1])

            # 비상03 Oil
            if self.get_CNS_time() == self.FixedRad + 4100: self.s_val(['KSW0130'], [1])

            # 비상03 RCP2
            if self.get_CNS_time() == self.FixedRad + 4200: self.s_val(['KSWO133'], [1])

            # 비상03 Spray Man
            if self.get_CNS_time() == self.FixedRad + 4350:
                if self.mem['KLAMPO119']['Val'] == 0: self.s_val(['KSWO128'], [1])

            # 비상03 Spary Up
            if self.get_CNS_time() >= self.FixedRad + 4450:
                if self.mem['ZINST66']['Val'] <= 50: self.s_val(['KSWO127'], [1])

        elif self.accident_name == 'MSLB':
            # 16.0
            for _tar, _val in zip(['WAFWS1', 'WAFWS2', 'WAFWS3'], ['KSWO143', 'KSWO152', 'KSWO155']):
                if self.mem[_tar]['Val'] < 20:
                    if self.get_CNS_time() >= self.FixedRad + 700: self.s_val([_val], [1])
            # 17.2
            if self.get_CNS_time() == self.FixedRad + 1500: self.s_val(['KSWO208'], [1])

            # 20.4
            if self.get_CNS_time() == self.FixedRad + 1700: self.s_val(['KSWO115'], [1])
            if self.get_CNS_time() == self.FixedRad + 1850: self.s_val(['KSWO123'], [1])

            # 21.3
            if self.get_CNS_time() == self.FixedRad + 2300: self.s_val(['KSWO129'], [1])
            if self.get_CNS_time() == self.FixedRad + 2450: self.s_val(['KSWO130'], [1])
            if self.get_CNS_time() == self.FixedRad + 2500: self.s_val(['KSWO131'], [1])

            # 비상03 4.2
            if str(self.mem['cMALC']['Val'])[0] == 1:   # MSLB 1번 고장
                if self.mem['WAFWS1']['Val'] != 0:
                    if self.get_CNS_time() >= self.FixedRad + 3200: self.s_val(['KSW0142'], [1])
            if str(self.mem['cMALC']['Val'])[0] == 2:   # MSLB 2번 고장
                if self.mem['WAFWS2']['Val'] != 0:
                    if self.get_CNS_time() >= self.FixedRad + 3200: self.s_val(['KSWO151'], [1])
            if str(self.mem['cMALC']['Val'])[0] == 3:   # MSLB 3번 고장
                if self.mem['WAFWS3']['Val'] != 0:
                    if self.get_CNS_time() >= self.FixedRad + 3200: self.s_val(['KSWO154'], [1])

        if self.logicpara != []:
            self._send_control_signal(self.logicpara, self.logicval)
        return modified_action

    def step(self, A):
        """
        A를 받고 1 step 전진
        :param A: [Act], numpy.ndarry, Act는 numpy.float32
        :return: 최신 state와 reward done 반환
        """
        # Old Data (time t) ---------------------------------------
        self.send_act(A)

        # New Data (time t+1) -------------------------------------
        super(ENVCNS, self).step()
        self._append_val_to_list()
        self.ENVStep += 1

        reward = self.get_reward()
        done = self.get_done()
        next_state = self.get_state()

        self.ENVlogging(s=self.Loger_txt)
        self.Loger_txt = f'{next_state}\t'
        return next_state, reward, done, A

    def reset(self, file_name):
        # 1] CNS 상태 초기화 및 초기화된 정보 메모리에 업데이트
        super(ENVCNS, self).reset(initial_nub=1, mal=True, mal_case=12, mal_opt=10050, mal_time=30, file_name=file_name)
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

    def Reset(self, mal_case, mal_opt, mal_time, file_name): # 절대 데이터 모으기 이외에 사용 금지
        # 1] CNS 상태 초기화 및 초기화된 정보 메모리에 업데이트
        super(ENVCNS, self).reset(initial_nub=1, mal=True, mal_case=mal_case, mal_opt=mal_opt, mal_time=mal_time, file_name=file_name)
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

    # CSF Tree
    def check_CSFTree(self):
        V = {
            # CSF 1 Value 미임계 상태 추적도
            'PowerRange': self.mem['ZINST1']['Val'],        'IntermediateRange': self.mem['ZINST2']['Val'],
            'SourceRange': self.mem['ZINST3']['Val'],
            # CSF 2 Valve 노심냉각 상태 추적도
            'CoreExitTemp': self.mem['UUPPPL']['Val'],      'PTCurve': 0,
             }


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