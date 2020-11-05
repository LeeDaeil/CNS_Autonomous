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
        self.want_tick = 50  # 1sec

        self.Loger_txt = ''

        self.input_info = [
            # (para, x_round, x_min, x_max), (x_min=0, x_max=0 is not normalized.)
            # ('PVCT',     1, 0,   2),          # VCT Press
            # ('ZVCT',     1, 0,   100),        # VCT Level

            # ('BLV616',   1, 0,   0),          # LV616, VCT->ChargingPump 유로 밸브
            # ('KLAMPO71', 1, 0,   0),          # Charging Pump 1
            # ('KLAMPO70', 1, 0,   0),          # Charging Pump 2
            # ('KLAMPO69', 1, 0,   0),          # Charging Pump 3
            # ('WCMINI',   1, 0,   4),          # Mini Flow
            # ('BHV30',    1, 0,   0),          # HV30 Mini Flow 밸브
            # ('BHV50',    1, 0,   0),          # HV50 ChargingPump->RCP Seal 유로 밸브
            # ('BFV122',   1, 0,   1),          # Charging Valve Pos
            # ('WAUXSP',   1, 0,   10),         # PZR aux spray Flow
            # ('BHV40',    1, 0,   1),          # PZR aux spray HV40, CVCS->HV40->PZR
            #
            ('WNETCH',   1, 0,   10),         # Total_Charging Flow
            # ('URHXUT',   1, 0,   300),        # Letdown RCS->VCT 온도 지시기
            # ('UCHGUT',   1, 0,   300),        # Charging Valve->RCS 온도 지시기
            ('ZINST58',  1, 0,   200),        # PZR_press : ZINST58
            ('ZINST63',  1, 0,   100),        # PZR_level : ZINST63
            #
            # ('BLV459',   1, 0,   1),          # Letdown LV459, RCS->VCT
            # ('BHV1',     1, 0,   1),          # HV1 Pos
            # ('BHV2',     1, 0,   1),          # HV2 Pos
            # ('BHV3',     1, 0,   1),          # HV3 Pos
            # ('BPV145',   1, 0,   1),          # Letdown_HX_pos = PV145 Pos
            # ('ZINST36',  1, 0,   40),         # Letdown HX Press
            # ('BHV41',    1, 0,   1),          # Letdown HV41, HV43->HV41->VCT
            # ('KHV43',    1, 0,   1),          # Letdown HV43, RCS->HV43->HV41
            #
            # ('WEXLD',    1, 0,   10),         # VCT_flow : WEXLD
            # ('WDEMI',    1, 0,   10),         # Total_in_VCT : WDEMI

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
        V = {
            'CNSTime': self.mem['KCNTOMS']['Val'],      # CNS Time : KCNTOMS
            'PVCT': self.mem['PVCT']['Val'],            # VCT_pressure : PVCT
            'VCT_level': self.mem['ZVCT']['Val'],       # VCT_level : ZVCT

            'BLV616': self.mem['BLV616']['Val'],        # LV616_pos : BLV616
            'KLAMPO71': self.mem['KLAMPO71']['Val'],    # CHP1 : KLAMPO71
            'KLAMPO72': self.mem['KLAMPO72']['Val'],    # CHP2 : KLAMPO72
            'KLAMPO73': self.mem['KLAMPO73']['Val'],    # CHP2 : KLAMPO73
            'WCMINI': self.mem['WCMINI']['Val'],        # Mini Flow : WCMINI
            'BHV30': self.mem['BHV30']['Val'],          # HV30_pos : BHV30
            'BHV50': self.mem['BHV50']['Val'],          # HV50_pos : BHV50
            'BFV122': self.mem['BFV122']['Val'],        # FV122_pos : BFV122
            'WAUXSP': self.mem['WAUXSP']['Val'],        # AuxSpray_Flow : WAUXSP
            'BHV40': self.mem['BHV40']['Val'],          # HV40 : BHV40

            'WNETCH': self.mem['WNETCH']['Val'],        # Total_Charging_Flow: WNETCH
            'URHXUT': self.mem['URHXUT']['Val'],        # Letdown_temp : URHXUT
            'UCHGUT': self.mem['UCHGUT']['Val'],        # Charging_temp : UCHGUT
            'ZINST58': self.mem['ZINST58']['Val'],      # PZR_press : ZINST58
            'PZR_level': self.mem['ZINST63']['Val'],    # PZR_level : ZINST63

            'BLV459': self.mem['BLV459']['Val'],        # Letdown_pos : BLV459
            'BHV1': self.mem['BHV1']['Val'],            # Orifice_1 : BHV1
            'BHV2': self.mem['BHV2']['Val'],            # Orifice_2 : BHV2
            'BHV3': self.mem['BHV3']['Val'],            # Orifice_3 : BHV3
            'BPV145': self.mem['BPV145']['Val'],        # Letdown_HX_pos : BPV145
            'ZINST36': self.mem['ZINST36']['Val'],      # Letdown HX Press : ZINST36
            'BHV41': self.mem['BHV41']['Val'],          # HV41_pos : BHV41
            'KHV43': self.mem['KHV43']['Val'],          # HV43_pos : KHV43

            'WEXLD': self.mem['WEXLD']['Val'],          # VCT_flow : WEXLD
            'WDEMI': self.mem['WDEMI']['Val'],          # Total_in_VCT : WDEMI
        }
        PZR_level_set = 57
        VCT_level_set = 74

        r = [0, 0]
        r[0] = TOOL.generate_r(curr=V['PZR_level'], setpoint=PZR_level_set, distance=0.5,
                               max_r=0.5, min_r=-5)
        # r[1] = TOOL.generate_r(curr=V['VCT_level'], setpoint=VCT_level_set, distance=0.5,
        #                        max_r=0.5, min_r=-5)
        self.Loger_txt += f'R:,{r},\t'
        r = sum(r)/100
        # r = self.normalize(sum(r), 0, -5, 0.5) / 10
        self.AcumulatedReward += r
        return r

    def get_done(self, r):
        V = {
            'CNSTime': self.mem['KCNTOMS']['Val'],  # CNS Time : KCNTOMS

            'PVCT': self.mem['PVCT']['Val'],  # VCT_pressure : PVCT
            'ZVCT': self.mem['ZVCT']['Val'],  # VCT_level : ZVCT

            'BLV616': self.mem['BLV616']['Val'],  # LV616_pos : BLV616
            'KLAMPO71': self.mem['KLAMPO71']['Val'],  # CHP1 : KLAMPO71
            'KLAMPO72': self.mem['KLAMPO72']['Val'],  # CHP2 : KLAMPO72
            'KLAMPO73': self.mem['KLAMPO73']['Val'],  # CHP2 : KLAMPO73
            'WCMINI': self.mem['WCMINI']['Val'],  # Mini Flow : WCMINI
            'BHV30': self.mem['BHV30']['Val'],  # HV30_pos : BHV30
            'BHV50': self.mem['BHV50']['Val'],  # HV50_pos : BHV50
            'BFV122': self.mem['BFV122']['Val'],  # FV122_pos : BFV122
            'WAUXSP': self.mem['WAUXSP']['Val'],  # AuxSpray_Flow : WAUXSP
            'BHV40': self.mem['BHV40']['Val'],  # HV40 : BHV40

            'WNETCH': self.mem['WNETCH']['Val'],  # Total_Charging_Flow: WNETCH
            'URHXUT': self.mem['URHXUT']['Val'],  # Letdown_temp : URHXUT
            'UCHGUT': self.mem['UCHGUT']['Val'],  # Charging_temp : UCHGUT
            'ZINST58': self.mem['ZINST58']['Val'],  # PZR_press : ZINST58
            'ZINST63': self.mem['ZINST63']['Val'],  # PZR_level : ZINST63

            'BLV459': self.mem['BLV459']['Val'],  # Letdown_pos : BLV459
            'BHV1': self.mem['BHV1']['Val'],  # Orifice_1 : BHV1
            'BHV2': self.mem['BHV2']['Val'],  # Orifice_2 : BHV2
            'BHV3': self.mem['BHV3']['Val'],  # Orifice_3 : BHV3
            'BPV145': self.mem['BPV145']['Val'],  # Letdown_HX_pos : BPV145
            'BHV41': self.mem['BHV41']['Val'],  # HV41_pos : BHV41
            'KHV43': self.mem['KHV43']['Val'],  # HV43_pos : KHV43

            'WEXLD': self.mem['WEXLD']['Val'],  # VCT_flow : WEXLD
            'WDEMI': self.mem['WDEMI']['Val'],  # Total_in_VCT : WDEMI
        }
        if V['CNSTime'] >= 25000:
            d = True
        else:
            d = False

        # 1. 너무 많이 벗어 나면 - 10점
        if 45 <= V['ZINST63'] <= 60:
            pass
        else:
            d = True
            r = -10

        self.Loger_txt += f'{d}\t'

        if self.Name == 0:
            print(r)

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
            'CNSTime': self.mem['KCNTOMS']['Val'],  # CNS Time : KCNTOMS

            'PVCT': self.mem['PVCT']['Val'],  # VCT_pressure : PVCT
            'ZVCT': self.mem['ZVCT']['Val'],  # VCT_level : ZVCT

            'BLV616': self.mem['BLV616']['Val'],  # LV616_pos : BLV616
            'KLAMPO71': self.mem['KLAMPO71']['Val'],  # CHP1 : KLAMPO71
            'KLAMPO72': self.mem['KLAMPO72']['Val'],  # CHP2 : KLAMPO72
            'KLAMPO73': self.mem['KLAMPO73']['Val'],  # CHP2 : KLAMPO73
            'WCMINI': self.mem['WCMINI']['Val'],  # Mini Flow : WCMINI
            'BHV30': self.mem['BHV30']['Val'],  # HV30_pos : BHV30
            'BHV50': self.mem['BHV50']['Val'],  # HV50_pos : BHV50
            'BFV122': self.mem['BFV122']['Val'],  # FV122_pos : BFV122
            'WAUXSP': self.mem['WAUXSP']['Val'],  # AuxSpray_Flow : WAUXSP
            'BHV40': self.mem['BHV40']['Val'],  # HV40 : BHV40

            'WNETCH': self.mem['WNETCH']['Val'],  # Total_Charging_Flow: WNETCH
            'URHXUT': self.mem['URHXUT']['Val'],  # Letdown_temp : URHXUT
            'UCHGUT': self.mem['UCHGUT']['Val'],  # Charging_temp : UCHGUT
            'ZINST58': self.mem['ZINST58']['Val'],  # PZR_press : ZINST58
            'ZINST63': self.mem['ZINST63']['Val'],  # PZR_level : ZINST63

            'BLV459': self.mem['BLV459']['Val'],  # Letdown_pos : BLV459
            'BHV1': self.mem['BHV1']['Val'],  # Orifice_1 : BHV1
            'BHV2': self.mem['BHV2']['Val'],  # Orifice_2 : BHV2
            'BHV3': self.mem['BHV3']['Val'],  # Orifice_3 : BHV3
            'BPV145': self.mem['BPV145']['Val'],  # Letdown_HX_pos : BPV145
            'BHV41': self.mem['BHV41']['Val'],  # HV41_pos : BHV41
            'KHV43': self.mem['KHV43']['Val'],  # HV43_pos : KHV43

            'WEXLD': self.mem['WEXLD']['Val'],  # VCT_flow : WEXLD
            'WDEMI': self.mem['WDEMI']['Val'],  # Total_in_VCT : WDEMI

            'BPV145MA': self.mem['KLAMPO89']['Val'],  # BPV145 Man(1)/Auto(0) : KLAMPO89
            'BFV122MA': self.mem['KLAMPO95']['Val'],  # BFV122 Man(1)/Auto(0) : KLAMPO95
        }
        ActOrderBook = {
            'BPV145Man': (['KSWO89'], [1]),
            'BFV122Man': (['KSWO100'], [1]),
        }
        # if self.Name == 0:
        #     print(f'{self.Name}_PV145:{V["BPV145"]}_BFV122:{V["BFV122"]}')
        #     print(round(AMod[0] / 100, 4))

        # if V['BPV145MA'] == 0: self._send_control_save(ActOrderBook['BPV145Man'])
        if V['BFV122MA'] == 0: self._send_control_save(ActOrderBook['BFV122Man'])

        if self.Name == 0:
            print(AMod)

        if AMod[0] < -0.6: Update_pos = V['BFV122'] + 0.02
        elif AMod[0] <= abs(0.6):
            if AMod < - 0.2: Update_pos = V['BFV122'] + 0.01
            elif AMod[0] <= abs(0.2): Update_pos = V['BFV122']
            else: Update_pos = V['BFV122'] - 0.01
        else: Update_pos = V['BFV122'] - 0.02

        Update_pos = np.clip(Update_pos, a_min=0.1, a_max=1)
        self._send_control_save((['BFV122'], [Update_pos]))

        # Letdown_pos = np.clip(round(AMod[1] / 100, 5) + V['BPV145'], a_min=0, a_max=0.8)
        # self._send_control_save((['BPV145'], [Letdown_pos]))

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
        self.want_tick = int(50)

        self.Monitoring_ENV.push_ENV_val(i=self.Name,
                                         Dict_val={f'{Para}': self.mem[f'{Para}']['Val'] for Para in
                                                   ['cMAL', 'cMALA', 'KCNTOMS',
                                                    'PVCT', 'ZVCT', 'ZINST58', 'ZINST63', 'BFV122', 'BPV145',
                                                    'WDEMI', 'WNETCH', 'WEXLD']}
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
        # 4.1] Monitoring init
        self.Monitoring_ENV.init_ENV_val(self.Name)
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