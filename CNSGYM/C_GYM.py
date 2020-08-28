import numpy as np

from COMMONTOOL import TOOL
from CNS_UDP_FAST import CNS


class ENVCNS(CNS):
    def __init__(self, Name, IP, PORT):
        super(ENVCNS, self).__init__(threrad_name=Name,
                                     CNS_IP=IP, CNS_Port=PORT, Remote_IP='192.168.0.29', Remote_Port=PORT, Max_len=10)
        self.AcumulatedReward = 0

        self.action_space = 1
        self.observation_space = 7
        self.want_tick = 25

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
        return r

    def get_done(self):
        if self.AcumulatedReward < -500 or self.AcumulatedReward > 400:
            d = True
        else:
            d = False
        return d

    def send_action_append(self, pa, va):
        for _ in range(len(pa)):
            self.para.append(pa[_])
            self.val.append(va[_])

    def step(self, A):
        """
        A를 받고 1 step 전진
        :param A: [Act], numpy.ndarry, Act는 numpy.float32
        :return: 최신 state와 reward done 반환
        """

        self.para, self.val = [], []

        # 사용되는 파라메터 전체 업데이트
        self.Time_tick = self.mem['KCNTOMS']['Val']
        self.Reactor_power = self.mem['QPROREL']['Val']  # 0.02
        self.Tavg = self.mem['UAVLEGM']['Val']  # 308.21
        self.Tref = self.mem['UAVLEGS']['Val']  # 308.22
        self.rod_pos = [self.mem[nub_rod]['Val'] for nub_rod in ['KBCDO10', 'KBCDO9', 'KBCDO8', 'KBCDO7']]
        #
        self.charging_valve_state = self.mem['KLAMPO95']['Val']  # 0(Auto) - 1(Man)
        self.main_feed_valve_1_state = self.mem['KLAMPO147']['Val']
        self.main_feed_valve_2_state = self.mem['KLAMPO148']['Val']
        self.main_feed_valve_3_state = self.mem['KLAMPO149']['Val']
        self.vct_level = self.mem['ZVCT']['Val']  # 74.45
        self.pzr_level = self.mem['ZINST63']['Val']  # 34.32
        #
        self.Turbine_setpoint = self.mem['KBCDO17']['Val']
        self.Turbine_ac = self.mem['KBCDO18']['Val']  # Turbine ac condition
        self.Turbine_real = self.mem['KBCDO19']['Val']  # 20
        self.load_set = self.mem['KBCDO20']['Val']  # Turbine load set point
        self.load_rate = self.mem['KBCDO21']['Val']  # Turbine load rate
        self.Mwe_power = self.mem['KBCDO22']['Val']  # 0

        self.Netbreak_condition = self.mem['KLAMPO224']['Val']  # 0 : Off, 1 : On
        self.trip_block = self.mem['KLAMPO22']['Val']  # Trip block condition 0 : Off, 1 : On
        #
        self.steam_dump_condition = self.mem['KLAMPO150']['Val']  # 0: auto 1: man
        self.heat_drain_pump_condition = self.mem['KLAMPO244']['Val']  # 0: off, 1: on
        self.main_feed_pump_1 = self.mem['KLAMPO241']['Val']  # 0: off, 1: on
        self.main_feed_pump_2 = self.mem['KLAMPO242']['Val']  # 0: off, 1: on
        self.main_feed_pump_3 = self.mem['KLAMPO243']['Val']  # 0: off, 1: on
        self.cond_pump_1 = self.mem['KLAMPO181']['Val']  # 0: off, 1: on
        self.cond_pump_2 = self.mem['KLAMPO182']['Val']  # 0: off, 1: on
        self.cond_pump_3 = self.mem['KLAMPO183']['Val']  # 0: off, 1: on

        self.ax_off = self.mem['CAXOFF']['Val']  # -0.63

        self.C_7Sig = self.mem['KLAMPO206']['Val']  #
        self.C_5Sig = self.mem['KLAMPO205']['Val']  #

        self.TRIP = self.mem['KRXTRIP']['Val']


        self.para = []
        self.val = []

        # 주급수 및 CVCS 자동
        if self.charging_valve_state == 1:
            self.send_action_append(['KSWO100'], [0])
        if self.main_feed_valve_1_state == 1 or self.main_feed_valve_2_state == 1 or self.main_feed_valve_3_state == 1:
            self.send_action_append(['KSWO171', 'KSWO165', 'KSWO159'], [0, 0, 0])

        # self.rod_pos = [self.CNS.mem[nub_rod]['Val'] for nub_rod in ['KBCDO10', 'KBCDO9', 'KBCDO8', 'KBCDO7']]
        if self.rod_pos[0] >= 228 and self.rod_pos[1] >= 228 and self.rod_pos[0] >= 100:
            # 거의 많이 뽑혔을때 Makeup
            self.send_action_append(['KSWO78', 'WDEWT'], [1, 1])  # Makeup

        # 절차서 구성 순서로 진행
        # 1) 출력이 4% 이상에서 터빈 set point를 맞춘다.
        if self.Reactor_power >= 0.04 and self.Turbine_setpoint != 1800:
            if self.Turbine_setpoint < 1790:  # 1780 -> 1872
                self.send_action_append(['KSWO213'], [1])
            elif self.Turbine_setpoint >= 1790:
                self.send_action_append(['KSWO213'], [0])
        # 1) 출력 4% 이상에서 터빈 acc 를 200 이하로 맞춘다.
        if self.Reactor_power >= 0.04 and self.Turbine_ac != 210:
            if self.Turbine_ac < 200:
                self.send_action_append(['KSWO215'], [1])
            elif self.Turbine_ac >= 200:
                self.send_action_append(['KSWO215'], [0])
        # 2) 출력 10% 이상에서는 Trip block 우회한다.
        if self.Reactor_power >= 0.10 and self.trip_block != 1:
            self.send_action_append(['KSWO22', 'KSWO21'], [1, 1])
        # 2) 출력 10% 이상에서는 rate를 50까지 맞춘다.
        if self.Reactor_power >= 0.10 and self.Mwe_power <= 0:
            if self.load_set < 100:
                self.send_action_append(['KSWO225', 'KSWO224'], [1, 0])  # 터빈 load를 150 Mwe 까지,
            else:
                self.send_action_append(['KSWO225', 'KSWO224'], [0, 0])
            if self.load_rate < 25:
                self.send_action_append(['KSWO227', 'KSWO226'], [1, 0])
            else:
                self.send_action_append(['KSWO227', 'KSWO226'], [0, 0])

        def range_fun(st, end, goal):
            if st <= self.Reactor_power < end:
                if self.load_set < goal:
                    self.send_action_append(['KSWO225', 'KSWO224'], [1, 0])  # 터빈 load를 150 Mwe 까지,
                else:
                    self.send_action_append(['KSWO225', 'KSWO224'], [0, 0])

        range_fun(st=0.10, end=0.20, goal=100)
        range_fun(st=0.20, end=0.25, goal=150)
        range_fun(st=0.25, end=0.30, goal=200)
        range_fun(st=0.30, end=0.35, goal=250)
        range_fun(st=0.35, end=0.40, goal=300)
        range_fun(st=0.40, end=0.45, goal=350)
        range_fun(st=0.45, end=0.50, goal=400)
        range_fun(st=0.50, end=0.55, goal=450)
        range_fun(st=0.55, end=0.60, goal=500)
        range_fun(st=0.60, end=0.65, goal=550)
        range_fun(st=0.65, end=0.70, goal=600)
        range_fun(st=0.70, end=0.75, goal=650)
        range_fun(st=0.75, end=0.80, goal=700)
        range_fun(st=0.80, end=0.85, goal=750)
        range_fun(st=0.85, end=0.90, goal=800)
        range_fun(st=0.90, end=0.95, goal=850)
        range_fun(st=0.95, end=1.00, goal=900)
        range_fun(st=1.00, end=1.50, goal=930)

        # 3) 출력 15% 이상 및 터빈 rpm이 1800이 되면 netbreak 한다.
        if self.Reactor_power >= 0.15 and self.Turbine_real >= 1790 and self.Netbreak_condition != 1:
            self.send_action_append(['KSWO244'], [1])

        # 3.1) C-5 점등, C-7 점등
        if self.C_5Sig == 0 and self.C_7Sig == 0:
            # 4) 출력 15% 이상 및 전기 출력이 존재하는 경우, steam dump auto로 전향
            if self.Reactor_power >= 0.15 and self.Mwe_power > 0 and self.steam_dump_condition == 1:
                self.send_action_append(['KSWO176'], [0])
            # 5) 출력 15% 이상 및 전기 출력이 존재하는 경우, heat drain pump on
            if self.Reactor_power >= 0.15 and self.Mwe_power > 0 and self.heat_drain_pump_condition == 0:
                self.send_action_append(['KSWO195'], [1])
            # 6) 출력 20% 이상 및 전기 출력이 190Mwe 이상 인경우
            # if self.Reactor_power >= 0.20 and self.Mwe_power >= 190 and self.cond_pump_2 == 0:
            if self.Reactor_power >= 0.20 and self.cond_pump_2 == 0:
                self.send_action_append(['KSWO205'], [1])
            # 7) 출력 40% 이상 및 전기 출력이 380Mwe 이상 인경우
            # if self.Reactor_power >= 0.40 and self.Mwe_power >= 380 and self.main_feed_pump_2 == 0:
            if self.Reactor_power >= 0.40 and self.main_feed_pump_2 == 0:
                self.send_action_append(['KSWO193'], [1])
            # 8) 출력 50% 이상 및 전기 출력이 475Mwe
            # if self.Reactor_power >= 0.50 and self.Mwe_power >= 475 and self.cond_pump_3 == 0:
            if self.Reactor_power >= 0.50 and self.cond_pump_3 == 0:
                self.send_action_append(['KSWO206'], [1])
            # 9) 출력 80% 이상 및 전기 출력이 765Mwe
            # if self.Reactor_power >= 0.80 and self.Mwe_power >= 600 and self.main_feed_pump_3 == 0:
            if self.Reactor_power >= 0.80 and self.main_feed_pump_3 == 0:
                self.send_action_append(['KSWO192'], [1])

        if A == '':
            pass
        else:
            # 9) 제어봉 조작 신호를 보내기
            if A == 0:
                self.send_action_append(['KSWO33', 'KSWO32'], [0, 0])  # Stay
            elif A == 1:
                self.send_action_append(['KSWO33', 'KSWO32'], [1, 0])  # Out
            elif A == 2:
                self.send_action_append(['KSWO33', 'KSWO32'], [0, 1])  # In
        # ---------------------
        print(f"A: {A}")

        self._send_control_signal(self.para, self.val)

        super(ENVCNS, self).step()
        self._append_val_to_list()

        next_state = np.array([self.mem['ZINST78']['Val']/1000,
                               self.mem['ZINST77']['Val']/1000,
                               self.mem['ZINST76']['Val']/1000,
                               self.mem['WAFWS1']['Val']/100,
                               self.mem['WAFWS2']['Val']/100,
                               self.mem['WAFWS3']['Val']/100,
                               self.mem['KLAMPO9']['Val'],
                               ])

        reward = self.get_reward()
        done = self.get_done()

        return next_state, reward, done, 0

    def reset(self, file_name):
        # 1] CNS 상태 초기화 및 초기화된 정보 메모리에 업데이트
        # super(ENVCNS, self).reset(initial_nub=25, mal=False, mal_case=1, mal_opt=1, mal_time=1, file_name=file_name)
        # super(ENVCNS, self).reset(initial_nub=27, mal=False, mal_case=1, mal_opt=1, mal_time=1, file_name=file_name)
        super(ENVCNS, self).reset(initial_nub=26, mal=False, mal_case=1, mal_opt=1, mal_time=1, file_name=file_name)
        # 2] 업데이트된 'Val'를 'List'에 추가
        self._append_val_to_list()
        # 3] 'Val'을 상태로 제작후 반환
        state = np.array([self.mem['ZINST78']['Val']/1000,
                          self.mem['ZINST77']['Val']/1000,
                          self.mem['ZINST76']['Val']/1000,
                          self.mem['WAFWS1']['Val']/100,
                          self.mem['WAFWS2']['Val']/100,
                          self.mem['WAFWS3']['Val']/100,
                          self.mem['KLAMPO9']['Val'],
                          ])
        # 4] 보상 누적치 초기화
        self.AcumulatedReward = 0

        return state


if __name__ == '__main__':
    C_GYM = ENVCNS(Name="TEST", IP='192.168.0.103', PORT=7101)
    C_GYM.reset(file_name='TEST')

    while True:
        if C_GYM.mem['KLAMPO28']['Val'] == 1:
            C_GYM.step(int(4))
        else:
            A = input()
            try:
                C_GYM.step(int(A))
                C_GYM.step(int(0))
                C_GYM.step(int(0))
                C_GYM.step(int(0))
                C_GYM.step(int(0))
                # A_ = 0
                # while True:
                #     A_ += 1
                #     C_GYM.step(int(A_))
                #     if A_ >= 2:
                #         A_ = 0
                # # for A_ in [1,0,1,0,1,0,1,0,1,0,
                # #            2,0,2,0,2,0,2,0,2,0]:
                # # for A_ in [1, 0, 2, 0, 1, 0, 2, 0, 1, 0,
                # #            2, 0, 1, 0, 2, 0, 1, 0, 2, 0]:
                # #     C_GYM.step(int(A_))
            except:
                pass