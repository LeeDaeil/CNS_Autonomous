from CNS_data_share import CnsDataShare
from CNS_power_control import PowerControl
from CNS_plot import DrawPlot

class StartAutonomous_CNS:
    def __init__(self):
        # 공유되는 mother memory 선언
        self.mother_memory = {}
        self.mother_memory_list = {}
        self.mother_memory_deque = {}
        self.mother_memory_nub = {'Nub': []}
        self.clean = { 'Sig': False }

        # 실행되는 모듈 리스트 작성
        self.module_list = [CnsDataShare(self.mother_memory, self.mother_memory_list, self.mother_memory_deque,
                                         self.mother_memory_nub, self.clean),   # CNS data share module
                            PowerControl(self.mother_memory, self.clean),   # Power control
                            DrawPlot(self.mother_memory_list, self.mother_memory_nub)
                            ]

    def Start_all_module(self):
        # 순차적으로 모듈 실행
        for __ in self.module_list:
            __.start()


if __name__ == '__main__':
    Autonomous = StartAutonomous_CNS()
    Autonomous.Start_all_module()
