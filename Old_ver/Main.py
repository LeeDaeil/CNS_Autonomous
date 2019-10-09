from multiprocessing import Manager
from Old_ver.CNS_data_share import CnsDataShare
from Old_ver.CNS_plot import DrawPlot
from Old_ver.CNS_data_cleaner import CnsDataCleaner

class StartAutonomous_CNS:
    def __init__(self):
        # 공유되는 mother memory 선언
        self.mother_memory = self.make_memory()

        # 실행되는 모듈 리스트 작성
        self.module_list = [
            CnsDataShare(self.mother_memory),
            DrawPlot(self.mother_memory),
            CnsDataCleaner(self.mother_memory),
        ]

    def Start_all_module(self):
        # 순차적으로 모듈 실행
        jobs = []
        for __ in self.module_list:
            __.start()
            jobs.append(__)

        for __ in jobs:
            __.join()

    def make_memory(self):
        Mother_mem = Manager().dict({
            'Single': {},           # {'PID' : {'Sig': 0, 'Val': 0, 'Nub': idx}}
            'List': {},             # {'PID' : {'Sig': 0, 'Val': [], 'Nub': idx}}
            'List_Deque': {},       # {'PID' : {'Sig': 0, 'Val': deque(max_length), 'Nub': idx}}
            'Nub': [],             # [0, 1, .... read time from CNS]
            'Inter': 1,
            'Clean': False         # Memory clean
        })
        return Mother_mem

if __name__ == '__main__':
    Autonomous = StartAutonomous_CNS()
    Autonomous.Start_all_module()
