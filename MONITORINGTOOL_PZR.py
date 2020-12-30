import time
import sys
import multiprocessing
from PyQt5.QtWidgets import QApplication, QWidget, QSizePolicy, QPushButton, QVBoxLayout, QHBoxLayout
from PyQt5.QtCore import QTimer

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
import scipy.stats as stats
from COMMONTOOL import PTCureve


class MonitoringMEM:
    def __init__(self, nub_agent):
        self.nub_agent = nub_agent
        self.StepInEachAgent = {i: 0 for i in range(nub_agent)}
        self.ENVVALInEachAgent = {i: {
            'UUPPPL': [], 'UPRZ': [], 'ZINST58': [], 'ZINST63': [],
            'BHV142': [], 'BFV122': [], 'ZINST66': [],
        } for i in range(nub_agent)}

        self.ENVReward = {i: {
            'R': [], 'CurAcuR': 0
        } for i in range(nub_agent)}

        self.ENVActDis = {i: {
            'Mean': [], 'Std': [],
            'A0': [], 'OA0': [],
            'A1': [], 'OA1': [],
            # 'A2': [], 'OA2': [],
        } for i in range(nub_agent)}

        self.ENVReward['AcuR/Ep'] = []
        self.ENVReward['q1/Ep'] = []
        self.ENVReward['q2/Ep'] = []
        self.ENVReward['p/Ep'] = []

    def push_currentEP(self, i, ep):
        self.StepInEachAgent[i] = ep

    def push_ENV_epinfo(self, Dict_val):
        for key in Dict_val:
            self.ENVReward[key].append(Dict_val[key])

    def push_ENV_val(self, i, CNSMem):
        """
        :param i: 에이전트 넘버 <-
        :param CNSMem: ...
        """
        for key in self.ENVVALInEachAgent[i].keys():
            self.ENVVALInEachAgent[i][key].append(CNSMem[key]['Val'])

    def push_ENV_reward(self, i, Dict_val):
        self.ENVReward[i]['R'].append(Dict_val['R'])
        self.ENVReward[i]['CurAcuR'] = Dict_val['AcuR']

    def push_ENV_ActDis(self, i, Dict_val):
        self.ENVActDis[i]['Mean'].append(Dict_val['Mean'])
        self.ENVActDis[i]['Std'].append(Dict_val['Std'])
        self.ENVActDis[i]['A0'].append(Dict_val['A0'])
        self.ENVActDis[i]['A1'].append(Dict_val['A1'])
        # self.ENVActDis[i]['A2'].append(Dict_val['A2'])
        self.ENVActDis[i]['OA0'].append(Dict_val['OA0'])
        self.ENVActDis[i]['OA1'].append(Dict_val['OA1'])
        # self.ENVActDis[i]['OA2'].append(Dict_val['OA2'])

    def init_ENV_val(self, i):
        # 종료 또는 초기화로
        for key in self.ENVVALInEachAgent[i].keys():
            self.ENVVALInEachAgent[i][key] = []

        self.ENVReward[i]['R'] = []
        self.ENVReward[i]['CurAcuR'] = 0

        self.ENVActDis[i]['Mean'] = []
        self.ENVActDis[i]['Std'] = []
        self.ENVActDis[i]['A0'] = []
        self.ENVActDis[i]['A1'] = []
        # self.ENVActDis[i]['A2'] = []
        self.ENVActDis[i]['OA0'] = []
        self.ENVActDis[i]['OA1'] = []
        # self.ENVActDis[i]['OA2'] = []

    def get_currentEP(self):
        get_db = self.StepInEachAgent
        return get_db

    def get_ENV_val(self, i):
        return self.ENVVALInEachAgent[i]

    def get_ENV_all_val(self):
        return self.ENVVALInEachAgent

    def get_ENV_ActDis(self, i):
        return [self.ENVActDis[i]['Mean'], self.ENVActDis[i]['Std'],
                self.ENVActDis[i]['A0'], self.ENVActDis[i]['OA0'],
                self.ENVActDis[i]['A1'], self.ENVActDis[i]['OA1'],
                # self.ENVActDis[i]['A2'], self.ENVActDis[i]['OA2']
                ]

    def get_ENV_reward_val(self, i):
        return [self.ENVReward[i]['R'], self.ENVReward['AcuR/Ep'],
                self.ENVReward['q1/Ep'], self.ENVReward['q2/Ep'], self.ENVReward['p/Ep']]
    def get_ENV_nub(self):
        return self.nub_agent


class Monitoring(multiprocessing.Process):
    def __init__(self, Monitoring_ENV):
        multiprocessing.Process.__init__(self)
        self.daemon = True
        self.mem = Monitoring_ENV

    def run(self):
        print('Run Monitoring TOOL')
        app = QApplication(sys.argv)
        w = Mainwindow(self.mem)
        sys.exit(app.exec_())


class Mainwindow(QWidget):
    def __init__(self, mem):
        super().__init__()
        self.mem = mem

        timer = QTimer(self)
        for _ in [self.update_plot]:
            timer.timeout.connect(_)
        timer.start(300)

        self.initUI()

    def initUI(self):
        self.setGeometry(100, 100, 640, 400)
        main_layout = QVBoxLayout(self)
        layout_1 = QHBoxLayout()
        layout_2 = QVBoxLayout()

        # Layout_1
        self.button = QPushButton('Next')
        self.button.nub = 0
        self.button.setToolTip('Next')
        self.button.move(0, 0)
        self.button.resize(50, 20)
        self.button.clicked.connect(self.NEXTPAGE)

        self.stopbutton = QPushButton('-')
        self.stopbutton.cond_stop = False
        if not self.stopbutton.cond_stop: self.stopbutton.setText('Run')
        self.stopbutton.move(50, 0)
        self.stopbutton.resize(50, 20)
        self.stopbutton.clicked.connect(self.PAUSE)

        self.SaveFig = QPushButton('Save')
        self.SaveFig.move(100, 0)
        self.SaveFig.resize(50, 20)
        self.SaveFig.clicked.connect(self.Savefig)

        layout_1.addWidget(self.button)
        layout_1.addWidget(self.stopbutton)
        layout_1.addWidget(self.SaveFig)

        # Layout_2
        self.GP = PlotCanvas()
        layout_2.addWidget(self.GP)

        # MainClose --
        main_layout.addLayout(layout_1)
        main_layout.addLayout(layout_2)

        self.show()

    def update_plot(self):
        self.setWindowTitle(f'AGENT_{self.button.nub}')
        if not self.stopbutton.cond_stop:
            try:
                self.GP.plot(self.mem.get_ENV_val(self.button.nub),
                             self.mem.get_ENV_ActDis(self.button.nub),
                             self.mem.get_ENV_reward_val(self.button.nub),
                             )
            except:
                pass

    def NEXTPAGE(self):
        target_nub = self.button.nub + 1
        if target_nub >= self.mem.get_ENV_nub():    target_nub = 0
        self.button.nub = target_nub

    def PAUSE(self):
        if self.stopbutton.cond_stop:
            self.stopbutton.setText('Run')
            self.stopbutton.cond_stop = False
        else:
            self.stopbutton.setText('Stop')
            self.stopbutton.cond_stop = True

    def Savefig(self):
        self.GP.saveFig()


class PlotCanvas(FigureCanvas):
    def __init__(self):
        FigureCanvas.__init__(self, plt.figure())

        self.gs = GridSpec(6, 2, figure=self.figure)
        self.axes = [
            # left
            self.figure.add_subplot(self.gs[0:2, 0:1]),
            self.figure.add_subplot(self.gs[2:4, 0:1]),
            # self.figure.add_subplot(self.gs[5:6, 0:1], projection='3d'),
            self.figure.add_subplot(self.gs[4:5, 0:1]),
            self.figure.add_subplot(self.gs[5:6, 0:1]),
            # right
            self.figure.add_subplot(self.gs[0:2, 1:2]), # temp
            self.figure.add_subplot(self.gs[2:4, 1:2]), # pres,level
            self.figure.add_subplot(self.gs[4:6, 1:2]), # pos
        ]

        self.figure.set_tight_layout(True)

    def plot(self, val, a_dis, r_val):
        [ax.clear() for ax in self.axes]
        y = [_ for _ in range(len(val['UUPPPL']))]
        self.axes[0].plot(r_val[0], label='Current Reward')
        self.axes[1].plot(r_val[1], label='Accumulated Reward/Ep')
        self.axes[2].plot(r_val[2], label='Q1')
        self.axes[2].plot(r_val[3], label='Q2')
        self.axes[3].plot(r_val[4], label='p')
        # self.axes[2].plot(a_dis[2], label='A0')
        # self.axes[2].plot(a_dis[4], label='A1')
        # self.axes[2].plot(a_dis[6], label='A2')
        # self.axes[3].plot(a_dis[3], label='A0')
        # self.axes[3].plot(a_dis[5], label='A1')
        # self.axes[3].plot(a_dis[7], label='A2')

        # Distribution
        # mean, std = a_dis[0], a_dis[1]  # [0, 1]
        # mean_x = mean[-1]
        # mean = np.array(mean[-50:])
        # print(np.shape(mean))
        # print(mean)
        # mean = mean.reshape(len(mean), 1)
        # std = np.array(std[-50:])
        # std = std.reshape(len(std), 1)
        # x = np.array([np.arange(-1, 1, 0.01) for _ in range(0, len(mean))])
        # y1 = stats.norm(mean, std).pdf(x)
        # for _ in range(0, len(mean)):
        #     self.axes[2].plot(x[_], y1[_], zs=-_, zdir='y', color='blue', alpha=0.02 * _)
        #     self.axes[2].plot([-1, -1], [0, 1], zs=-_, zdir='y', color='red', alpha=0.5)
        #     self.axes[2].plot([0, 0], [0, 1], zs=-_, zdir='y', color='red', alpha=0.5)
        #     self.axes[2].plot([1, 1], [0, 1], zs=-_, zdir='y', color='red', alpha=0.5)

        self.axes[4].plot(val['UUPPPL'], label='CoreExitTemp')
        self.axes[4].plot(val['UPRZ'], label='PzrTemp')

        self.axes[5].plot(val['ZINST58'], label='PZR Pres')
        self.axes[5].plot(val['ZINST63'], label='PZR Level')

        self.axes[6].step(y, val['BHV142'], label='Letdown Pos')
        self.axes[6].step(y, val['BFV122'], label='Charging Pos')
        self.axes[6].step(y, [_/31 for _ in val['ZINST66']], label='PZR Spray Pos')

        for ax_, i in zip(self.axes, range(len(self.axes))):
            if i in [4, 5, 6]:
                ax_.legend(bbox_to_anchor=(1.01, 1.0), loc='upper left', borderaxespad=0.)
            else:
                # if i != 2: ax_.legend(loc=2)
                ax_.legend(loc=2)

            # get_tick_ = ax_.get_yticks()  # List
            # if i == 3: ax_.set_yticklabels([f'{_:0.0f}[℃]' for _ in get_tick_])
            # if i == 4: ax_.set_yticklabels([f'{_:0.0f}' for _ in get_tick_])
            # if i == 5: ax_.set_yticklabels([f'{_ * 100:0.0f}[%]' for _ in get_tick_])

            ax_.grid()

        self.figure.set_tight_layout(True)
        self.figure.canvas.draw()

    def saveFig(self):
        self.figure.savefig('SaveFIg.svg', format='svg', dpi=1200)


if __name__ == '__main__':
    # Test UI
    print('Run Monitoring TOOL')
    app = QApplication(sys.argv)
    w = Mainwindow(None)
    sys.exit(app.exec_())