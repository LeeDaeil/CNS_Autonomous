import time
import sys
import multiprocessing
from PySide2.QtWidgets import QApplication, QWidget, QMainWindow, QSizePolicy, QPushButton
from PySide2.QtCore import QTimer

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
            'BHV142': [], 'BFV122': [], 'ZINST65': [], 'ZINST63': [],
        } for i in range(nub_agent)}

        self.ENVReward = {i: {
            'R': [], 'CurAcuR': 0
        } for i in range(nub_agent)}

        self.ENVActDis = {i: {
            'Mean': [], 'Std': []
        } for i in range(nub_agent)}

        self.ENVReward['AcuR'] = []

    def push_currentEP(self, i, ep):
        self.StepInEachAgent[i] = ep

    def push_ENV_val(self, i, Dict_val):
        """
        :param i: 에이전트 넘버
        :param Dict_val: {'Para': Val}
        """
        for key in Dict_val.keys():
            if key == 'KCNTOMS':
                self.ENVVALInEachAgent[i][key].append(- Dict_val[key])
            else:
                self.ENVVALInEachAgent[i][key].append(Dict_val[key])

    def push_ENV_reward(self, i, Dict_val):
        self.ENVReward[i]['R'].append(Dict_val['R'])
        self.ENVReward[i]['CurAcuR'] = Dict_val['AcuR']

    def push_ENV_ActDis(self, i, Dict_val):
        self.ENVActDis[i]['Mean'].append(Dict_val['Mean'])
        self.ENVActDis[i]['Std'].append(Dict_val['Std'])

    def init_ENV_val(self, i):
        # 종료 또는 초기화로
        self.ENVReward['AcuR'].append(self.ENVReward[i]['CurAcuR'])

        for key in self.ENVVALInEachAgent[i].keys():
            self.ENVVALInEachAgent[i][key] = []
        self.ENVReward[i]['R'] = []
        self.ENVReward[i]['CurAcuR'] = 0

        self.ENVActDis[i]['Mean'] = []
        self.ENVActDis[i]['Std'] = []

    def get_currentEP(self):
        get_db = self.StepInEachAgent
        return get_db

    def get_ENV_val(self, i):
        return self.ENVVALInEachAgent[i]

    def get_ENV_all_val(self):
        return self.ENVVALInEachAgent

    def get_ENV_ActDis(self, i):
        return [self.ENVActDis[i]['Mean'], self.ENVActDis[i]['Std']]

    def get_ENV_reward_val(self, i):
        return [self.ENVReward[i]['R'], self.ENVReward['AcuR']]

    def get_ENV_nub(self):
        return self.nub_agent


class Monitoring(multiprocessing.Process):
    def __init__(self, Monitoring_ENV):
        multiprocessing.Process.__init__(self)
        self.daemon = True
        self.mem = Monitoring_ENV

    def run(self):
        app = QApplication(sys.argv)
        w = Mainwindow(self.mem)
        sys.exit(app.exec_())


class Mainwindow(QMainWindow):
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

        self.GP = PlotCanvas(self, width=16, height=8)
        self.GP.move(0, 0)

        self.button = QPushButton('Next', self)
        self.button.nub = 0
        self.button.setToolTip('Next')
        self.button.move(0, 0)
        self.button.resize(50, 20)
        self.button.clicked.connect(self.NEXTPAGE)

        self.stopbutton = QPushButton('', self)
        self.stopbutton.cond_stop = False
        if not self.stopbutton.cond_stop: self.stopbutton.setText('Run')
        self.stopbutton.move(50, 0)
        self.stopbutton.resize(50, 20)
        self.stopbutton.clicked.connect(self.PAUSE)

        self.SaveFig = QPushButton('Save', self)
        self.SaveFig.move(100, 0)
        self.SaveFig.resize(50, 20)
        self.SaveFig.clicked.connect(self.Savefig)

        self.show()

    def update_plot(self):
        self.setWindowTitle(f'AGENT_{self.button.nub}')
        if not self.stopbutton.cond_stop:
            self.GP.plot(self.mem.get_ENV_val(self.button.nub),
                         self.mem.get_ENV_all_val(),
                         self.mem.get_ENV_reward_val(self.button.nub),
                         self.mem.get_ENV_ActDis(self.button.nub),
                         )

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
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = plt.figure(figsize=(width, height), dpi=dpi)

        gs = GridSpec(2, 2, figure=self.fig)
        self.ax1 = self.fig.add_subplot(gs[0:1, 0:1])
        self.ax2 = self.fig.add_subplot(gs[0:1, 1:2])
        self.ax3 = self.fig.add_subplot(gs[1:2, 0:1], projection='3d')

        FigureCanvas.__init__(self, self.fig)
        self.setParent(parent)

        FigureCanvas.setSizePolicy(self,
                                   QSizePolicy.Expanding,
                                   QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)

    def plot(self, val, A_val, reward_mem, ActDis):
        self.ax1.clear()
        self.ax2.clear()
        self.ax3.clear()

        self.ax1.plot(val['BHV142'])
        self.ax1.plot(val['BFV122'])

        for _ in range(0, 4):
            self.ax2.plot(A_val[_]['ZINST65'], label=f'Agent{_}')
        self.ax2.plot(A_val[4]['ZINST65'], label='PID')
        self.ax2.legend(loc=0)

        # Distribution
        mean, std = ActDis[0], ActDis[1]  # [0, 1]
        mean_x = mean[-1]
        mean = np.array(mean[-50:])
        mean = mean.reshape(len(mean), 1)
        std = np.array(std[-50:])
        std = std.reshape(len(std), 1)
        x = np.array([np.arange(-1, 1, 0.01) for _ in range(0, len(mean))])
        y1 = stats.norm(mean, std).pdf(x)
        for _ in range(0, len(mean)):
            self.ax3.plot(x[_], y1[_], zs=-_, zdir='y', color='blue', alpha=0.02 * _)
        self.ax3.plot([-1, -1], [0, 1], zs=-_, zdir='y', color='red', alpha=0.5)
        self.ax3.plot([0, 0], [0, 1], zs=-_, zdir='y', color='red', alpha=0.5)
        self.ax3.plot([1, 1], [0, 1], zs=-_, zdir='y', color='red', alpha=0.5)

        self.ax1.grid()
        self.ax2.grid()
        self.ax3.grid()

        self.fig.set_tight_layout(True)
        self.fig.canvas.draw()

    def saveFig(self):
        self.fig.savefig('SaveFIg.png', dpi=100)