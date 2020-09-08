import time
import sys
import multiprocessing
from PySide2.QtWidgets import QApplication, QWidget, QMainWindow, QSizePolicy, QPushButton
from PySide2.QtCore import QTimer

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
from COMMONTOOL import PTCureve


class MonitoringMEM:
    def __init__(self, nub_agent):
        self.nub_agent = nub_agent
        self.StepInEachAgent = {i: 0 for i in range(nub_agent)}
        self.ENVVALInEachAgent = {i: {
            'UAVLEG2': [], 'KCNTOMS': [], 'ZINST65': [],
            'KLAMPO6': [], 'KLAMPO9': [],
        } for i in range(nub_agent)}

        self.ENVVALSetTime = {i: {'Time': 0, 'Temp': 0} for i in range(nub_agent)}
        self.ENVRATE = {i: {'RateX': [], 'RateY': [], 'RateZ': [],
                            'Zero': []} for i in range(nub_agent)}

        self.ENVReward = {i: {
            'R': [], 'CurAcuR': 0
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
        if Dict_val['KLAMPO6'] == 0 and Dict_val['KLAMPO9'] == 1 and Dict_val['KCNTOMS'] > 1500: # TODO
            if self.ENVVALSetTime[i]['Time'] == 0:
                self.ENVVALSetTime[i]['Time'] = Dict_val['KCNTOMS']
                self.ENVVALSetTime[i]['Temp'] = Dict_val['UAVLEG2']
            rate = -55 / (60 * 60 * 5)
            get_temp = rate * (Dict_val['KCNTOMS'] - self.ENVVALSetTime[i]['Time']) + self.ENVVALSetTime[i]['Temp']
            self.ENVRATE[i]['RateX'].append(get_temp)
            self.ENVRATE[i]['RateY'].append(- Dict_val['KCNTOMS'])
            self.ENVRATE[i]['RateZ'].append(0)
        self.ENVRATE[i]['Zero'].append(0)

    def push_ENV_reward(self, i, Dict_val):
        self.ENVReward[i]['R'].append(Dict_val['R'])
        self.ENVReward[i]['CurAcuR'] = Dict_val['AcuR']

    def init_ENV_val(self, i):
        # 종료 또는 초기화로
        self.ENVReward['AcuR'].append(self.ENVReward[i]['CurAcuR'])

        for key in self.ENVVALInEachAgent[i].keys():
            self.ENVVALInEachAgent[i][key] = []
        self.ENVVALSetTime[i] = {'Time': 0, 'Temp': 0}
        self.ENVRATE[i] = {'RateX': [], 'RateY': [], 'RateZ': [], 'Zero': []}
        self.ENVReward[i]['R'] = []
        self.ENVReward[i]['CurAcuR'] = 0

    def get_currentEP(self):
        get_db = self.StepInEachAgent
        return get_db

    def get_ENV_val(self, i):
        return self.ENVVALInEachAgent[i]

    def get_ENV_SetTime(self, i):
        return self.ENVVALSetTime[i]

    def get_ENV_RATE(self, i):
        return self.ENVRATE[i]

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
                         self.mem.get_ENV_SetTime(self.button.nub),
                         self.mem.get_ENV_RATE(self.button.nub),
                         self.mem.get_ENV_reward_val(self.button.nub))

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

        gs = GridSpec(3, 3, figure=self.fig)
        self.ax1 = self.fig.add_subplot(gs[0:4, 0:2], projection='3d')
        self.ax2 = self.fig.add_subplot(gs[0:1, 2:3])   # 에이전트 누적 Reward
        self.ax3 = self.fig.add_subplot(gs[1:3, 2:3])   # 현재 보상
        # self.ax4 = self.fig.add_subplot(gs[2:4, 2:3])   # 현재 보상

        FigureCanvas.__init__(self, self.fig)
        self.setParent(parent)

        FigureCanvas.setSizePolicy(self,
                                   QSizePolicy.Expanding,
                                   QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)

    def plot(self, val, settime, rate, reward_mem):
        self.ax1.clear()
        self.ax2.clear()
        self.ax3.clear()
        if True:
            Temp = []
            UpPres = []
            BotPres = []
            for _ in range(0, 350):
                uppres, botpres = PTCureve()._get_pres(_)
                Temp.append([_])
                UpPres.append([uppres])
                BotPres.append([botpres])

            PTX = np.array(Temp)
            BotZ = np.array(BotPres)
            UpZ = np.array(UpPres)
            PTY = np.array([[0] for _ in range(0, 350)])

            PTX = np.hstack([PTX[:, 0:1], Temp])
            BotZ = np.hstack([BotZ[:, 0:1], BotPres])
            UpZ = np.hstack([UpZ[:, 0:1], UpPres])
            PTY = np.hstack([PTY[:, 0:1], np.array([[val['KCNTOMS'][-1]] for _ in range(0, 350)])])


            self.ax1.plot3D(rate['RateX'], rate['RateY'], rate['RateZ'], color='orange', lw=1.5, ls='--')
            self.ax1.plot3D([170, 0, 0, 170, 170],
                            [val['KCNTOMS'][-1], val['KCNTOMS'][-1], 0, 0, val['KCNTOMS'][-1]],
                            [29.5, 29.5, 29.5, 29.5, 29.5], color='black', lw=0.5, ls='--')
            self.ax1.plot3D([170, 0, 0, 170, 170],
                            [val['KCNTOMS'][-1], val['KCNTOMS'][-1], 0, 0, val['KCNTOMS'][-1]],
                            [17, 17, 17, 17, 17], color='black', lw=0.5, ls='--')
            self.ax1.plot3D([170, 170], [val['KCNTOMS'][-1], val['KCNTOMS'][-1]],
                            [17, 29.5], color='black', lw=0.5, ls='--')
            self.ax1.plot3D([170, 170], [0, 0], [17, 29.5], color='black', lw=0.5, ls='--')
            self.ax1.plot3D([0, 0], [val['KCNTOMS'][-1], val['KCNTOMS'][-1]], [17, 29.5], color='black', lw=0.5, ls='--')
            self.ax1.plot3D([0, 0], [0, 0], [17, 29.5], color='black', lw=0.5, ls='--')

            self.ax1.plot_surface(PTX, PTY, UpZ, rstride=8, cstride=8, alpha=0.15, color='r')
            self.ax1.plot_surface(PTX, PTY, BotZ, rstride=8, cstride=8, alpha=0.15, color='r')

            # 3D plot
            self.ax1.plot3D(val['UAVLEG2'], val['KCNTOMS'], val['ZINST65'], color='blue', lw=1.5)

            # linewidth or lw: float
            self.ax1.plot3D([val['UAVLEG2'][-1], val['UAVLEG2'][-1]],
                            [val['KCNTOMS'][-1], val['KCNTOMS'][-1]],
                            [0, val['ZINST65'][-1]], color='blue', lw=0.5, ls='--')
            self.ax1.plot3D([0, val['UAVLEG2'][-1]],
                            [val['KCNTOMS'][-1], val['KCNTOMS'][-1]],
                            [val['ZINST65'][-1], val['ZINST65'][-1]], color='blue', lw=0.5, ls='--')
            self.ax1.plot3D([val['UAVLEG2'][-1], val['UAVLEG2'][-1]],
                            [0, val['KCNTOMS'][-1]],
                            [val['ZINST65'][-1], val['ZINST65'][-1]], color='blue', lw=0.5, ls='--')
            # each
            self.ax1.plot3D(val['UAVLEG2'], val['KCNTOMS'], rate['Zero'], color='black', lw=1, ls='--')  # temp
            self.ax1.plot3D(rate['Zero'], val['KCNTOMS'], val['ZINST65'], color='black', lw=1, ls='--')  # pres
            self.ax1.plot3D(val['UAVLEG2'], rate['Zero'], val['ZINST65'], color='black', lw=1, ls='--')  # PT

            # 절대값 처리
            self.ax1.set_yticklabels([int(_) for _ in abs(self.ax1.get_yticks())])

            self.ax1.set_xlabel('Temperature')
            self.ax1.set_ylabel('Time [Tick]')
            self.ax1.set_zlabel('Pressure')

            self.ax1.set_xlim(0, 350)
            self.ax1.set_zlim(0, 200)

        if True:
            self.ax2.plot(reward_mem[1])    # 'AcuR 전체
            self.ax3.plot(reward_mem[0])
            pass
        self.fig.set_tight_layout(True)
        self.fig.canvas.draw()

    def saveFig(self):
        self.fig.savefig('SaveFIg.png', dpi=100)