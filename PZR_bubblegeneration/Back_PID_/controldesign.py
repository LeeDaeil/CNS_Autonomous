from struct import unpack, pack
from datetime import datetime
import socket
import numpy as np
import PID_Na
from db import db_make as db_dict
from numpy import shape
import sys
from PyQt5.QtWidgets import *
from PyQt5.QtGui import QPainter, QBrush, QPen, QFont
from PyQt5.QtCore import Qt, QTimer
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib import animation
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QMessageBox
from PyQt5 import QtCore, QtGui, QtWidgets
from time import sleep
from PyQt5.QtCore import QDate, Qt

from BACK.PID import subWindows

class AnimationWidget(QWidget):
    def __init__(self, ip, port, cns_ip, cns_port, kp, ki, kd):
        QMainWindow.__init__(self)
        self.setWindowTitle('원자로 기동운전')

        # 메모리 구조 선언
        self.mem = db_dict().make_db_structure(len_deque=2)
        # 소켓 생성 및 초기화
        self.address = (ip, port)
        self.address_cns = (cns_ip, cns_port)
        self.main_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.main_socket.bind(self.address)
        self.main_socket.settimeout(5)  # 5초 동안 연결 없으면 error
        #
        # i = 0
        self.event_condition = []
        self.sample_time = 1.0
        self.pid = PID_Na.PID(kp, ki, kd)
        self.pid.SetPoint_pres = 25.0

        self.time_list = []
        self.control_list = []
        self.pressure_list = []
        self.heat_rate_list = []
        self.rcs_temp_list = []
        self.prz_level_list = []
        self.charging_position = []  # charging valve position list
        self.letdown_position = []  # letdown valve position list

        self.start_time = datetime.now()
        self.stop_time = datetime.now()
        self.total_time = self.start_time - self.stop_time
        self.middle_time = self.total_time

        vbox = QVBoxLayout()
        box = QGridLayout()
        self.start_button = QPushButton("가열운전 제어 시작 버튼", self)
        self.start_button.setStyleSheet('color:blue; background:white; font:bold; font-size:20px')
        box.addWidget(self.start_button, 0, 0)

        self.stop_button = QPushButton("가열운전 제어 종료 버튼", self)
        self.stop_button.setStyleSheet('color:blue; background:white; font:bold; font-size:20px')
        box.addWidget(self.stop_button, 2, 0)

        self.start_button.clicked.connect(self.on_start)
        self.stop_button.clicked.connect(self.on_stop)

        self.blank = QLabel('', self)
        self.blank.setStyleSheet('font-size:1px')
        box.addWidget(self.blank, 1, 0)

        self.warningLabel = QLabel('운전 상황 표시', self)
        self.warningLabel.setAlignment(Qt.AlignCenter)
        self.warningLabel.setMargin(5)
        self.warningLabel.setStyleSheet("color:blue; background:white; font:bold; font-size:20px;"
                                        "border-style:solid; border-width:1px; border-color:black; border-radius:3px")
        box.addWidget(self.warningLabel, 0, 1)

        self.warningCondition = QLabel('        정상                         경보', self)
        # self.warningCondition.setStyleSheet('color:white; background:green; font:bold; font-size:30px')
        self.warningCondition.setStyleSheet('font:bold; font-size:20px')
        box.addWidget(self.warningCondition, 2, 1)
        #
        self.blank = QLabel('', self)
        self.blank.setStyleSheet('font-size:1px')
        box.addWidget(self.blank, 3, 0)

        # self.warningCondition = QLabel('운전시간', self)
        # self.warningCondition.setStyleSheet('color:white; background:green; font:bold; font-size:30px')
        self.warningCondition.setStyleSheet('font:bold; font-size:20px')
        box.addWidget(self.warningCondition, 4, 0)

        vbox.addLayout(box)

        self.canvas = MyMplCanvas(self, width=8, height=8, dpi=100)

        vbox.addWidget(self.canvas)
        self.setLayout(vbox)

        self.x11 = np.arange(500)
        self.y11 = np.ones(500, dtype=np.float) * np.nan
        self.line11, = self.canvas.axes11.plot(self.x11, self.y11, animated=True, color='blue', lw=2)
        self.x12 = np.arange(500)
        self.y12 = np.ones(500, dtype=np.float) * np.nan
        self.line12, = self.canvas.axes12.plot(self.x12, self.y12, animated=True, color='red', lw=2)

        self.x21 = np.arange(500)
        self.y21 = np.ones(500, dtype=np.float) * np.nan
        self.line21, = self.canvas.axes21.plot(self.x21, self.y21, animated=True, color='blue', lw=2)
        self.x22 = np.arange(500)
        self.y22 = np.ones(500, dtype=np.float) * np.nan
        self.line22, = self.canvas.axes22.plot(self.x22, self.y22, animated=True, color='red', lw=2)

        self.show()
        # Sub window
        self.Subwindow = subWindows(mem=self.mem, pid=self.pid, cns_ip='192.168.100.45', cns_port=7001)
        self.Subwindow.show()


    def initial(self):
        return self.line11, self.line12, self.line21, self.line21

    def update_line(self, i):

        self.current_time = datetime.now()
        self.total_time = self.middle_time + (self.current_time - self.stop_time) - (self.new_start_time - self.stop_time)

        # i = i+1
        # 1. 통신으로 데이터 받기
        temp_data, add = self.main_socket.recvfrom(44388)

        # 2. 받은 데이터 메모리에 업데이트
        self.update_mem(temp_data[8:])

        # 3. 메모리 데이터를 사용하여 계산
        time_current = float(i * self.sample_time)
        # if i != 0:  # 0이 아닐 때 계속 반복

        self.pid.Kp = self.Subwindow.kp
        self.pid.Ki = self.Subwindow.ki
        self.pid.Kd = self.Subwindow.kd
        self.pid.SetPoint_pres = self.Subwindow.setpoint

        error_pres = self.pid.SetPoint_pres - self.mem['ZINST65']['V']
        self.pid.update(error_pres, self.sample_time)
        valve_control = self.pid.output

        pres = self.mem['ZINST65']['V']  # pressure (random number generation)
        rcs_temp = self.mem['UUPPPL']['V']
        prz_level = self.mem['ZINST63']['V']
        pos1 = self.mem['BFV122']['V']  # charging valve position
        pos2 = self.mem['BHV142']['V']  # letdown valve position
        self.pressure_list.append(pres)  # pressure list
        self.charging_position.append(pos1)  # charging valve position list
        self.letdown_position.append(pos2)  # letdown valve position list
        self.rcs_temp_list.append(rcs_temp)
        self.prz_level_list.append(prz_level)

        self.control_list.append(valve_control)  # control rod reactivity vector

        if i <= 70:
            print(i, valve_control, pres, error_pres, rcs_temp, prz_level)

        # 4. 데이터 보내기
        # self.send_control(LM08RE113, 0)
        if prz_level > 90:
            if valve_control >= 0.005:
                self.send_control(['KSWO100', 'KSWO101', 'KSWO102'], [1, 0, 1])  # Charging Valve Open
                if i % 3 == 0:
                    self.send_control(['KSWO231', 'KSWO232'], [1, 0])  # Letdown Valve close
            elif abs(valve_control) < 0.005:
                self.send_control(['KSWO100', 'KSWO101', 'KSWO102'], [1, 0, 0])  # Charging no action
                self.send_control(['KSWO231', 'KSWO232'], [0, 0])  # Letdown no action
            else:
                self.send_control(['KSWO100', 'KSWO101', 'KSWO102'], [1, 1, 0])  # Charging Valve close
                if i % 3 == 0:
                    self.send_control(['KSWO231', 'KSWO232'], [0, 1])  # Letdown Valve Open

        if prz_level <= 90:
            self.send_control(['KSWO130'], [1])  # RCP OIL PUMP
            self.send_control(['KSWO133'], [1])  # RCP ON
            if valve_control >= 0.005:
                self.send_control(['KSWO128', 'KSWO126', 'KSWO127'], [1, 1, 0])  # Spray Valve close
            #         # if i % 5 == 0:
            #         #     self.send_control(['KSWO231', 'KSWO232'], [1, 0])  # Letdown Valve close
            elif abs(valve_control) < 0.005:
                self.send_control(['KSWO128', 'KSWO126', 'KSWO127'], [1, 0, 0])  # Spray Valve no action
            else:
                self.send_control(['KSWO128', 'KSWO126', 'KSWO127'], [1, 0, 1])  # Spray Valve open
        # if i % 5 == 0:
        #     self.send_control(['KSWO231', 'KSWO232'], [0, 1])  # Letdown Valve Open

        # elif prz_level >= 100:
        #     self.send_control(['KSWO130'], [0])  # RCP OIL PUMP
        #     self.send_control(['KSWO133'], [0])  # RCP OFF

        if i > 70:
            heat_rate = (sum(self.rcs_temp_list[-10:-1]) / 10.0 - sum(self.rcs_temp_list[-69:-60]) / 10.0) / 60  # deg C/min
            # if charging_valve_list[-1] <= 0.15 and (sum(pressure_list[-5:-1]) / 5.0) > (sum(pressure_list[-64:-60])/5.0):  # pressure increasing
            #     if i % 30 == 0:
            #         self.send_control(['KSWO120', 'KSWO121', 'KSWO122'], [1, 1, 0])  # Heater down
            #
            # if charging_valve_list[-1] >= 0.95 and (sum(pressure_list[-5:-1]) / 5.0) < (sum(pressure_list[-64:-60])/5.0):  # pressure decreasing
            #     if i % 30 == 0:
            #         self.send_control(['KSWO120', 'KSWO121', 'KSWO122'], [1, 0, 1])  # Heater up

            print(i, valve_control, pres, error_pres, rcs_temp, prz_level, heat_rate)

        if rcs_temp > 176:  # if RCS temp > 177, RHRS is isolated
            self.send_control(['KSWO53'], [0])  # RHR Pump stop
            self.send_control(['KSWO57'], [0])  # RHR valve HV18  close

        # 5. 데이터 저장
        self.save_file()

        # 최종 출력
        y11 = pres                 # PRZ pressure
        self.save_y11 = y11
        old_y11 = self.line11.get_ydata()       #그래프 데이터
        new_y11 = np.r_[old_y11[1:], y11]
        self.line11.set_ydata(new_y11)

        y12 = rcs_temp          # RCS temperature
        old_y12 = self.line12.get_ydata()
        new_y12 = np.r_[old_y12[1:], y12]
        self.line12.set_ydata(new_y12)

        y21 = pos1              # charging valve position
        old_y21 = self.line21.get_ydata()
        new_y21 = np.r_[old_y21[1:], y21]
        self.line21.set_ydata(new_y21)

        y22 = pos2              # letdown valve position
        old_y22 = self.line22.get_ydata()
        new_y22 = np.r_[old_y22[1:], y22]
        self.line22.set_ydata(new_y22)

        if y11 >= 25.5 or y11 <= 23.5:
            self.event_condition = 1
            self.paintEvent(self.event_condition)
        else:
            self.event_condition = 0
            self.paintEvent(self.event_condition)

        return self.line11, self.line12, self.line21, self.line22

    def update_mem(self, data):
        pid_list = []
        for i in range(0, len(data), 20):
            sig = unpack('h', data[16 + i: 18 + i])[0]
            para = '12sihh' if sig == 0 else '12sfhh'
            pid, val, sig, idx = unpack(para, data[i:20 + i])
            pid = pid.decode().rstrip('\x00')  # remove '\x00'

            if pid != '':
                # 'NBANK': {'V': 0, 'L': [], 'type': 0},
                self.mem[pid]['V'] = val
                self.mem[pid]['D'].append(val)

    def send_control(self, para, val):
        UDP_header = b'\x00\x00\x00\x10\xa8\x0f'
        buffer = b'\x00' * 4008
        temp_data = b''

        for i in range(shape(para)[0]):
            pid_temp = b'\x00' * 12
            pid_temp = bytes(para[i], 'ascii') + pid_temp[len(para[i]):]

            para_sw = '12sihh' if self.mem[para[i]]['type'] == 0 else '12sfhh'

            temp_data += pack(para_sw,
                              pid_temp,
                              val[i],
                              self.mem[para[i]]['type'],
                              i)

        buffer = UDP_header + pack('h', shape(para)[0]) + temp_data + buffer[len(temp_data):]
        self.main_socket.sendto(buffer, self.address_cns)

    def save_file(self):
        line_data = ''
        with open('save_db.csv', 'w') as f:
            for pid_name in self.mem.keys():
                if len(self.mem[pid_name]['D']) > 0:
                    line_data += str(self.mem[pid_name]['V'])
                    line_data += ','
            line_data += '\n'
            f.write(line_data)

    def paintEvent(self, event):
        qp = QPainter(self)
        qp.setPen(QPen(Qt.black, 2, Qt.SolidLine))       # normal condition mark
        font = QFont()
        font.setFamily('Times')
        font.setPointSize(16)
        qp.setFont(font)
        qp.drawText(20, 100, 68, 30, Qt.AlignLeft, str(self.total_time))
        qp.drawEllipse(510, 60, 30, 30)
        qp.drawEllipse(700, 60, 30, 30)
        if self.event_condition == 1:
            qp.setBrush(QBrush(Qt.red, Qt.SolidPattern))
            qp.drawEllipse(700, 60, 30, 30)
        else:
            qp.setBrush(QBrush(Qt.darkGreen, Qt.SolidPattern))
            qp.drawEllipse(510, 60, 30, 30)

    def on_start(self, painter):
        self.new_start_time = datetime.now()
        timer = QTimer(self)  # paintEvent 발생
        timer.timeout.connect(self.update)
        timer.start(1000)
        self.ani = animation.FuncAnimation(fig=self.canvas.figure, func=self.update_line, init_func=self.initial,
                                           frames=50000, blit=True, interval=25)

    def on_stop(self):
        self.middle_time = self.total_time
        self.stop_time = datetime.now()
        self.ani._stop()

    def closeEvent(self, QCloseEvent):          #오버라이딩
        ans = QMessageBox.question(self, "종료 확인", "종료하시겠습니까?",
                                   QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
        if ans == QMessageBox.Yes:
            QCloseEvent.accept()
        else:
            QCloseEvent.ignore()

class MyMplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=8, height=8, dpi=300):
        fig = Figure(figsize=(width, height), dpi=dpi)

        self.axes11 = fig.add_subplot(211, xlim=(0, 500), ylim=(20.0, 35.0))
        self.axes11.grid(True)  # grid on
        self.axes11.set_ylabel('Pressure (kgf/cm$^2$)', fontsize=10, color="blue")
        self.axes11.set_xlabel('recent time steps(sec)', fontsize=10)
        # self.axes11.legend('pressure', loc='upper left', fontsize=10)

        self.axes12 = self.axes11.twinx()  # 2nd y-axis
        self.axes12.set_ylim(50.0, 250.0)  # temperature plot limit
        self.axes12.set_ylabel('Temperature ($^o$C)', fontsize=10, color="red")
        # self.axes12.legend('temperature', loc='upper right', fontsize=10)

        self.axes21 = fig.add_subplot(212, xlim=(0, 500), ylim=(-0.25, 1.25))
        self.axes21.grid(True)  # grid on
        self.axes21.set_ylabel('Charging valve opening (%)', fontsize=10, color="blue")
        self.axes21.set_xlabel('recent time steps(sec)', fontsize=10)

        self.axes22 = self.axes21.twinx()  # 2nd y-axis
        self.axes22.set_ylim(-0.25, 1.25)  # temperature plot limit
        self.axes22.set_ylabel('Letdown valve opening (%)', fontsize=10, color="red")

        self.compute_initial_figure()
        FigureCanvas.__init__(self, fig)
        self.setParent(parent)

    def compute_initial_figure(self):
        pass



if __name__ == '__main__':
    qApp = QApplication(sys.argv)
    qApp.setFont(QFont("나눔명조", 7))
    aw = AnimationWidget(ip='192.168.100.2', port=7000, cns_ip='192.168.100.45', cns_port=7001, kp=0.03, ki=0.001, kd=1.0)
    sys.exit(qApp.exec_())
