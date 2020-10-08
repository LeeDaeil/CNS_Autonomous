from struct import unpack, pack
import socket
import numpy as np
from PyQt5.QtWidgets import *
from PyQt5.QtCore import Qt, QTimer
from numpy.ma.bench import timer

import BACK.PID_UI8 as PID_UI8

class subWindows(QMainWindow):
    def __init__(self, mem, pid, cns_ip='', cns_port=7001):
        QMainWindow.__init__(self)
        self.ui = PID_UI8.Ui_Form()
        self.ui.setupUi(self)

        self.pid = pid
        self.kp = pid.Kp
        self.kd = pid.Kd
        self.ki = pid.Ki
        self.setpoint = pid.SetPoint_pres

        self.mem = mem
        self.main_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.address_cns = (cns_ip, cns_port)

        self.ui.pushButton_3.clicked.connect(self.Setpointup)
        self.ui.pushButton_4.clicked.connect(self.Setpointdown)
        self.ui.pushButton_5.clicked.connect(self.Kpup)
        self.ui.pushButton.clicked.connect(self.Kpdown)
        self.ui.pushButton_6.clicked.connect(self.Kdup)
        # self.ui.pushButton_2.clicked.connect(self.kddown)
        self.ui.pushButton_7.clicked.connect(self.Kiup)
        self.ui.pushButton_8.clicked.connect(self.Kidown)
        self.ui.pushButton_9.clicked.connect(self.Kpup1)
        # self.ui.pushButton_10.clicked.connect(self.Kdup1)
        # self.ui.pushButton_11.clicked.connect(self.Kiup1)
        # self.ui.pushButton_12.clicked.connect(self.Kpdown1)
        # self.ui.pushButton_13.clicked.connect(self.Kddown1)
        # self.ui.pushButton_14.clicked.connect(self.Kidown1)
        # self.ui.pushButton_15.clicked.connect(self.spdown1)
        # self.ui.pushButton_16.clicked.connect(self.spup1)



        timer = QTimer(self)
        timer.timeout.connect(self.update_mem)
        timer.start(1000)

    def send_control(self, para, val):
        UDP_header = b'\x00\x00\x00\x10\xa8\x0f'
        buffer = b'\x00' * 4008
        temp_data = b''

        for i in range(np.shape(para)[0]):
            pid_temp = b'\x00' * 12
            pid_temp = bytes(para[i], 'ascii') + pid_temp[len(para[i]):]

            para_sw = '12sihh' if self.mem[para[i]]['type'] == 0 else '12sfhh'

            temp_data += pack(para_sw,
                              pid_temp,
                              val[i],
                              self.mem[para[i]]['type'],
                              i)

        buffer = UDP_header + pack('h', np.shape(para)[0]) + temp_data + buffer[len(temp_data):]
        self.main_socket.sendto(buffer, self.address_cns)

    def update_mem(self):
        self.ui.lcdNumber.display(self.setpoint)
        self.ui.lcdNumber_2.display(self.mem['ZINST65']['V'])
        self.ui.lcdNumber_3.display(self.mem['BHV142']['V']*100)
        self.ui.lcdNumber_4.display(self.kp)
        self.ui.lcdNumber_5.display(self.kd)
        self.ui.lcdNumber_6.display(self.ki)
        self.ui.lcdNumber_7.display(self.mem['UUPPPL']['V']+self.mem['KCNTOMS']['V']/60000)
        self.ui.lcdNumber_8.display(self.mem['BHV603']['V']*100)
        self.ui.lcdNumber_9.display(self.mem['UUPPPL']['V'])
        self.ui.lcdNumber_10.display(self.kd)
        self.ui.lcdNumber_11.display(self.kp)
        self.ui.lcdNumber_12.display(self.ki)
        self.ui.lcdNumber_13.display(self.mem['UPRZ']['V'])
        self.ui.lcdNumber_14.display(self.mem['ZINST63']['V'])



        self.ui.progressBar.setValue(self.setpoint)
        self.ui.progressBar_2.setValue(self.mem['ZINST65']['V'])
        self.ui.progressBar_3.setValue(self.mem['BHV142']['V']*100)
        self.ui.progressBar_4.setValue(self.mem['UUPPPL']['V']+self.mem['KCNTOMS']['V']/60000)
        self.ui.progressBar_5.setValue(self.mem['BHV603']['V']*100)
        self.ui.progressBar_6.setValue(self.mem['UUPPPL']['V'])




    def Setpointup(self):
        print('이전', self.setpoint)
        self.setpoint += 1
        print('이후', self.setpoint)
        print("Call SETPOINTUP")
        self.update_mem()

    def Setpointdown(self):
        print('이전', self.setpoint)
        self.setpoint -= 1
        print('이후', self.setpoint)
        print("Call SETPOINTDOWN")
        self.update_mem()

    def Kpup(self):
        print('이전', self.kp)
        self.kp += 0.1
        print('이후', self.kp)
        print("Call Kp")
        self.update_mem()

    def Kpdown(self):
        print('이전', self.kp)
        self.kp -= 0.1
        print('이후', self.kp)
        print("Call Kp")
        self.update_mem()
    #
    def Kdup(self):
        print('이전', self.kd)
        self.kd += 0.1
        print('이후', self.kd)
        print("Call Kd")
        self.update_mem()
    #
    def Kddown(self):
        print('이전', self.kd)
        self.kd -= 0.1
        print('이후', self.kd)
        print("Call Kd")
        self.update_mem()
    #
    def Kiup(self):
        print('이전', self.ki)
        self.ki += 0.1
        print('이후', self.ki)
        print("Call Ki")
        self.update_mem()
    #
    def Kidown(self):
        print('이전', self.ki)
        self.ki -= 0.1
        print('이후', self.ki)
        print("Call Ki")
        self.update_mem()

    def Kpup1(self):
        print('이전', self.kp)
        self.kp += 0.1
        print('이후', self.kp)
        print("Call Kp")
        self.update_mem()