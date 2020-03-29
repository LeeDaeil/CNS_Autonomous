# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'Monitoring_tool.ui'
#
# Created by: PyQt5 UI code generator 5.14.1
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_Dialog(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(1081, 921)
        self.horizontalLayoutWidget = QtWidgets.QWidget(Dialog)
        self.horizontalLayoutWidget.setGeometry(QtCore.QRect(10, 39, 1061, 871))
        self.horizontalLayoutWidget.setObjectName("horizontalLayoutWidget")
        self.DIS = QtWidgets.QHBoxLayout(self.horizontalLayoutWidget)
        self.DIS.setContentsMargins(0, 0, 0, 0)
        self.DIS.setObjectName("DIS")
        self.comboBox = QtWidgets.QComboBox(Dialog)
        self.comboBox.setGeometry(QtCore.QRect(10, 10, 221, 22))
        self.comboBox.setObjectName("comboBox")

        self.retranslateUi(Dialog)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Dialog"))
