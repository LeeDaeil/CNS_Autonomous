# This file is a PID class file
# Copyright (C) 2015 Ivmech Mechatronics Ltd. <bilgi@ivmech.com>
#
# IvPID is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# IvPID is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

# title           : PID.py
# description     : python PID controller
# author          : Caner Durmusoglu
# date            :20151218
# modifier        : Man Gyun Na (20190901)
# ==============================================================================

# import time


class PID:
    """    PID Class   """

    def __init__(self, kp=0.2, ki=0.0, kd=0.0):

        self.Kp = kp   # proportional gain
        self.Ki = ki   # integral gain
        self.Kd = kd   # derivative gain
        self.SetPoint = 0

        #  self.sample_time = 0.00
        #  self.current_time = t0
        #  self.last_time = self.current_time

        self.clear()

    def clear(self):
        """Clears PID computations and coefficients"""
        # self.SetPoint = 0.0

        self.PAction = 0.0
        self.IAction = 0.0
        self.DAction = 0.0
        self.last_error = 0.0

        # Windup Guard
        self.int_error = 0.0
        self.windup_guard = 20.0

        self.output = 0.0

    def update(self, current_val, delta_time):
        """Calculates PID control output for a given error
            u(t) = K_p e(t) + K_i \int_{0}^{t} e(t)dt + K_d {de}/{dt}
        """
        # error = self.SetPoint - feedback_value

        # self.current_time = tf
        # delta_time = self.current_time - self.last_time

        error = self.SetPoint - current_val

        delta_error = error - self.last_error

        # if (delta_time >= self.sample_time):

        self.PAction = self.Kp * error
        self.IAction += error * delta_time

        if (self.IAction < -self.windup_guard):
            self.IAction = -self.windup_guard
        elif (self.IAction > self.windup_guard):
            self.IAction = self.windup_guard

        self.DAction = 0.0
        if delta_time > 0:
            self.DAction = delta_error / delta_time

        # Remember last time and last error for next calculation
        # self.last_time = self.current_time
            self.last_error = error

            self.output = self.PAction + (self.Ki * self.IAction) + (self.Kd * self.DAction)
        return self.output

    """def setKp(self, proportional_gain):
        self.Kp = proportional_gain

    def setKi(self, integral_gain):
        self.Ki = integral_gain

    def setKd(self, derivative_gain):
        self.Kd = derivative_gain"""

    def setWindup(self, windup):
        """Integral windup """
        self.windup_guard = windup

    """ def setSampleTime(self, sample_time):
        self.sample_time = sample_time """