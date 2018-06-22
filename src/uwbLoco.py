#!/bin/python

import time
import rospy
from geometry_msgs.msg import Twist
import sys
import signal
from std_msgs.msg import Empty
import numpy as np
import serial
from std_msgs.msg import Empty
import os
import math
from datetime import datetime
import json
from iisc_bebop_nav.msg import uwb_pos


#Parameters to be otained from ROSParamServer
anchorSigmas = [0.000803, 0.000502, 0.000487]
anchorOffsets = [-0.61, -0.58, -0.70]
numAnchors = 3
anchor1Pos = (0.0, 0.0)
anchor2Pos = (0.0, 6.0)
anchor3Pos = (6.0, 6.0)
wrtAnchors = [1,2,3]


def uwbMsg(wrtAnchors, curr_pos):
    msg = uwb_pos()
    msg.header.stamp = rospy.Time.now()
    msg.wrtAnchors = wrtAnchors
    msg.curr_pos = curr_pos
    return msg


# Assume origin is p1
def getPos(p1, p2, p3, r1, r2, r3):

    A = -2*p1[0] + 2*p2[0]
    B = -2*p1[1] + 2*p2[1]
    C = r1**2 - r2**2 - p1[0]**2 + p2[0]**2 - p1[1]**2 + p2[1]**2

    D = -2*p2[0] + 2*p3[0]
    E = -2*p2[1] + 2*p3[1]
    F = r2**2 - r3**2 - p2[0]**2 + p3[0]**2 - p2[1]**2 + p3[1]**2

    pos = (round((C*E - F*B)/(E*A - B*D), 2),
           round((C*D - A*F)/(B*D - A*E), 2))
    return pos


def uwb():

    
    ser = serial.Serial("/dev/ttyACM0", 576000)
    uwbPos = rospy.Publisher('uwb', uwb_pos, queue_size=10)

    radialPos = []
    prevRadialPos = []

    line = ser.readline()  # Read starting few junk lines if any
    line = ser.readline()
    line = ser.readline()  # Consistent readings start here

    radialPos = [float(i) for i in line.split(',')]
    prevRadialPos = radialPos

    while(not is rospy.is_shutdown()):

        try:
            line = ser.readline()
            radialPos = [float(i) for i in line.split(',')]
            radialPos = [radialPos[z]*alpha +
                         prevRadialPos[z]*(1-alpha) for z in range(0, numAnchors)]
            prevRadialPos = radialPos
            datx, daty = getPos(anchor1Pos, anchor2Pos, anchor3Pos,
                                radialPos[0]+anchorOffsets[0], radialPos[1]+anchorOffsets[1], radialPos[2]+anchorOffsets[2])
            uwbPos.publish(uwbMsg(wrtAnchors, [datx,daty]))
        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(
                exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)
            print e
    except ValueError:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)


if __name__ == '__main__':
    try:
        uwb()
    except rospy.ROSInterruptException:
        print("ROS master not started")
        pass
