#!/usr/bin/env python
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
from iisc_autonav_outdoors.msg import UWB_pos
import paho.mqtt.client as mqtt

MQTT_HOST = "gateways.rbccps.org"
MQTT_PORT = 1883
MQTT_KEEPALIVE_INTERVAL = 0

MQTT_TOPIC = "uwb"
MQTT_USERNAME = "loraserver"
MQTT_PASSWORD = "loraserver"



def uwbMsg(wrtAnchors, curr_pos):
    msg = UWB_pos()
    msg.header.stamp = rospy.Time.now()
    msg.wrtAnchors = wrtAnchors
    msg.curr_pos = curr_pos
    return msg

# Define on_publish event function
def on_publish(client, userdata, mid):
    print "Message Published..."

def on_connect(client, userdata, flags, rc):
    client.subscribe(MQTT_TOPIC)
    client.publish(MQTT_TOPIC, MQTT_MSG)


def on_message(client, userdata, msg):
    print(msg.payload) 
    msg = [float(i) for i in line.split(',')]
    uwbPos.publish(uwbMsg(msg[0], msg[1:3]))


# Initiate MQTT Client
mqttc = mqtt.Client()


mqttc.on_publish = on_publish
mqttc.on_connect = on_connect
mqttc.on_message = on_message
mqttc.username_pw_set(MQTT_USERNAME, MQTT_PASSWORD)



mqttc.connect(MQTT_HOST, MQTT_PORT, MQTT_KEEPALIVE_INTERVAL)



rospy.init_node('uwb', anonymous=True)
uwbPos = rospy.Publisher('uwb', UWB_pos, queue_size=10)
mqttc.loop_forever()


