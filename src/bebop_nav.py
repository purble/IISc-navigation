#!/usr/bin/env python
import rospy, sys
from geometry_msgs.msg import Twist
from iisc_bebop_nav.msg import Bebop_cmd
from std_msgs.msg import Empty
from bebop_msgs.msg import CommonCommonStateBatteryStateChanged

class bebop_navigation(object):

	def __init__(self):

		# Subscriber for actions to be executed by Bebop
		self.action_sub = rospy.Subscriber("actions", Bebop_cmd, self.action_callback)

		# Initiate publishers for sending commands to Bebop
		self.cmd_pub = rospy.Publisher('cmd', Twist, queue_size=10)
		self.land_pub = rospy.Publisher('land', Empty, queue_size=100)
		self.takeoff_pub = rospy.Publisher('takeoff', Empty, queue_size=100)
		self.cam_pub = rospy.Publisher('cam_control', Twist, queue_size=100)

		# Subscribe to battery data for landing under critical battery state
		self.battery_sub =  rospy.Subscriber("battery", CommonCommonStateBatteryStateChanged, self.battery_callback)

		# Initialize values for different manoeuvres
		self.gostraight_val = rospy.get_param('/iisc_bebop_nav/gostraight_val')
		self.gostraight_slow_val = rospy.get_param('/iisc_bebop_nav/gostraight_slow_val')
		self.turnleft_val = rospy.get_param('/iisc_bebop_nav/turnleft_val')
		self.turnright_val = rospy.get_param('/iisc_bebop_nav/turnright_val')
		self.goup_val = rospy.get_param('/iisc_bebop_nav/goup_val')
		self.godown_val = rospy.get_param('/iisc_bebop_nav/godown_val')
		self.goleft_val = rospy.get_param('/iisc_bebop_nav/goleft_val')
		self.goright_val = rospy.get_param('/iisc_bebop_nav/goright_val')

	def action_callback(self, data):
		
		string = data.action
		twist = data.twist

		if string == "Land":
			self.land()
		elif string == "TakeOff":
			self.takeoff()
		elif string == "GoStraight":
			self.gostraight()
		elif string == "GoStraightSlow":
			self.gostraightslow()
		elif string == "TurnLeft":
			self.turnleft()
		elif string == "TurnRight":
			self.turnright()
		elif string == "Hover":
			self.hover()
		elif string == "GoUp":
			self.goup()
		elif string == "GoDown":
			self.godown()
		elif string == "GoLeft":
			self.goleft()
		elif string == "GoRight":
			self.goright()
		elif string == "Cam":
			self.cam(twist)
		elif string == "Custom":
			self.custom(twist)

	def twist_obj(self, lis):

		cmd = Twist()
		cmd.linear.x = lis[0]
		cmd.linear.y = lis[1]
		cmd.linear.z = lis[2]
		cmd.angular.x = lis[3]
		cmd.angular.y = lis[4]
		cmd.angular.z = lis[5]
		return cmd

	def land(self):
		emp = Empty()
		self.land_pub.publish(emp)

	def takeoff(self):
		emp = Empty()
		self.takeoff_pub.publish(emp)

	def hover(self):
		l = [0.0]*6
		self.cmd_pub.publish(self.twist_obj(l))

	def gostraight(self):
		l = [0.0]*6
		l[0] = self.gostraight_val
		self.cmd_pub.publish(self.twist_obj(l))

	def gostraightslow(self):
		l = [0.0]*6
		l[0] = self.gostraight_slow_val
		self.cmd_pub.publish(self.twist_obj(l))

	def turnleft(self):
		l = [0.0]*6
		l[5] = self.turnleft_val
		self.cmd_pub.publish(self.twist_obj(l))

	def turnright(self):
		l = [0.0]*6
		l[5] = self.turnright_val
		self.cmd_pub.publish(self.twist_obj(l))

	def goup(self):
		l = [0.0]*6
		l[2] = self.goup_val
		self.cmd_pub.publish(self.twist_obj(l))

	def godown(self):
		l = [0.0]*6
		l[2] = self.godown_val
		self.cmd_pub.publish(self.twist_obj(l))

	def goleft(self):
		l = [0.0]*6
		l[1] = self.goleft_val
		self.cmd_pub.publish(self.twist_obj(l))

	def goright(self):
		l = [0.0]*6
		l[1] = self.goright_val
		self.cmd_pub.publish(self.twist_obj(l))

	def cam(self, twist):
		cmd = self.twist_obj(twist)
		self.cam_pub.publish(cmd)

	def custom(self, twist):
		cmd = self.twist_obj(twist)
		self.cmd_pub.publish(cmd)

	def battery_callback(self, data):
		if (data.percent < 10.0): self.land()

def main(args):
	# Initialize Node class
	rospy.init_node('IISc_bebop_nav', anonymous=True)
	b_n = bebop_navigation()
	
	# Spin the node
	try:
		rospy.spin()
	# Stop the node if Ctrl-C pressed
	except KeyboardInterrupt:
		print("Shutting down")
		cv2.destroyAllWindows()

if __name__ == '__main__':

	print("Hello world from iisc_bebop_nav node!")
	main(sys.argv)