#!/usr/bin/env python
import rospy, sys, cv2
from sensor_msgs.msg import Image
from iisc_bebop_nav.msg import Bebop_cmd
from cv_bridge import CvBridge, CvBridgeError
from ar_track_alvar_msgs.msg import AlvarMarker, AlvarMarkers
from nav_msgs.msg import Odometry

class juncTurn(object):

	def __init__(self):

		# Convert ROS image messages to OpenCV images
		self.bridge = CvBridge()

		# Publisher for actions to be executed by Bebop
		self.action_pub = rospy.Publisher("actions", Bebop_cmd, queue_size=10)

		# Pulisher for output image with renderings etc.
		self.image_pub = rospy.Publisher("output_image", Image, queue_size=10)

		# Image subscriber
		self.image_sub = rospy.Subscriber("camera", Image, self.image_callback)

		# Fiducial aided turning enabled
		if rospy.get_param('/iisc_junc_turning/turningWithFiducial'):
			# Fiducial marker info subscriber
			self.fiducial_sub = rospy.Subscriber("fiducial_marker", AlvarMarkers, self.fiducial_callback)
			# Initialize landing pad marker detection history list
			self.junctionTurning_fiducial_detected_list = [0]*rospy.get_param('/iisc_junc_turning/junctionTurning_fiducial_detected_list_size')
			self.junctionTurning_fiducial_detection_thresh = rospy.get_param('/iisc_junc_turning/junctionTurning_fiducial_detection_thresh')

		# Subscribe to odometry data, for turning using magnetometer
		self.odom_sub = rospy.Subscriber("odom", Odometry, self.odom_callback)

		# Turning Right, Left and Go Straight flags for appropriate renderings on output image
		self.turnR = False
		self.turnL = False
		self.goSt = False

		# Rotation flag for turning at junction
		self.rotate = False

		# Eps range round target yaw value to achieve
		self.eps = 0.04

		# Junctions' calibration
		self.pad_ids = rospy.get_param('/iisc_junc_turning/pad_ids')
		self.pad_turn_dir = rospy.get_param('/iisc_junc_turning/pad_turn_dir')
		self.pad_dist_det = rospy.get_param('/iisc_junc_turning/pad_dist_det')
		self.pad_dist_turn = rospy.get_param('/iisc_junc_turning/pad_dist_turn')
		self.pad_yaw = rospy.get_param('/iisc_junc_turning/yaw_values')

		# Active junction index
		self.idx = None

		# Store the markers seen previously, so that no action taken if seen again
		self.done_pad_ids = []

	def image_callback(self, data):

		# If state is not 2 continue
		if rospy.get_param('/nav_state') != 2: return

		# Convert streamed image data to bgr image
		try:
			cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
		except CvBridgeError as e:
			print(e)

		# Render appropriate graphics on output image signalling turning at junctions
		if self.rotate:
			if self.pad_turn_dir[self.idx]:
				cv2.circle(cv_image, (806,50), 50, (0,255,255), -1)
			else:
				cv2.circle(cv_image, (50,50), 50, (0,255,255), -1)
		elif self.goSt:
			cv2.circle(cv_image, (428,50), 50, (0,255,255), -1)

		# Publish the output image
		try:
			rospy.loginfo("State 2")
			self.image_pub.publish(self.bridge.cv2_to_imgmsg(cv_image, "bgr8"))
		except CvBridgeError as e:
			print(e)

	def fiducial_callback(self, data):

		# If landing or rotating then ignore the fiducial markers
		if rospy.get_param('/nav_state') == 0 or self.rotate: return
		# Otherwise check if any pad in the self.pad_ids list is detected in current view

		# Check if any marker is detected
		if  len(data.markers)>0: # I think this is redundant since there will be no callback in the absence of any data
			for marker in data.markers:
				cur_id = marker.id
				print("hello0")
				if cur_id in self.pad_ids and cur_id not in self.done_pad_ids:
					print("hello1")
					
					if self.robust_detection():

						if self.idx is None:
							self.idx = self.pad_ids.index(cur_id)

						cur_dist = marker.pose.pose.position.z

						if cur_dist < self.pad_dist_turn[self.idx]:
							self.rotate = True
							self.goSt = False
							self.done_pad_ids = [cur_id] + self.done_pad_ids
							if self.pad_turn_dir[self.idx]:
								self.turnR = True
							else:
								self.turnL = True
						elif self.goSt:
							print("hello4")
							msg = self.msg_pub('GoStraight', [0.0]*6)
						elif cur_dist < self.pad_dist_det[self.idx]:
							print("hello5")
							rospy.set_param('/nav_state', 2)
							self.goSt = True

					break

	def robust_detection(self, reset=False):

		if reset:
			self.junctionTurning_fiducial_detected_list = [0]*rospy.get_param('/iisc_junc_turning/junctionTurning_fiducial_detected_list_size')
		elif rospy.get_param('/nav_state') == 2:
			return True
		else:
			self.junctionTurning_fiducial_detected_list.pop()
			self.junctionTurning_fiducial_detected_list = [1] + self.junctionTurning_fiducial_detected_list
			if sum(self.junctionTurning_fiducial_detected_list) > self.junctionTurning_fiducial_detection_thresh:
				return True
			else:
				return False

	def msg_pub(self, str, lis):
		msg = Bebop_cmd()
		msg.header.stamp = rospy.Time.now()
		msg.action = str
		msg.twist = lis
		self.action_pub.publish(msg)
		return

	def odom_callback(self, data):
		if not self.rotate: return
		cur_yaw = data.pose.pose.orientation.z

		tar_yaw = self.pad_yaw[self.idx]

		ulim_yaw = tar_yaw + self.eps
		llim_yaw = tar_yaw - self.eps

		# Check if the current yaw is between the two limits
		if (ulim_yaw <= 1.0) and (llim_yaw >= -1.0) and (cur_yaw > (llim_yaw)) and (cur_yaw < (ulim_yaw)):
			self.rotate = False
		elif (ulim_yaw > 1.0):
			if (cur_yaw < 0.0): cur_yaw = 2 - cur_yaw
			if (cur_yaw < ulim_yaw) and (cur_yaw > llim_yaw):
				self.rotate = False
		elif (llim_yaw < -1.0): # (rot < ulim) will anyways hold true, since rot = [-1,1]
			if (cur_yaw > 0.0): cur_yaw = -2 + cur_yaw
			if (cur_yaw > llim_yaw and cur_yaw < ulim_yaw):
				self.rotate = False

		if self.rotate:
			if self.turnR:
				msg = self.msg_pub('TurnRight', [0.0]*6)
			elif self.turnL:
				msg = self.msg_pub('TurnLeft', [0.0]*6)
		else:
			# Switch to road following state
			rospy.set_param('/nav_state', 1)
			# Reset the junction marker detection queue
			self.robust_detection(True)
			# Reset the current pad index
			self.idx = None

def main(args):
	# Initialize Node class
	rospy.init_node('IISc_junc_turning', anonymous=True)
	j_t = juncTurn()
	
	# Spin the node
	try:
		rospy.spin()
	# Stop the node if Ctrl-C pressed
	except KeyboardInterrupt:
		print("Shutting down")
		cv2.destroyAllWindows()

if __name__ == '__main__':

	print("Hello world from IISc_junc_turning!")
	main(sys.argv)
