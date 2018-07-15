#!/usr/bin/env python
import rospy, sys, math
from sensor_msgs.msg import Image
from iisc_bebop_nav.msg import Bebop_cmd
from cv_bridge import CvBridge, CvBridgeError
from nav_msgs.msg import Odometry
from ar_track_alvar_msgs.msg import AlvarMarker, AlvarMarkers
from shape_detection import *
from PIDcontrol import *

class landing(object):

	def __init__(self):

		# Convert ROS image messages to OpenCV images
		self.bridge = CvBridge()

		# Initialize PIDcontroller object
		self.pid_controller = PIDcontroller1D()

		# Initialize Shape Detector object
		self.shape_detector = ShapeDetector()

		''' Publishers for red, green, blue masks.
			Useful for debugging colored landing pad detection.
		'''
		self.red_pub = rospy.Publisher("red", Image, queue_size=10)
		self.green_pub = rospy.Publisher("green", Image, queue_size=10)
		self.blue_pub = rospy.Publisher("blue", Image, queue_size=10)

		# Publisher for actions to be executed by Bebop
		self.action_pub = rospy.Publisher("actions", Bebop_cmd, queue_size=10)

		# Pulisher for output image with renderings etc.
		self.image_pub = rospy.Publisher("output_image", Image, queue_size=10)

		# Image subscriber
		self.image_sub = rospy.Subscriber("camera", Image, self.image_callback)
		# Subscribe to odometry data, for turning using magnetometer
		self.odom_sub = rospy.Subscriber("odom", Odometry, self.odom_callback)
		# Initialize pad detection history list
		self.pad_detected_list = [0]*rospy.get_param('/iisc_landing/landingpad_fiducial_detected_list_size')

		# Store and update current yaw in radians using Odometry data
		self.cur_yaw_rad = 0.0

		# Fiducial aided landing enabled
		if rospy.get_param('/iisc_landing/landingWithFiducial'):
			# Fiducial marker info subscriber
			self.fiducial_sub = rospy.Subscriber("fiducial_marker", AlvarMarkers, self.fiducial_callback)
			# Initialize landing pad marker detection history list
			self.landingpad_fiducial_detected_list = [0]*rospy.get_param('/iisc_landing/landingpad_fiducial_detected_list_size')

		# Set camera view to horizontal
		# self.action_pub.publish(self.msg_gen('Cam', [0.0] * 6))

	### Initiate Landing ###

	def initiate_landing_procedure(self):
		self.action_pub.publish(self.msg_gen('Land', [0.0] * 6))
		self.action_pub.publish(self.msg_gen('Cam', [0.0] * 6))
		rospy.set_param('/iisc_landing/landed', True)

	### image_callback and it's utility functions

	def image_callback(self, data):

		# If turning at junction, do not look for pad
		if rospy.get_param('/nav_state')>1: return

		# Convert streamed image data to bgr image
		try:
			cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
		except CvBridgeError as e:
			print(e)

		''' Get: 
		res: whether pad was detected
		img: image with rendered markings if pad detected else original image
		r: red mask
		g: green mask
		b: blue mask
		'''
		res, img, r, g, b = self.shape_detector.process(cv_image)

		# Publish data to relevant topics
		try:
		    self.red_pub.publish(self.bridge.cv2_to_imgmsg(r, "8UC1"))
		except CvBridgeError as e:
			print(e)

		try:
		    self.green_pub.publish(self.bridge.cv2_to_imgmsg(g, "8UC1"))
		except CvBridgeError as e:
			print(e)

		try:
		    self.blue_pub.publish(self.bridge.cv2_to_imgmsg(b, "8UC1"))
		except CvBridgeError as e:
			print(e)

		# Update the landing pad detection list
		self.update_landing_pad_detection_list(res)

		# If landing has been initiated, first hover to take care of inertia, then activate PID control loop
		if rospy.get_param('/nav_state') == 0:

			# Publish from this node only if landing is initiated
			try:
				rospy.loginfo("State 0")
				self.image_pub.publish(self.bridge.cv2_to_imgmsg(img, "bgr8"))
			except CvBridgeError as e:
				print(e)

			# If already landed, just publish the rendered image without going any further
			if rospy.get_param('/iisc_landing/landed'): return

			if rospy.get_param('/iisc_landing/landingpad_detect_hover_count')>0:
				self.action_pub.publish(self.msg_gen('Hover', [0.0] * 6))
				self.update_landingpad_detect_hover_count()
			else:
				cmd = self.pid_controller.eval_actuation(self.cur_yaw_rad)
				self.action_pub.publish(cmd)

			if sum(self.pad_detected_list) < rospy.get_param('/iisc_landing/landingpad_notinFOV_thresh'):
				if rospy.get_param('/iisc_landing/distance2Lmarker') < rospy.get_param('/iisc_landing/landing_max_dist_thresh'): # If distance closer than landing_thresh_high then land
					self.initiate_landing_procedure()
				else: # Refill the landing list
					self.pad_detected_list = [1]*rospy.get_param('/iisc_landing/landingpad_fiducial_detected_list_size')
			elif rospy.get_param('/iisc_landing/distance2Lmarker') < rospy.get_param('/iisc_landing/landing_min_dist_thresh'): # If distance closer than landing_thresh_low then land
				self.initiate_landing_procedure()
		else:
			# If pad has been enough number of times, set 'nav_state' parameter to 0
			if sum(self.pad_detected_list) > rospy.get_param('/iisc_landing/landingpad_detection_thresh'):
				rospy.set_param('/nav_state', 0)
				# Tilt the camera down
				self.action_pub.publish(self.msg_gen('Cam', [0.0,0.0,0.0,0.0,rospy.get_param('/iisc_landing/cam_tilt_pad_detection'),0.0]))

		return

	def msg_gen(self, str, lis):
		msg = Bebop_cmd()
		msg.header.stamp = rospy.Time.now()
		msg.action = str
		msg.twist = lis
		return msg

	def update_landingpad_detect_hover_count(self):
		val = rospy.get_param('/iisc_landing/landingpad_detect_hover_count')
		rospy.set_param('/iisc_landing/landingpad_detect_hover_count', val-1)

	def update_landing_pad_detection_list(self, res):

		if res:
			self.pad_detected_list = self.update_list(self.pad_detected_list, [1])
		else:
			self.pad_detected_list = self.update_list(self.pad_detected_list, [0])

	def mag2radians(self, ang):
		###
		# if between -1 and -0. take acos as it is 
		# "     "    0.0 and 1.0 take acos + 3.14

		if (ang < 0.0) and (ang >= -1.0):
			res_ang = math.acos(-1.0*ang)*2.0
		else:
			res_ang = math.acos(-1.0*ang)*2.0

			return res_ang

	### Fiducial_callback and it's utility functions

	def fiducial_callback(self, data):

		for marker in data.markers:
			cur_id = marker.id
		
			# Update the 'distance2Lmarker' rosparam
			if cur_id == rospy.get_param('/iisc_landing/landingpad_fiducial_id'):
				rospy.set_param('/iisc_landing/distance2Lmarker', marker.pose.pose.position.z)

			print("^^^^^^ Landing max dist threshold : landing pad distance : landing min dist threshold ", rospy.get_param('/iisc_landing/landing_max_dist_thresh'), " : ", rospy.get_param('/iisc_landing/landingpad_fiducial_id'), " : ", rospy.get_param('/iisc_landing/landing_min_dist_thresh'))

			# If already landingpad fiducial detected, return
			if rospy.get_param('/landing_fiducial_detected'): return

			# Else, update landing fiducial detected list
			if cur_id == rospy.get_param('/iisc_landing/landingpad_fiducial_id'):
				self.landingpad_fiducial_detected_list = self.update_list(self.landingpad_fiducial_detected_list, [1])

				''' If number of positive detections has crossed the threshold, set landing_fiducial_detected to True
					Road following speed will decrease once 'landing_fiducial_detected' is set to True
				'''
				if sum(self.landingpad_fiducial_detected_list)>rospy.get_param('/iisc_landing/landingpad_fiducial_detected_thresh'):
					rospy.set_param('landing_fiducial_detected', True)
					## instead call for iisc_bebop_nav service to switch to slower move ahead speed, to be updated later
			return

	def update_list(self, lst, val):
		lst.pop()
		lst = val + lst
		return lst

	def odom_callback(self, data):
		# Current yaw value
		cur_yaw = data.pose.pose.orientation.z
		self.cur_yaw_rad = self.mag2radians(cur_yaw)

def main(args):
	# Initialize Node class
	rospy.init_node('IISc_landing', anonymous=True)
	l = landing()
	
	# Spin the node
	try:
		rospy.spin()
	# Stop the node if Ctrl-C pressed
	except KeyboardInterrupt:
		print("Shutting down")
		cv2.destroyAllWindows()

if __name__ == '__main__':

	print("Hello world from iisc_landing node!")
	main(sys.argv)
