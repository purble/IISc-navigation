#!/usr/bin/env python
import rospy, sys
from sensor_msgs.msg import Image
from iisc_bebop_nav.msg import Bebop_cmd
from cv_bridge import CvBridge, CvBridgeError
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
		# Initialize pad detection history list
		self.pad_detected_list = [0]*rospy.get_param('/iisc_landing/landingpad_fiducial_detected_list_size')

		# Fiducial aided landing enabled
		if rospy.get_param('/iisc_landing/landingWithFiducial'):
			# Fiducial marker info subscriber
			self.fiducial_sub = rospy.Subscriber("fiducial_marker", AlvarMarkers, self.fiducial_callback)
			# Initialize landing pad marker detection history list
			self.landingpad_fiducial_detected_list = [0]*rospy.get_param('/iisc_landing/landingpad_fiducial_detected_list_size')

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

				cmd = self.pid_controller.eval_actuation()
				self.action_pub.publish(cmd)

			if sum(self.pad_detected_list) < rospy.get_param('/iisc_landing/landingpad_notinFOV_thresh'):
				self.action_pub.publish(self.msg_gen('Land', [0.0] * 6))
				self.action_pub.publish(self.msg_gen('Cam', [0.0] * 6))
				rospy.set_param('/iisc_landing/landed', True)

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

	### Fiducial_callback and it's utility functions

	def fiducial_callback(self, data):
		
		# If already landingpad fiducial detected, return
		if rospy.get_param('/landing_fiducial_detected'): return

		# Else, update landing fiducial detected list
		if (len(data.markers)>0) and (data.markers[0].id == rospy.get_param('/iisc_landing/landingpad_fiducial_id')):
			self.landingpad_fiducial_detected_list = self.update_list(self.landingpad_fiducial_detected_list, [1])
		else:
			self.landingpad_fiducial_detected_list = self.update_list(self.landingpad_fiducial_detected_list, [0])

		''' If number of positive detections has crossed the threshold, set landing_fiducial_detected to True
			Road following speed will decrease once 'landing_fiducial_detected' is set to True
		'''
		if sum(self.landingpad_fiducial_detected_list)>rospy.get_param('/iisc_landing/landingpad_fiducial_detected_thresh'):
			rospy.set_param('landing_fiducial_detected', True)
			## instead call for iisc_bebop_nav service to switch to slower move ahead speed
		return

	def update_list(self, lst, val):
		lst.pop()
		lst = val + lst
		return lst

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
