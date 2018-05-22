#!/usr/bin/env python
import rospy, sys, cv2, time
from sensor_msgs.msg import Image
from iisc_rf.msg import CNN_logits
from iisc_bebop_nav.msg import Bebop_cmd
from cv_bridge import CvBridge, CvBridgeError
from models import *
import numpy as np

class iisc_rf(object):

	def __init__(self):

		self.model = get_model(rospy.get_param('/iisc_rf/model_type'))
		self.model.inference()
		self.model.restoreSession()

		# Convert ROS image messages to OpenCV images
		self.bridge = CvBridge()

		# Image subscriber
		self.image_sub = rospy.Subscriber("camera", Image, self.image_callback)

		# Pulisher for output image with renderings etc.
		self.image_pub = rospy.Publisher("output_image", Image, queue_size=10)

		# Publisher for actions to be executed by Bebop
		self.action_pub = rospy.Publisher("actions", Bebop_cmd, queue_size=10)

		# Publisher for logits i.e. output of forward pass throught the DNN model
		self.logits_pub = rospy.Publisher("logits", CNN_logits, queue_size=10)

		# Set the various parameters
		self.lr_bias = rospy.get_param('/iisc_rf/lr_bias')
		self.gost_thresh = rospy.get_param('/iisc_rf/gost_thresh')
		self.confusion_correction = rospy.get_param('/iisc_rf/confusion_correction')
		self.dirn_history = [1] * rospy.get_param('/iisc_rf/dirn_history_sz')
		self.dirn_history_thresh = rospy.get_param('/iisc_rf/dirn_history_thresh')
		self.bias_lr_thresh = rospy.get_param('/iisc_rf/bias_lr_thresh')
		self.confusion_corr_thresh1 = rospy.get_param('/iisc_rf/confusion_corr_thresh1')
		self.confusion_corr_thresh2 = rospy.get_param('/iisc_rf/confusion_corr_thresh2')
		self.prev_dirn = 0
		self.inf_rate_q = [0.0]*50

	def image_callback(self, data):

		# If turning at junction, or landing do not follow the road
		if rospy.get_param('/nav_state')!=1: return

		# Convert streamed image data to bgr image
		try:
			cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
		except CvBridgeError as e:
			print(e)

		# Convert the BGR image to np.array in appropriate shape for feeding into the model
		img = self.model.image_conv(cv_image)

		t1 = time.time()
		# Run inference on the model
		logits = self.model.infer(img)
		self.inf_rate_q.pop()
		self.inf_rate_q = [time.time()-t1] + self.inf_rate_q
		print(">>> ", sum(self.inf_rate_q)/50)

		# Publish the logits
		msg = CNN_logits()
		msg.header.stamp = rospy.Time.now()
		msg.logits = logits.tolist()
		self.logits_pub.publish(msg)

		# Find the index with largest value
		output = np.argmax(logits, axis=1)[0]

		# Calculate respective probabilities from logits
		probs = (np.exp(logits[0]) / np.sum(np.exp(logits[0]), axis=0)) * 130

		# Run through heuristic for biasing the decision more for TurnLeft, TurnRight in case of confusion
		# Helps in turning before reaching the edge of the road
		if self.lr_bias:
			output = self.bias_lr(probs, output)

		# Account for oscillations in case of confusion, by making the drone move straight slowly
		if self.confusion_correction and output!=1:
			output = self.confusion_corr(output, probs)

		# Render the graphics on image, and send command to Bebop
		out_image = self.render_graphics_and_send_cmd(cv_image, output, probs)

		# Publish the output image
		try:
			rospy.loginfo("State 1")
			self.image_pub.publish(self.bridge.cv2_to_imgmsg(out_image, "bgr8"))
		except CvBridgeError as e:
			print(e)

	def bias_lr(self, probs, output):

		if probs[1] > self.gost_thresh:
			if (abs(probs[0]-probs[1]) < self.bias_lr_thresh): output = 0
			elif (abs(probs[2]-probs[1]) < self.bias_lr_thresh): output = 2
		return output

	def confusion_corr(self, output, probs):
		# make output +1/-1
		out = output - 1
		# Update the direction consistency history accordingly
		self.dirn_history.pop()
		# if there is a change in direction
		if out * self.prev_dirn == -1:
			self.dirn_history = [-1] + self.dirn_history
		elif probs[0]> self.confusion_corr_thresh1 and probs[2]> self.confusion_corr_thresh1 and abs(probs[0]-probs[2]) < self.confusion_corr_thresh2:
			self.dirn_history = [-1] + self.dirn_history
		else:
			self.dirn_history = [1] + self.dirn_history
		self.prev_dirn = out
		# Go slow, if confusion i.e. sum of dirn_history goes below the threshold
		if sum(self.dirn_history) < self.dirn_history_thresh:
			output = 3
		return output

	def render_graphics_and_send_cmd(self, image, output, probs):

		msg = Bebop_cmd()
		msg.header.stamp = rospy.Time.now()
		msg.twist = [0.0]*6

		# cv_image : 480x856x3
		if output==1:
			# straight
			cv2.circle(image, (428,50), 50, (0,255,0), -1)
			# Once go slow service by iisc_bebop_nav is created, this will be simplified to just below line
			# msg.action = "GoStraight"

			# Till then
			# If fiducial marker for landing pad is detected, slow the speed of road following
			if rospy.get_param('/landing_fiducial_detected'):
				msg.action = "GoStraightSlow"
			else:
				msg.action = "GoStraight"
		elif output==0:
			# left
			cv2.circle(image, (50,50), 50, (0,255,0), -1)
			msg.action = "TurnLeft"
		elif output==2:
			# right
			cv2.circle(image, (806,50), 50, (0,255,0), -1)
			msg.action = "TurnRight"
		elif output==3:
			# straight at slow speed
			cv2.circle(image, (428,50), 25, (0,255,0), -1)
			msg.action = "GoStraightSlow"
		else:
			cv2.circle(image, (50,50), 50, (0,0,255), -1)

		# Draw lines corresponding to probabilities on image
		cv2.line(image, (400,480), (400,480-int(probs[0])), (255,0,0), 12)
		cv2.line(image, (430,480), (430,480-int(probs[1])), (0,255,0), 12)
		cv2.line(image, (460,480), (460,480-int(probs[2])), (0,0,255), 12)

		self.action_pub.publish(msg)

		return image

def main(args):
	# Initialize Node class
	rospy.init_node('IISc_rf', anonymous=True)
	i_r = iisc_rf()
	
	# Spin the node
	try:
		rospy.spin()
	# Stop the node if Ctrl-C pressed
	except KeyboardInterrupt:
		print("Shutting down")
		cv2.destroyAllWindows()

if __name__ == '__main__':

	print("Hello world from iisc_rf node!")
	main(sys.argv)