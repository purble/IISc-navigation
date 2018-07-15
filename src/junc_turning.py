#!/usr/bin/env python
import rospy
import sys, math
import cv2
from sensor_msgs.msg import Image
from iisc_autonav_outdoors.msg import Bebop_cmd
from cv_bridge import CvBridge, CvBridgeError
from ar_track_alvar_msgs.msg import AlvarMarker, AlvarMarkers
from nav_msgs.msg import Odometry
from iisc_autonav_outdoors.msg import UWB_pos
import numpy as np

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

        # UWB data subscriber
        self.uwb_sub = rospy.Subscriber("uwb", UWB_pos, self.uwb_callback)

        # Subscribe to odometry data, for turning using magnetometer
        self.odom_sub = rospy.Subscriber("odom", Odometry, self.odom_callback)

        # Turning Right, Left and Go Straight flags for appropriate renderings on output image
        self.uwbMotion = False
        self.OdomTurn = False

        # Rotation flag for turning at junction
        self.rotate_dir = None
        self.cur_yaw = 0.0

        # Epsilon range round target yaw value to achieve
        # self.eps_yaw = 0.04
        self.eps_yaw = 0.1

        self.anchor_ids = rospy.get_param('/iisc_junc_turning/anchor_ids')
        self.junc_turn_thresh = rospy.get_param('/iisc_junc_turning/junc_turn_thresh')
        self.junc_det_thresh = rospy.get_param('/iisc_junc_turning/junc_det_thresh')
        self.junc_turning_angles = rospy.get_param('/iisc_junc_turning/junc_turning_angles')
        self.junc_turning_dir = rospy.get_param('/iisc_junc_turning/junc_turning_dir')

        # Active junction index, and previous junction index
        self.junc_idx = -1

    def image_callback(self, data):

        # If state is not 2 continue
        if rospy.get_param('/nav_state') != 2:
            return

        # Convert streamed image data to bgr image
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)

        # Render appropriate graphics on output image signalling turning at junctions
        if self.OdomTurn:
            if self.rotate_dir:
                cv2.circle(cv_image, (806, 50), 50, (0, 255, 255), -1)
            else:
                cv2.circle(cv_image, (50, 50), 50, (0, 255, 255), -1)
        elif self.uwbMotion:
            cv2.circle(cv_image, (428, 50), 50, (0, 255, 255), -1)

        # Publish the output image
        try:
            # rospy.loginfo("State 2")
            self.image_pub.publish(self.bridge.cv2_to_imgmsg(cv_image, "bgr8"))
        except CvBridgeError as e:
            print(e)

    def uwb_callback(self, data):

		if self.OdomTurn: return # odom_callback is taking care for time being of the drone's motion

		# Get the anchor id and distance
		a_id, d = anchor_id, dist_from_anchor

		# If the anchor id is in the list of anchors to be encountered along the journey
		if anchor_id in self.anchor_ids:
			
			# Get index of current anchor id in the list
			self.junc_idx = self.anchor_ids.index(anchor_id)

			# If current distance is less than the threshold to turn, start turning
			if (d < self.junc_turn_thresh[self.junc_idx]):
				self.OdomTurn = True
				self.uwbMotion = False
				return
			elif self.uwbMotion: # Already in region between junction detection and turning threshold distances
				self.msg_pub('GoStraight', [0.0]*6)
			elif d < self.junc_det_thresh[self.junc_idx]: # Check if in region between junction detection and turning threshold distances
				rospy.set_param('/nav_state', 2) # Turn to junction navigation state
				self.uwbMotion = True

    def bookeeping(self):

        # Switch back to road following
        rospy.set_param('/nav_state', 1)

        # Turn off Odometer turning flag
        self.OdomTurn = False

    def msg_pub(self, str, lis):
        msg = Bebop_cmd()
        msg.header.stamp = rospy.Time.now()
        msg.action = str
        msg.twist = lis
        self.action_pub.publish(msg)
        return

    def odom_callback(self, data):

        # Current yaw value
        cur_yaw = data.pose.pose.orientation.z

        if not self.OdomTurn: return

        print("Rotating!!!")

        # Target yaw value
        tar_yaw = self.junc_turning_angles[self.junc_idx]

        # Get direction to rotate if currently unknown
        self.rotate_dir = self.junc_turning_dir[self.junc_idx]

        # Get the upper and lower limits for target yaw
        ulim_yaw = tar_yaw + self.eps_yaw
        llim_yaw = tar_yaw - self.eps_yaw

        print("Yaws..", cur_yaw, ulim_yaw, llim_yaw)

        # Check if the current yaw is between the two limits
        if (ulim_yaw <= 1.0) and (llim_yaw >= -1.0) and (cur_yaw > (llim_yaw)) and (cur_yaw < (ulim_yaw)):
            self.OdomTurn = False
        elif (ulim_yaw > 1.0):
            if (cur_yaw < 0.0):
                cur_yaw = 2 - cur_yaw
            if (cur_yaw < ulim_yaw) and (cur_yaw > llim_yaw):
                self.OdomTurn = False
        # (rot < ulim) will anyways hold true, since rot = [-1,1]
        elif (llim_yaw < -1.0):
            if (cur_yaw > 0.0):
                cur_yaw = -2 + cur_yaw
            if (cur_yaw > llim_yaw and cur_yaw < ulim_yaw):
                self.OdomTurn = False

        # If the target yaw is not achieved then rotate
        if self.OdomTurn:
            if self.rotate_dir:
                self.msg_pub('TurnRight', [0.0]*6)
            else:
                self.msg_pub('TurnLeft', [0.0]*6)
        else:  # Do the required bookeeping and switch to road following state
            self.bookeeping()

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


# 3

# def fiducial_callback(self, data):

# 		# If landing or rotating then ignore the fiducial markers
# 		if rospy.get_param('/nav_state') == 0 or self.rotate: return
# 		# Otherwise check if any pad in the self.pad_ids list is detected in current view

# 		# Check if any marker is detected
# 		if  len(data.markers)>0: # I think this is redundant since there will be no callback in the absence of any data
# 			for marker in data.markers:
# 				cur_id = marker.id
# 				print("hello0")
# 				if cur_id in self.pad_ids and cur_id not in self.done_pad_ids:
# 					print("hello1")

# 					if self.robust_detection():

# 						if self.idx is None:
# 							self.idx = self.pad_ids.index(cur_id)

# 						cur_dist = marker.pose.pose.position.z

# 						if cur_dist < self.pad_dist_turn[self.idx]:
# 							self.rotate = True
# 							self.goSt = False
# 							self.done_pad_ids = [cur_id] + self.done_pad_ids
# 							if self.pad_turn_dir[self.idx]:
# 								self.turnR = True
# 							else:
# 								self.turnL = True
# 						elif self.goSt:
# 							print("hello4")
# 							self.msg_pub('GoStraight', [0.0]*6)
# 						elif cur_dist < self.pad_dist_det[self.idx]:
# 							print("hello5")
# 							rospy.set_param('/nav_state', 2)
# 							self.goSt = True

# 					break

# 	def robust_detection(self, reset=False):

# 		if reset:
# 			self.junctionTurning_fiducial_detected_list = [0]*rospy.get_param('/iisc_junc_turning/junctionTurning_fiducial_detected_list_size')
# 		elif rospy.get_param('/nav_state') == 2:
# 			return True
# 		else:
# 			self.junctionTurning_fiducial_detected_list.pop()
# 			self.junctionTurning_fiducial_detected_list = [1] + self.junctionTurning_fiducial_detected_list
# 			if sum(self.junctionTurning_fiducial_detected_list) > self.junctionTurning_fiducial_detection_thresh:
# 				return True
# 			else:
# 				return False