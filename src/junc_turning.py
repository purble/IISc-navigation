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

		# UWB data subscriber
		self.uwb_sub = rospy.Subscriber("uwb", UWBAnchors, self.uwb_callback)

		# Subscribe to odometry data, for turning using magnetometer
		self.odom_sub = rospy.Subscriber("odom", Odometry, self.odom_callback)

		# Turning Right, Left and Go Straight flags for appropriate renderings on output image
		self.uwbMotion = False
		self.OdomTurn = False

		# Rotation flag for turning at junction
		self.rotate = False
		self.rotate_dir = None

		# Epsilon range round target yaw value to achieve
		self.eps_yaw = 0.04

		# Epsilon range around x and y coordinates of destination
		self.eps_xy = 0.1

		# Junctions' calibration
		self.junc_ids = [0, 1]
		self.anchor_ids = [[0,1,2], [3,4,5]] # First anchor is at (0,0), second at (x,0), third at (0,y)
		self.nav_dir_ref = [[[, ], [, ], [, ]], [[, ], [, ]]]
		self.anchor_conf_side = [[4.0,5.0], [6.0, 3.0]] # [x,y] values for each triplet of anchors
		self.junc_dist_thresh = [[1.0, 1.0, 2.0], [1.0, 1.0]] # Start UWB based motion once close enough to first waypoint
		self.junc_waypoints = [[[], [], []], [[], []]]
		self.diff_dirn_motion_angles = [[, , ], [, ]]
		self.junc_turning_angles = [[, , ], [, ]]

		# Active junction index, and previous junction index
		self.junc_idx = 0
		self.prev_junc_idx = None

		# Active waypoint to move next towards, encoded as index of self.junc_waypoint sublist
		self.active_wayp = -1

		# Position in 3D plane
		self.x = None
		self.y = None
		self.z = None

		# Flag to characterize the direction of approach at a junction
		self.ref_idx = None

		# Store the prev junction id, so that no action taken for it until another junction encountered
		self.past_junc_indices = [] #???

	def image_callback(self, data):

		# If state is not 2 continue
		if rospy.get_param('/nav_state') != 2: return

		# Convert streamed image data to bgr image
		try:
			cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
		except CvBridgeError as e:
			print(e)

		# Render appropriate graphics on output image signalling turning at junctions
		if self.OdomTurn:
			if self.turn_dir[self.idx]:
				cv2.circle(cv_image, (806,50), 50, (0,255,255), -1)
			else:
				cv2.circle(cv_image, (50,50), 50, (0,255,255), -1)
		elif self.uwbMotion:
			cv2.circle(cv_image, (428,50), 50, (0,255,255), -1)

		# Publish the output image
		try:
			rospy.loginfo("State 2")
			self.image_pub.publish(self.bridge.cv2_to_imgmsg(cv_image, "bgr8"))
		except CvBridgeError as e:
			print(e)

	def uwb_callback(self, data):

		# Get data from current junction
		[self.x, self.y, self.z] = self.getFilteredUWBdata(data)
		if self.x == None or self.y == None or self.z == None: return

		# Check if drone nearby any reference point
		if self.ref_idx == None:
			self.ref_idx = self.check_if_near_ref_pt()
			if self.ref_idx is not None:
				# So near some reference point, hence change the navigation state
				rospy.set_param('/nav_state', 2)
				# Change self.active_wayp to 0
				self.active_wayp = 0
			else:
				return

		# Execute the trajectory along the set of predefined waypoints
		if not self.rotate:
			self.navigate_along_wayps()

	def bookeeping(self):
		# Increment the junction index
		self.junc_idx += 1

		# Set ref_idx to None
		self.ref_idx = None

		# Switch back to road following
		rospy.set_param('/nav_state', 1)

		# Set rotate_dir to None
		self.rotate_dir = None

	def getFilteredUWBdata(self, data):
		#
		# data format:
		# {'header': .., 'anchors': [{'id': 0, 'dist': 3.5, 'mean': 0.4, 'offset': -0.6, 'var': 0.08}, {'id': 1, ...}, ..]}
		#
		dist = [None, None, None]
		offset = [None, None, None]
		mean = [None, None, None]
		var = [None, None, None]

		# Fill up the distances from three anchors of current active junction denoted by self.junc_idx
		if len(data.anchors)>0:
			for anchor in data.anchors:
				if anchor.id in self.anchor_ids[self.junc_idx]:
					idx = self.anchor_ids.index(anchor.id)
					dist[idx] = anchor.dist
					offset[idx] = anchor.offset
					mean[idx] = anchor.mean
					var[idx] = anchor.var

		# Apply filtering algoirthm to sensor readings
		x, y, z = self.filter_data(dist, offset, mean, var) #??? Yet to code

		return x, y, z

	def check_if_near_ref_pt(self):
		nav_dir_ref = self.nav_dir_ref[self.junc_idx]
		for ix, ref in enumerate(nav_dir_ref):
			if self.within_thresh(ref): #??? Make more robust with repeated measurements
				return ix
		return None

	def within_thresh(self, ref):
		d = np.sqrt((ref[0]-self.x)^2 + (ref[1]-self.y)^2)
		if d <= self.eps_xy:
			return True
		else:
			return False

	def navigate_along_wayps(self):

		# Get the list of waypoints for active junction, based on direction of approach
		wps = self.junc_waypoints[self.junc_idx]
		wayps = wps[self.ref_idx]

		# If self.active_wayp > length of wayps i.e. last waypoint has been reached, set self.rotate to True
		if self.active_wayp==len(wayps):
			self.rotate = True
			return

		# Else get the x,y coordinate of the waypoint to head towards next
		wayp = wayps[self.active_wayp]

		# Check if already reached current waypoint, then increment self.active_wayp
		if self.within_thresh(wayp): #??? Make more robust with repeated measurements
			self.active_wayp += 1
			return
		else: # Else follow the trajectory to reach it
			self.follow_trajectory(wayp) #??? Yet to code

	def follow_trajectory(self, wayp):


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

		# Current yaw value
		cur_yaw = data.pose.pose.orientation.z

		# Target yaw value
		tar_yaw = self.junc_turning_angles[self.junc_idx][self.ref_idx]

		# Get direction to rotate if currently unknown
		if self.rotate_dir is None:
			self.rotate_dir = self.get_rot_dir(cur_yaw, tar_yaw)

		# Get the upper and lower limits for target yaw
		ulim_yaw = tar_yaw + self.eps_yaw
		llim_yaw = tar_yaw - self.eps_yaw

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

		# If the target yaw is not achieved then rotate
		if self.rotate:
			if self.rotate_dir:
				msg = self.msg_pub('TurnRight', [0.0]*6)
			else:
				msg = self.msg_pub('TurnLeft', [0.0]*6)
		else: # Do the required bookeeping and switch to road following state
			self.bookeeping()

	def get_rot_dir(self, cur_yaw, tar_yaw):
		# Convert to a unit vector in 2D coordinates with angle theta w.r.t x axis
		# yaw : theta -- 0.0 : 0, 0.5 : 90, 1.0 : 180, 1.5 : 270, 2.0 : 360,
		
		# Get current and target yaw in radian in new coordinate system
		cyaw = (cur_yaw + 1.0)*math.pi
		tyaw = (tar_yaw + 1.0)*math.pi

		# Get x,y coordinates of unit vector for both current and target yaw
		cx = math.acos(cyaw)
		cy = math.asin(cyaw)
		tx = math.acos(tyaw)
		ty = math.asin(tyaw)

		# Get the cross product between two unit vectors
		c = [cx, cy]
		t = [tx, ty]
		cross_pd = np.cross(c, t)

		# If cross_pd is negative then turn right, else turn left
		if cross_pd <=0.0:
			return True
		else:
			return False

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
