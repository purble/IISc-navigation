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
        self.rotate = False
        self.rotate_dir = None
        self.cur_yaw_rad = 0.0

        # Epsilon range round target yaw value to achieve
        # self.eps_yaw = 0.04
        self.eps_yaw = 0.1

        # Epsilon range around x and y coordinates of destination
        self.eps_xy = 0.5

        # Junctions' calibration
        self.junc_ids = rospy.get_param('/iisc_junc_turning/junc_ids')
        # First anchor is at (0,0), second at (x,0), third at (0,y)
        self.anchor_ids = rospy.get_param('/iisc_junc_turning/anchor_ids')
        self.nav_dir_ref = rospy.get_param('/iisc_junc_turning/nav_dir_ref')
        # [x,y] values for each triplet of anchors
        self.anchor_conf_side = rospy.get_param('/iisc_junc_turning/anchor_conf_side')
        # Start UWB based motion once close enough to first waypoint
        self.junc_dist_thresh = rospy.get_param('/iisc_junc_turning/junc_dist_thresh')
        self.junc_waypoints = rospy.get_param('/iisc_junc_turning/junc_waypoints')
        self.junc_wayp_headings = rospy.get_param('/iisc_junc_turning/junc_wayp_headings')
        self.junc_wayp_n_samples = rospy.get_param('/iisc_junc_turning/junc_wayp_n_samples')
        self.diff_dirn_ref_yaw = rospy.get_param('/iisc_junc_turning/diff_dirn_ref_yaw')
        self.junc_turning_angles = rospy.get_param('/iisc_junc_turning/junc_turning_angles')
        self.junc_calib_angles = rospy.get_param('/iisc_junc_turning/junc_calib_angles')

        # Active junction index, and previous junction index
        self.junc_idx = 0
        self.prev_junc_idx = None

        # Active waypoint to move next towards, encoded as index of self.junc_waypoint sublist
        self.active_wayp = -1

        # Waypoint sample count
        self.n_samples = 0

        # Position in 3D plane
        self.x = None
        self.y = None
        self.z = None

        # Flag to characterize the direction of approach at a junction
        self.ref_idx = None

        # PID constants
        self.yaw_p = rospy.get_param('/iisc_junc_turning/yaw_p')
        self.yaw_i = rospy.get_param('/iisc_junc_turning/yaw_i')
        self.yaw_d = rospy.get_param('/iisc_junc_turning/yaw_d')

        self.xy_p = rospy.get_param('/iisc_junc_turning/xy_p')
        self.xy_i = rospy.get_param('/iisc_junc_turning/xy_i')
        self.xy_d = rospy.get_param('/iisc_junc_turning/xy_d')

        # Pid controllers' integral lists, and delta_caches
        self.yaw_iList = rospy.get_param(
            '/iisc_junc_turning/yaw_i_log_len')*[0.0]
        self.xy_iList = 2 * \
            [rospy.get_param('/iisc_junc_turning/xy_i_log_len')*[0.0]]
        self.yaw_delta_cache = 0.0
        self.xy_delta_cache = [0.0, 0.0]

        # PID actuation thresholds
        self.yaw_pid_thresh = rospy.get_param(
            'iisc_junc_turning/yaw_pid_thresh')
        self.pitch_pid_thresh = rospy.get_param(
            'iisc_junc_turning/pitch_pid_thresh')
        self.roll_pid_thresh = rospy.get_param(
            'iisc_junc_turning/roll_pid_thresh')

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
            if self.turn_dir[self.idx]:
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

		# Get data from current junction
		[self.x, self.y] = data.curr_pos
		if self.x == None or self.y == None :
		    return

		# print("....", self.x, " - ", self.y)

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
		elif (self.rotate is False):
			print("ref_idx :: ", self.ref_idx)
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

        # Reset active_wayp to -1 i.e. look for next reference point
        self.active_wayp = -1

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
        if len(data.anchors) > 0:
            for anchor in data.anchors:
                if anchor.id in self.anchor_ids[self.junc_idx]:
                    idx = self.anchor_ids.index(anchor.id)
                    dist[idx] = anchor.dist
                    offset[idx] = anchor.offset
                    mean[idx] = anchor.mean
                    var[idx] = anchor.var

        # Apply filtering algoirthm to sensor readings
        x, y, z = self.filter_data(dist, offset, mean, var)  # ??? Yet to code

        return x, y, z

    def check_if_near_ref_pt(self):
        nav_dir_ref = self.nav_dir_ref[self.junc_idx]
        for ix, ref in enumerate(nav_dir_ref):
            # ??? Make more robust with repeated measurements
            if self.within_thresh(ref, ix):
                return ix
        return None

    def within_thresh(self, ref, ix):
    	print("ref ", ref)
        d = np.sqrt(pow((ref[0]-self.x), 2) + pow((ref[1]-self.y), 2))
        # print(",,,,:: ", self.active_wayp)
        if self.active_wayp<0:
        	dref = self.junc_dist_thresh[self.junc_idx][ix]
        else:
        	dref = self.eps_xy

        print("####  distances curr_d dref", d, dref)
        if d <= dref:
        	if self.active_wayp<0:
        		return True # No need to collect multiple samples to detect if a junction has arrived
        	else:
        		self.n_samples += 1 # Need to collect junc_wayp_n_samples at each wayp, b4 moving to next
	        	if self.n_samples == self.junc_wayp_n_samples:
		            return True
        return False

    def navigate_along_wayps(self):

		# Get the list of waypoints for active junction, based on direction of approach
		wps = self.junc_waypoints[self.junc_idx]
		wayps = wps[self.ref_idx]

		# If self.active_wayp > length of wayps i.e. last waypoint has been reached, set self.rotate to True
		if self.active_wayp == len(wayps):
		    self.rotate = True
		    return

		# Else get the x,y,z coordinate of the waypoint to head towards next
		wayp = wayps[self.active_wayp]

		# Check if already reached current waypoint, then increment self.active_wayp
		if self.within_thresh(wayp, self.ref_idx):  # ??? Make more robust with repeated measurements
		    self.active_wayp += 1
		    self.n_samples = 0 # Reset the samples collected so far for next waypoint
		    return
		else:  # Else follow the trajectory to reach it
			print(".... wayp", wayp)
			self.follow_trajectory(wayp)

    def follow_trajectory(self, wayp):

        # Get the yaw value from PID controller over the current yaw, and setpoint yaw
        yaw = self.get_yaw_vel_pid()

        # Get y_vel, x_vel and z_vel in UWB coordinate system
        x_vel, y_vel = self.get_xy_vel_pid(wayp)

        # Get pitch and roll values from x_vel and y_vel
        pitch, roll = self.get_pitch_roll(x_vel, y_vel)

        # Send commands to Bebop
        print("<<<<>>>> ", pitch, roll, yaw, x_vel, y_vel)
        if abs(pitch)>0.08:
        	pitch = np.sign(pitch)*0.08
        if abs(roll)>0.06:
        	roll = np.sign(roll)*0.06
        self.msg_pub('Custom', [pitch, roll, 0.0, 0.0, 0.0, yaw])

    def get_yaw_vel_pid(self):

        # Target state
        tar = self.junc_wayp_headings[self.junc_idx][self.ref_idx][self.active_wayp]
        tar = self.mag2radians(tar)

        # Current state
        cur = self.cur_yaw_rad

        # Difference
        _, delta = self.get_rot_dir_n_diff(cur, tar)

        # Calculate the three PID components
        p_comp = self.yaw_p * delta
        d_comp = self.yaw_d * delta-self.yaw_delta_cache
        self.yaw_delta_cache = delta
        self.yaw_iList.pop()
        self.yaw_iList = [delta] + self.yaw_iList
        i_comp = self.yaw_i * sum(self.yaw_iList)

        # Caculate the final actuation value
        val = (p_comp + d_comp + i_comp)*-1.0 # Clockwise is negative

        # Return capped value
        if abs(val) > self.yaw_pid_thresh:
            return self.yaw_pid_thresh * np.sign(val)
        else:
            return val

    def get_xy_vel_pid(self, wayp):

        # Target x, y
        tx, ty = wayp[0], wayp[1]

        # Current x, y
        cx, cy = self.x, self.y

        # Deltas
        delta_x = tx - cx  # go right/left is negative/positive
        delta_y = ty - cy  # go forward/backward is positive/negative

        # Compute the three PID components
        px_comp = self.xy_p * delta_x
        py_comp = self.xy_p * delta_y
        dx_comp = self.xy_d * (delta_x-self.xy_delta_cache[0])
        dy_comp = self.xy_d * (delta_y-self.xy_delta_cache[1])
        self.xy_delta_cache = [delta_x, delta_y]
        self.xy_iList[0].pop()
        self.xy_iList[1].pop()
        self.xy_iList[0] = [delta_x] + self.xy_iList[0]
        self.xy_iList[1] = [delta_y] + self.xy_iList[1]
        ix_comp = self.xy_i * sum(self.xy_iList[0])
        iy_comp = self.xy_i * sum(self.xy_iList[1])

        # Cacluate x_vel, y_vel
        x_vel = px_comp + dx_comp + ix_comp
        y_vel = py_comp + dy_comp + iy_comp

        return x_vel, y_vel

    def mag2radians(self, ang):
    	###
		# if between -1 and -0. take acos as it is 
		# "     "    0.0 and 1.0 take acos + 3.14

    	if (ang < 0.0) and (ang >= -1.0):
        	res_ang = math.acos(-1.0*ang)*2.0
        else:
        	res_ang = math.acos(-1.0*ang)*2.0

       	return res_ang

    def get_pitch_roll(self, x_vel, y_vel):

		# Get theta between x axis of triplet of anchors and curr_yaw
		ang1 = self.diff_dirn_ref_yaw[self.junc_idx][self.ref_idx]
		# ang2 = self.junc_calib_angles[self.junc_idx]
		ang2 = self.cur_yaw_rad # I guess it is better to take actual yaw, since while turning at multiple wayps it may take some time to achieve the setpoint yaw

		# Convert to radians
		ang1 = self.mag2radians(ang1)
		# ang2 = self.mag2radians(ang2) # Already converted to radians        

		print("Angles ref_yaw junc_calib!!!!!", ang1, ang2)

		# Get anti-clockwise theta, from ang1 to ang2
		theta = self.get_anticlockwise_theta(ang1, ang2)
		print("$$$$ theta", theta)

		# Calculate the actuation velocities using this theta
		pitch = x_vel*math.cos(theta) + y_vel*math.sin(theta)
		roll = x_vel*math.sin(theta) - y_vel*math.cos(theta)

		# Return capped velocities
		ret_pitch, ret_roll = None, None

		if abs(pitch) > self.pitch_pid_thresh:
		    ret_pitch = self.pitch_pid_thresh * np.sign(pitch)
		else:
		    ret_pitch = pitch

		if abs(roll) > self.roll_pid_thresh:
		    ret_roll = self.roll_pid_thresh * np.sign(roll)
		else:
		    ret_roll = roll

		return ret_pitch, -1.0*ret_roll  # go right/left is negative/positive

    def get_anticlockwise_theta(self, ang1, ang2):
    	# ang1: Ref angle
    	# ang2: Jun Calib x-axis angle
    	if ang1 >= ang2:
    		return ang1 - ang2
    	else:
    		return 2*math.pi - (ang2-ang1)

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
        self.cur_yaw_rad = self.mag2radians(cur_yaw)

        print("Mag value in radians::::", self.cur_yaw_rad)

        if not self.rotate: return

        print("Rotating!!!")

        # Target yaw value
        tar_yaw = self.junc_turning_angles[self.junc_idx][self.ref_idx]
        tar_yaw_rad = self.mag2radians(tar_yaw)

        # Get direction to rotate if currently unknown
        if self.rotate_dir is None:
            self.rotate_dir, _ = self.get_rot_dir_n_diff(self.cur_yaw_rad, tar_yaw_rad)

        # Get the upper and lower limits for target yaw
        ulim_yaw = tar_yaw + self.eps_yaw
        llim_yaw = tar_yaw - self.eps_yaw

        print("Yaws..", cur_yaw, ulim_yaw, llim_yaw)

        # Check if the current yaw is between the two limits
        if (ulim_yaw <= 1.0) and (llim_yaw >= -1.0) and (cur_yaw > (llim_yaw)) and (cur_yaw < (ulim_yaw)):
            self.rotate = False
        elif (ulim_yaw > 1.0):
            if (cur_yaw < 0.0):
                cur_yaw = 2 - cur_yaw
            if (cur_yaw < ulim_yaw) and (cur_yaw > llim_yaw):
                self.rotate = False
        # (rot < ulim) will anyways hold true, since rot = [-1,1]
        elif (llim_yaw < -1.0):
            if (cur_yaw > 0.0):
                cur_yaw = -2 + cur_yaw
            if (cur_yaw > llim_yaw and cur_yaw < ulim_yaw):
                self.rotate = False

        # If the target yaw is not achieved then rotate
        if self.rotate:
            if self.rotate_dir:
                self.msg_pub('TurnRight', [0.0]*6)
            else:
                self.msg_pub('TurnLeft', [0.0]*6)
        else:  # Do the required bookeeping and switch to road following state
            self.bookeeping()

    def get_rot_dir_n_diff(self, cur_yaw, tar_yaw):
        # Convert to a unit vector in 2D coordinates with angle theta w.r.t x axis
        # yaw : theta -- 0.0 : 0, 0.5 : 90, 1.0 : 180, 1.5 : 270, 2.0 : 360,

        # Get current and target yaw in radian in new coordinate system
        # cyaw = (cur_yaw + 1.0)*math.pi
        # tyaw = (tar_yaw + 1.0)*math.pi
        cyaw = cur_yaw
        tyaw = tar_yaw

        # Get x,y coordinates of unit vector for both current and target yaw
        cx = math.cos(cyaw)
        cy = math.sin(cyaw)
        tx = math.cos(tyaw)
        ty = math.sin(tyaw)

        # Get the cross product between two unit vectors
        c = [cx, cy]
        t = [tx, ty]
        cross_pd = np.cross(c, t)

        # If cross_pd is negative then turn right, else turn left
        if cross_pd <= 0.0:
            return True, self.turnRdiff(cur_yaw, tar_yaw)
        else:
            return False, self.turnLdiff(cur_yaw, tar_yaw)

    def turnRdiff(self, cur_yaw, tar_yaw):
        # If both values are of same sign
        if np.sign(cur_yaw)*np.sign(tar_yaw) == 1.0:
            return tar_yaw - cur_yaw
        elif tar_yaw > 0.0 and cur_yaw < 0.0:
            return (-1.0 - cur_yaw)+(tar_yaw - 1.0)
        elif tar_yaw < 0.0 and cur_yaw > 0.0:
            return (0.0 - cur_yaw)+(tar_yaw)

    def turnLdiff(self, cur_yaw, tar_yaw):
        # If both values are of same sign
        if np.sign(cur_yaw)*np.sign(tar_yaw) == 1.0:
            return tar_yaw - cur_yaw
        elif tar_yaw > 0.0 and cur_yaw < 0.0:
            return (0.0 - cur_yaw)+(tar_yaw)
        elif tar_yaw < 0.0 and cur_yaw > 0.0:
            return (1.0 - cur_yaw)+(tar_yaw + 1.0)


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