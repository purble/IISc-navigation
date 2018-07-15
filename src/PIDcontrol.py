import rospy
from iisc_bebop_nav.msg import Bebop_cmd

# PID control along tilt and yaw
class PIDcontroller2D(object):

	def __init__(self):
		self.err_x_cache = 0.0
		self.i_log_x = [0.0] * rospy.get_param('/iisc_landing/i_log_len')

		# Initialize gains for tilt
		self.kp = rospy.get_param('/iisc_landing/kp')
		self.kd = rospy.get_param('/iisc_landing/kd')
		self.ki = rospy.get_param('/iisc_landing/ki')

		# Initialize gains for yaw
		self.yaw_p = rospy.get_param('/iisc_junc_turning/yaw_p')
		self.yaw_i = rospy.get_param('/iisc_junc_turning/yaw_i')
		self.yaw_d = rospy.get_param('/iisc_junc_turning/yaw_d')

		# Get target landing yaw
		self.landing_yaw = rospy.get_param('iisc_landing/landing_yaw')
		self.landing_yaw_rad = self.mag2radians(self.landing_yaw)

		# Yaw pid controller's integral list, and delta_cache
		self.yaw_iList = rospy.get_param('/iisc_junc_turning/yaw_i_log_len')*[0.0]
		self.yaw_delta_cache = 0.0

		# PID actuation thresholds
		self.yaw_pid_thresh = rospy.get_param('iisc_junc_turning/yaw_pid_thresh')

		# +/- Range within which target is achieved
		self.x_thresh = rospy.get_param('/iisc_landing/x_thresh')

		# Upper limit on |pitvh|
		self.roll_cap = rospy.get_param('/iisc_landing/roll_cap')
		self.fw_vel = rospy.get_param('/iisc_landing/fw_vel')

		# Sign function
		self.sign = lambda x: (1, -1)[x < 0]

	def eval_actuation(self, cur_yaw_rad):
		err_x = rospy.get_param('/iisc_landing/err_x')
		vel_x = 0.0

		# If target not achieved, generate appropriate actuation signals
		if (abs(err_x) > self.x_thresh):
			# Calculate differential error
			d_err_x = err_x - self.err_x_cache
			# Update err_x_cache value and integral error queue
			self.update(err_x)
			# Calculate integral error
			i_err_x = sum(self.i_log_x)
			# Calculate roll
			vel_x = self.kp*err_x + self.kd*d_err_x + self.ki*i_err_x
			# Clip the roll value, to prevent sudden jerks
			vel_x = self.clip(vel_x)

		# Get the yaw value from PID controller over the current yaw, and setpoint yaw
		yaw = self.get_yaw_vel_pid(cur_yaw_rad)

		# Generate msg
		msg = Bebop_cmd()
		msg.header.stamp = rospy.Time.now()
		msg.action = 'Custom'
		msg.twist = [self.fw_vel, vel_x, 0.0, 0.0, 0.0, yaw]

		return msg

	def get_yaw_vel_pid(self, cur_yaw_rad):

		# Target state
		tar = self.landing_yaw_rad

		# Current state
		cur = cur_yaw_rad

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

	def clip(self, z):
		if abs(z)<self.roll_cap: return z
		else: return self.sign(z)*self.roll_cap

	def mag2radians(self, ang):
		###
		# if between -1 and -0. take acos as it is 
		# "     "    0.0 and 1.0 take acos + 3.14

		if (ang < 0.0) and (ang >= -1.0):
			res_ang = math.acos(-1.0*ang)*2.0
		else:
			res_ang = math.acos(-1.0*ang)*2.0

			return res_ang

	def update(self, err_x):
		self.err_x_cache = err_x
		self.i_log_x.pop()
		self.i_log_x = [err_x] + self.i_log_x

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


# PID control only along tilt
class PIDcontroller1D(object):

	def __init__(self):
		self.err_x_cache = 0.0
		self.i_log_x = [0.0] * rospy.get_param('/iisc_landing/i_log_len')

		# Initialize gains
		self.kp = rospy.get_param('/iisc_landing/kp')
		self.kd = rospy.get_param('/iisc_landing/kd')
		self.ki = rospy.get_param('/iisc_landing/ki')

		# +/- Range within which target is achieved
		self.x_thresh = rospy.get_param('/iisc_landing/x_thresh')

		# Upper limit on |pitvh|
		self.roll_cap = rospy.get_param('/iisc_landing/roll_cap')
		self.fw_vel = rospy.get_param('/iisc_landing/fw_vel')

		# Sign function
		self.sign = lambda x: (1, -1)[x < 0]

	def eval_actuation(self):
		err_x = rospy.get_param('/iisc_landing/err_x')
		vel_x = 0.0

		# If target not achieved, generate appropriate actuation signals
		if (abs(err_x) > self.x_thresh):
			# Calculate differential error
			d_err_x = err_x - self.err_x_cache
			# Update err_x_cache value and integral error queue
			self.update(err_x)
			# Calculate integral error
			i_err_x = sum(self.i_log_x)
			# Calculate roll
			vel_x = self.kp*err_x + self.kd*d_err_x + self.ki*i_err_x
			# Clip the roll value, to prevent sudden jerks
			vel_x = self.clip(vel_x)

		# Generate msg
		msg = Bebop_cmd()
		msg.header.stamp = rospy.Time.now()
		msg.action = 'Custom'
		msg.twist = [self.fw_vel, vel_x, 0.0, 0.0, 0.0, 0.0]

		return msg

	def clip(self, z):
		if abs(z)<self.roll_cap: return z
		else: return self.sign(z)*self.roll_cap

	def update(self, err_x):
		self.err_x_cache = err_x
		self.i_log_x.pop()
		self.i_log_x = [err_x] + self.i_log_x