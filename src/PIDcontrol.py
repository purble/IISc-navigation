import rospy
from iisc_bebop_nav.msg import Bebop_cmd

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