import rospy
import cv2, imutils
import numpy as np

class ShapeDetector(object):

	def __init__(self):
		self.detected = False
		self.lower_red1 = np.asarray(rospy.get_param('/iisc_landing/lower_red1'))
		self.lower_red2 = np.asarray(rospy.get_param('/iisc_landing/lower_red2'))
		self.upper_red1 = np.asarray(rospy.get_param('/iisc_landing/upper_red1'))
		self.upper_red2 = np.asarray(rospy.get_param('/iisc_landing/upper_red2'))
		self.lower_green = np.asarray(rospy.get_param('/iisc_landing/lower_green'))
		self.upper_green = np.asarray(rospy.get_param('/iisc_landing/upper_green'))
		self.lower_blue = np.asarray(rospy.get_param('/iisc_landing/lower_blue'))
		self.upper_blue = np.asarray(rospy.get_param('/iisc_landing/upper_blue'))
		self.x_err_log = [0.0]*rospy.get_param('/iisc_landing/i_log_len')

	def process(self, image):

		self.detected = False

		# convert bgr 2 hsv
		hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

		# Prepare red, green, blue masks
		mask_r = cv2.inRange(hsv, self.lower_red1, self.upper_red1) + cv2.inRange(hsv, self.lower_red2, self.upper_red2)
		mask_g = cv2.inRange(hsv, self.lower_green, self.upper_green)
		mask_b = cv2.inRange(hsv, self.lower_blue, self.upper_blue)

		# Get landing pad specifications
		bck_color = rospy.get_param('/iisc_landing/landingpad_background_color')
		centre_color = rospy.get_param('/iisc_landing/landingpad_centre_color')

		# If already landed, then just return the masks with renderings
		if rospy.get_param('/iisc_landing/landed'):
			cv2.circle(image, (428,50), 50, (0,255,255), -1)
			return self.detected, image, mask_r, mask_g, mask_b

		# Prepare backgroud color mask
		mask_bck = None
		if bck_color == "red":
			mask_bck = mask_r
		elif bck_color == "green":
			mask_bck = mask_g
		elif bck_color == "blue":
			mask_bck = mask_b
		else:
			rospy.logerr("'landingpad_background_color' returned the invalid value %s", bck_color)

		# Prepare centre color mask
		mask_centre = None
		if centre_color == "red":
			mask_centre = mask_r
		elif centre_color == "green":
			mask_centre = mask_g
		elif centre_color == "blue":
			mask_centre = mask_b
		else:
			rospy.logerr("'landingpad_background_color' returned the invalid value %s", centre_color)

		# Find contours in image
		cnts = cv2.findContours(mask_bck.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
		cnts = cnts[0] if imutils.is_cv2() else cnts[1]

		# Iterate over contours:
		for c in cnts:

			# compute perimeter and area of detected contour
			peri = cv2.arcLength(c, True)
			area = cv2.contourArea(c)

			# approximate detected contour as a polygon
			approx = cv2.approxPolyDP(c, 0.1 * peri, True)

			# If contour has 3/4/5 sides and satisfies the area & perimeter requirements
			if (len(approx) == 3 or len(approx) == 4 or len(approx) == 5) and area > 1000.0 and peri > 50.0 and peri < 1500.0:
				# Check if the color at the centre matches
				image = self.process_Contour(image, c, mask_centre)
				if self.detected: break

		return self.detected, image, mask_r, mask_g, mask_b

	def process_Contour(self, image, c, mask):
		
		# Get dimensions of image
		ht, wd, ch = image.shape

		# Get the center of contour
		M = cv2.moments(c)
		
		# Get the coordinates of the centre of the contour
		cX = int(M["m10"] / M["m00"])
		cY = int(M["m01"] / M["m00"])

		# Draw the contour, and a circle at the centre of image
		cv2.circle(image, (cX, cY), 3, (0,255,0), -1)
		cv2.drawContours(image, [c], -1, (0,255,0), 2)

		# If the pixel at the centre of contour in mask is on, pad is detected, else return the image
		if mask[cY,cX]==255: self.detected = True
		else: return image

		# Otherwise draw line joining centre of image to centre of contour
		cv2.line(image, (wd/2, ht/2), (cX, cY), (136,0,108), 2)

		# Compute errors in x and y directions for PID loop
		errx = (wd/2)-cX
		erry = (ht/2)-cY

		# Update err logs
		self.x_err_log.pop()
		self.x_err_log = [abs(errx)] + self.x_err_log

		# Update parameter values
		rospy.set_param('/iisc_landing/err_x', errx)

		# Ideally, call the service of PID node to update err_logs

		return image