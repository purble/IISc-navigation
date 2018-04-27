import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

def get_model(model):
	if model == 'iisc_rf':
		from trained_models.iisc_rf.iisc_rf import *
		m = iisc_rf_net()
	elif model == 'giusti_rf':
		from trained_models.giusti_rf.giusti_rf import *
		m = giusti_rf_net()
	elif model == 'dronet_rf':
		from trained_models.dronet_rf.dronet_rf import *
		m = dronet_rf_net()
	else:
		rospy.logerr("'model_type' returned the invalid value %s", model)
	return m

if __name__ == '__main__':
	main()