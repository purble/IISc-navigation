import tensorflow as tf
import os, cv2, inspect
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import numpy as np

def BatchNorm_layer(x, epsilon=0.001, decay=.99):
    '''
    Perform a batch normalization after a conv layer or a fc layer
    gamma: a scale factor
    beta: an offset
    epsilon: the variance epsilon - a small float number to avoid dividing by 0
    '''
    with tf.variable_scope('BatchNorm', reuse=tf.AUTO_REUSE) as bnscope:
        shape = x.get_shape().as_list()
        gamma = tf.get_variable("gamma", shape[-1], initializer=tf.constant_initializer(1.0))
        beta = tf.get_variable("beta", shape[-1], initializer=tf.constant_initializer(0.0))
        moving_avg = tf.get_variable("moving_avg", shape[-1], initializer=tf.constant_initializer(0.0), trainable=False)
        moving_var = tf.get_variable("moving_var", shape[-1], initializer=tf.constant_initializer(1.0), trainable=False)
        control_inputs = []
        avg = moving_avg
        var = moving_var
        with tf.control_dependencies(control_inputs):
            output = tf.nn.batch_normalization(x, avg, var, offset=beta, scale=gamma, variance_epsilon=epsilon)
    return output

def conv2d(x, W, b, strides):
    with tf.variable_scope('conv', reuse=tf.AUTO_REUSE):
        x = tf.nn.conv2d(x,W,strides=[1,strides,strides,1], padding='SAME')
        x = tf.nn.bias_add(x,b)
    return tf.nn.relu(x)

def maxpool2d(x, k=2):
    with tf.variable_scope('pool', reuse=tf.AUTO_REUSE):
        return tf.nn.max_pool(x, ksize=[1,k,k,1], strides=[1,k,k,1], padding='VALID')
    
def resBlock(x1, w_lis, b_lis, strides=2):
    with tf.variable_scope('conv1', reuse=tf.AUTO_REUSE):
        x2 = BatchNorm_layer(x1)
        x2 = tf.nn.relu(x2)
        x2 = conv2d(x2, w_lis[0], b_lis[0], strides)
        
    with tf.variable_scope('conv2', reuse=tf.AUTO_REUSE):
        x2 = BatchNorm_layer(x2)
        x2 = tf.nn.relu(x2)
        x2 = conv2d(x2, w_lis[1], b_lis[1], 1)

    with tf.variable_scope('conv3', reuse=tf.AUTO_REUSE):
        x1 = conv2d(x1, w_lis[2], b_lis[2], strides)
        
    with tf.variable_scope('output', reuse=tf.AUTO_REUSE):
        x3 = tf.add(x1,x2)
    
    return x3

class dronet_rf_net(object):
    def __init__(self):
        self.weight_scale = 0.05
        self.num_filters_convL = [32]
        self.num_filters_resBl = [32,64,128]
        self.keep_prob = tf.Variable(0.5, dtype=tf.float32, trainable=False)
        self.n_classes = 3
        self.init_params()
        self._session = None
        self.X = tf.placeholder(tf.float32, shape=(1,200,200,1))
    
    def init_params(self):
        self.weights = {
            # 5x5 conv, 3 input, 16 outputs
            'wc0': self.weight_scale*tf.Variable(tf.random_normal([5,5,1,self.num_filters_convL[0]]), name='wc0'),
            'wc1r1': self.weight_scale*tf.Variable(tf.random_normal([3,3,self.num_filters_convL[0],self.num_filters_resBl[0]]), name='wc1r1'),
            'wc2r1': self.weight_scale*tf.Variable(tf.random_normal([3,3,self.num_filters_resBl[0],self.num_filters_resBl[0]]), name='wc2r1'),
            'wc3r1': self.weight_scale*tf.Variable(tf.random_normal([1,1,self.num_filters_convL[0],self.num_filters_resBl[0]]), name='wc3r1'),
            'wc1r2': self.weight_scale*tf.Variable(tf.random_normal([3,3,self.num_filters_resBl[0],self.num_filters_resBl[1]]), name='wc1r2'),
            'wc2r2': self.weight_scale*tf.Variable(tf.random_normal([3,3,self.num_filters_resBl[1],self.num_filters_resBl[1]]), name='wc2r2'),
            'wc3r2': self.weight_scale*tf.Variable(tf.random_normal([1,1,self.num_filters_resBl[0],self.num_filters_resBl[1]]), name='wc3r2'),
            'wc1r3': self.weight_scale*tf.Variable(tf.random_normal([3,3,self.num_filters_resBl[1],self.num_filters_resBl[2]]), name='wc1r3'),
            'wc2r3': self.weight_scale*tf.Variable(tf.random_normal([3,3,self.num_filters_resBl[2],self.num_filters_resBl[2]]), name='wc2r3'),
            'wc3r3': self.weight_scale*tf.Variable(tf.random_normal([1,1,self.num_filters_resBl[1],self.num_filters_resBl[2]]), name='wc3r3'),
            'wd1': self.weight_scale*tf.Variable(tf.random_normal([self.num_filters_resBl[2]*7*7,self.n_classes]), name='wd1')
        }

        self.biases = {
            'bc0': tf.Variable(tf.random_normal([self.num_filters_convL[0]]), name='bc0'),
            'bc1r1': tf.Variable(tf.random_normal([self.num_filters_resBl[0]]), name='bc1r1'),
            'bc2r1': tf.Variable(tf.random_normal([self.num_filters_resBl[0]]), name='bc2r1'),
            'bc3r1': tf.Variable(tf.random_normal([self.num_filters_resBl[0]]), name='bc3r1'),
            'bc1r2': tf.Variable(tf.random_normal([self.num_filters_resBl[1]]), name='bc1r2'),
            'bc2r2': tf.Variable(tf.random_normal([self.num_filters_resBl[1]]), name='bc2r2'),
            'bc3r2': tf.Variable(tf.random_normal([self.num_filters_resBl[1]]), name='bc3r2'),
            'bc1r3': tf.Variable(tf.random_normal([self.num_filters_resBl[2]]), name='bc1r3'),
            'bc2r3': tf.Variable(tf.random_normal([self.num_filters_resBl[2]]), name='bc2r3'),
            'bc3r3': tf.Variable(tf.random_normal([self.num_filters_resBl[2]]), name='bc3r3'),
            'bd1': tf.Variable(tf.random_normal([self.n_classes]), name='bd1')
        }

    def inference(self):
        
        inp = self.X
        
        # Convolutional layers
        for i in range(len(self.num_filters_convL)):
            with tf.variable_scope('convL'+str(i+1), reuse=tf.AUTO_REUSE):
                conv = conv2d(inp, self.weights['wc'+str(i)], self.biases['bc'+str(i)], 2)
                inp = maxpool2d(conv, k=2)
            
        # Residual blocks
        for i in range(len(self.num_filters_resBl)):
            with tf.variable_scope('resBl'+str(i+1), reuse=tf.AUTO_REUSE):
                w_lis = []
                b_lis = []
                for j in range(3):
                    w_lis = w_lis + [self.weights['wc'+str(j+1)+'r'+str(i+1)]]
                    b_lis = b_lis + [self.biases['bc'+str(j+1)+'r'+str(i+1)]]
                inp = resBlock(inp, w_lis, b_lis)
                
        # Fully connnected layer
        with tf.variable_scope('fc1', reuse=tf.AUTO_REUSE):
	        # Reshape input
            fc1 = tf.layers.flatten(inp)
            fc1 = tf.nn.relu(fc1)
            fc1 = tf.nn.dropout(fc1, self.keep_prob)

        # Output class prediction
        self.logits = tf.add(tf.matmul(fc1, self.weights['wd1']), self.biases['bd1'])

    def restoreSession(self):
		# Online Inference
		self._saver = tf.train.Saver()
		self._session = tf.InteractiveSession()
		init_op = tf.global_variables_initializer()
		self._session.run(init_op)
		dirname = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
		filename = os.path.join(dirname, 'checkpoints/checkpoint')
		ckpt = tf.train.get_checkpoint_state(os.path.dirname(filename))
		if ckpt and ckpt.model_checkpoint_path:
			self._saver.restore(self._session, ckpt.model_checkpoint_path)
    
    def infer(self, inp):
        logits = self._session.run(self.logits, feed_dict={self.X:inp, self.keep_prob:1.0})
        return logits

    def image_conv(self, image):
        image = cv2.resize(image, (200,200)).copy()
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # print(">>")
        # cv2.imshow('img', image)
        # cv2.waitKey(0)
        image = image.reshape(1,200,200,1)
        image = image.astype(np.float32) / 255.0
        return image