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
        x = tf.nn.conv2d(x,W,strides=[1,strides,strides,1], padding='VALID')
        x = tf.nn.bias_add(x,b)
        x = BatchNorm_layer(x)
    return tf.nn.relu(x)

def maxpool2d(x, k=2):
    with tf.variable_scope('pool', reuse=tf.AUTO_REUSE):
        return tf.nn.max_pool(x, ksize=[1,k,k,1], strides=[1,k,k,1], padding='VALID')

class giusti_rf_net(object):
    def __init__(self):
        self.weight_scale = 0.05
        self.num_filters = [32,32,32,32]
        self.hidden_dims = [200]
        self.keep_prob = tf.Variable(0.5, dtype=tf.float32, trainable=False)
        self.n_classes = 3
        self.init_params()
        self._session = None
        self.X = tf.placeholder(tf.float32, shape=(1,101,101,3))
    
    def init_params(self):
        self.weights = {
            # 5x5 conv, 3 input, 16 outputs
            'wc0': self.weight_scale*tf.Variable(tf.random_normal([4,4,3,self.num_filters[0]]), name='wc0'),
            'wc1': self.weight_scale*tf.Variable(tf.random_normal([4,4,self.num_filters[0],self.num_filters[1]]), name='wc1'),
            'wc2': self.weight_scale*tf.Variable(tf.random_normal([4,4,self.num_filters[1],self.num_filters[2]]), name='wc2'),
            'wc3': self.weight_scale*tf.Variable(tf.random_normal([3,3,self.num_filters[2],self.num_filters[3]]), name='wc3'),
            'wd1': self.weight_scale*tf.Variable(tf.random_normal([self.num_filters[3]*4*4,self.hidden_dims[0]]), name='wd1'),
            'out': self.weight_scale*tf.Variable(tf.random_normal([self.hidden_dims[0],self.n_classes]), name='out')
            }

        self.biases = {
            'bc0': tf.Variable(tf.random_normal([self.num_filters[0]]), name='bc0'),
            'bc1': tf.Variable(tf.random_normal([self.num_filters[1]]), name='bc1'),
            'bc2': tf.Variable(tf.random_normal([self.num_filters[2]]), name='bc2'),
            'bc3': tf.Variable(tf.random_normal([self.num_filters[3]]), name='bc3'),
            'bd1': tf.Variable(tf.random_normal([self.hidden_dims[0]]), name='bd1'),
            'out': tf.Variable(tf.random_normal([self.n_classes]), name='out')
        }

    def inference(self):
        
        inp = self.X
        
        # Convolutional layers
        for i in range(len(self.num_filters)):
            with tf.variable_scope('convL'+str(i+1), reuse=tf.AUTO_REUSE):
                conv = conv2d(inp, self.weights['wc'+str(i)], self.biases['bc'+str(i)], 1)
                inp = maxpool2d(conv, k=2)
                inp = tf.nn.dropout(inp,self.keep_prob)
                
        # Fully connnected layer
        with tf.variable_scope('fc1', reuse=tf.AUTO_REUSE):
            # Reshape input
            fc1 = tf.reshape(inp, [-1, self.weights['wd1'].get_shape().as_list()[0]])
            fc1 = tf.add(tf.matmul(fc1, self.weights['wd1']), self.biases['bd1'])
            fc1 = tf.nn.relu(fc1)
            fc1 = tf.nn.dropout(fc1, self.keep_prob)

        # Output class prediction
        self.logits = tf.add(tf.matmul(fc1, self.weights['out']), self.biases['out'])

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
			print(">> Loading checkpoint state!")
			self._saver.restore(self._session, ckpt.model_checkpoint_path)
		print(">> Loaded!")
    
    def infer(self, inp):
        logits = self._session.run(self.logits, feed_dict={self.X:inp, self.keep_prob:1.0})
        return logits

    def image_conv(self, image):
        image = cv2.resize(image, (101,101)).copy()
        a,b,c = image.shape
        image = image.reshape(1,a,b,c)
        image = image.astype(np.float32) / 255.0
        return image