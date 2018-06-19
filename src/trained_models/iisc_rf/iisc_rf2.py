import tensorflow as tf
import os, cv2, inspect
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import numpy as np
import pickle

class iisc_rf_net(object):
    def __init__(self):
        # Pickle file to load parameters from
        file = "/home/rbccps/Desktop/videos/prafull/BinExp/models/model9_16_16_16_80_binWfpAct_93pct.pkl"
        input = open(file, 'rb')
        self.model = pickle.load(input)
        input.close()

        self.weight_scale = 0.05
        self.num_filters = [16,16,16]
        self.hidden_dims = [80]
        self.keep_prob = tf.Variable(1.0, dtype=tf.float32, trainable=False)
        self.n_classes = 3
        self.init_params()
        self._session = None
        self.X = tf.placeholder(tf.float32, shape=(1,49,49,3))
        # self.m_v = tf.Variable(1.0, dtype=tf.float32, trainable=False)
    
    def BatchNorm_layer(self, x, var, epsilon=1e-5, decay=0.9):
        '''
        Perform a batch normalization after a conv layer or a fc layer
        gamma: a scale factor
        beta: an offset
        epsilon: the variance epsilon - a small float number to avoid dividing by 0
        '''
        with tf.variable_scope('BatchNorm', reuse=tf.AUTO_REUSE) as bnscope:
            # shape = x.get_shape().as_list()
            # gamma = tf.get_variable("gamma", shape[-1])
            # beta = tf.get_variable("beta", shape[-1])
            # moving_avg = tf.get_variable("moving_avg", shape[-1], trainable=False)
            # moving_var = tf.get_variable("moving_var", shape[-1], trainable=False)
            # control_inputs = []
            # avg = moving_avg
            # var = moving_var
            control_inputs = []
            gamma = self.batch_norm[var+'/BatchNorm/gamma']
            beta = self.batch_norm[var+'/BatchNorm/beta']
            avg = self.batch_norm[var+'/BatchNorm/moving_avg']
            var = self.batch_norm[var+'/BatchNorm/moving_var']
            self.m_v = var
            with tf.control_dependencies(control_inputs):
                output = tf.nn.batch_normalization(x, avg, var, offset=beta, scale=gamma, variance_epsilon=epsilon)
        return output

    def conv2d(self, x, W, b, strides, var):
        with tf.variable_scope('conv', reuse=tf.AUTO_REUSE):
            x = tf.nn.conv2d(x,W,strides=[1,strides,strides,1], padding='VALID')
            x = tf.nn.bias_add(x,b)
            x = self.BatchNorm_layer(x, var+'/conv')
        return tf.nn.relu(x)

    def maxpool2d(self, x, k=2):
        with tf.variable_scope('pool', reuse=tf.AUTO_REUSE):
            return tf.nn.max_pool(x, ksize=[1,k,k,1], strides=[1,k,k,1], padding='VALID')

    def init_params(self):
        self.weights = {
            # 5x5 conv, 3 input, 16 outputs
            'wc0': tf.Variable(self.model.params['W1'].transpose(2,3,1,0).astype('float32'), name='wc0'),
            'wc1': tf.Variable(self.model.params['W2'].transpose(2,3,1,0).astype('float32'), name='wc1'),
            'wc2': tf.Variable(self.model.params['W3'].transpose(2,3,1,0).astype('float32'), name='wc2'),
            'wd1': tf.Variable(self.model.params['W4'].astype('float32'), name='wd1'),
            'out': tf.Variable(self.model.params['W5'].astype('float32'), name='out')
            }

        self.biases = {
            'bc0': tf.Variable(self.model.params['b1'].astype('float32'), name='bc0'),
            'bc1': tf.Variable(self.model.params['b2'].astype('float32'), name='bc1'),
            'bc2': tf.Variable(self.model.params['b3'].astype('float32'), name='bc2'),
            'bd1': tf.Variable(self.model.params['b4'].astype('float32'), name='bd1'),
            'bout': tf.Variable(self.model.params['b5'].astype('float32'), name='bout')
        }

        self.batch_norm = {
            'convL1/conv/BatchNorm/gamma': tf.Variable(self.model.params['gamma1'].astype('float32'), name='convL1/conv/BatchNorm/gamma'),
            'convL2/conv/BatchNorm/gamma': tf.Variable(self.model.params['gamma2'].astype('float32').astype('float32'), name='convL2/conv/BatchNorm/gamma'),
            'convL3/conv/BatchNorm/gamma': tf.Variable(self.model.params['gamma3'].astype('float32'), name='convL3/conv/BatchNorm/gamma'),
            'fc1/BatchNorm/gamma': tf.Variable(self.model.params['gamma4'].astype('float32'), name='fc1/BatchNorm/gamma'),
            'convL1/conv/BatchNorm/beta': tf.Variable(self.model.params['beta1'].astype('float32'), name='convL1/conv/BatchNorm/beta'),
            'convL2/conv/BatchNorm/beta': tf.Variable(self.model.params['beta2'].astype('float32'), name='convL2/conv/BatchNorm/beta'),
            'convL3/conv/BatchNorm/beta': tf.Variable(self.model.params['beta3'].astype('float32'), name='convL3/conv/BatchNorm/beta'),
            'fc1/BatchNorm/beta': tf.Variable(self.model.params['beta4'].astype('float32'), name='fc1/BatchNorm/beta'),
            'convL1/conv/BatchNorm/moving_avg': tf.Variable(self.model.bn_params['bn_param1']['running_mean'].astype('float32'), name='convL1/conv/BatchNorm/moving_avg'),
            'convL2/conv/BatchNorm/moving_avg': tf.Variable(self.model.bn_params['bn_param2']['running_mean'].astype('float32'), name='convL2/conv/BatchNorm/moving_avg'),
            'convL3/conv/BatchNorm/moving_avg': tf.Variable(self.model.bn_params['bn_param3']['running_mean'].astype('float32'), name='convL3/conv/BatchNorm/moving_avg'),
            'fc1/BatchNorm/moving_avg': tf.Variable(self.model.bn_params['bn_param4']['running_mean'].astype('float32'), name='fc1/BatchNorm/moving_avg'),
            'convL1/conv/BatchNorm/moving_var': tf.Variable(self.model.bn_params['bn_param1']['running_var'].astype('float32'), name='convL1/conv/BatchNorm/moving_var'),
            'convL2/conv/BatchNorm/moving_var': tf.Variable(self.model.bn_params['bn_param2']['running_var'].astype('float32'), name='convL2/conv/BatchNorm/moving_var'),
            'convL3/conv/BatchNorm/moving_var': tf.Variable(self.model.bn_params['bn_param3']['running_var'].astype('float32'), name='convL3/conv/BatchNorm/moving_var'),
            'fc1/BatchNorm/moving_var': tf.Variable(self.model.bn_params['bn_param4']['running_var'].astype('float32'), name='fc1/BatchNorm/moving_var')
        }

    def inference(self):
        
        inp = self.X
        
        # Convolutional layers
        # for i in range(len(self.num_filters)):
        #     with tf.variable_scope('convL'+str(i+1), reuse=tf.AUTO_REUSE):
        #         conv = self.conv2d(inp, self.weights['wc'+str(i)], self.biases['bc'+str(i)], 1, 'convL'+str(i+1))
        #         inp = self.maxpool2d(conv, k=2)
        #         inp = tf.nn.dropout(inp,self.keep_prob)
                
        with tf.variable_scope('convL1', reuse=tf.AUTO_REUSE):
                conv = self.conv2d(inp, self.weights['wc0'], self.biases['bc0'], 1, 'convL1')
                inp = self.maxpool2d(conv, k=2)
                self.inp_ = conv

        with tf.variable_scope('conv2', reuse=tf.AUTO_REUSE):
                conv = self.conv2d(inp, self.weights['wc1'], self.biases['bc1'], 1, 'convL2')
                inp = self.maxpool2d(conv, k=2)

        with tf.variable_scope('convL3', reuse=tf.AUTO_REUSE):
                conv = self.conv2d(inp, self.weights['wc2'], self.biases['bc2'], 1, 'convL2')
                inp = self.maxpool2d(conv, k=2)

        # Fully connnected layer
        with tf.variable_scope('fc1', reuse=tf.AUTO_REUSE):
            # Reshape input
            fc1 = tf.reshape(inp, [-1, self.weights['wd1'].get_shape().as_list()[0]])
            fc1 = tf.add(tf.matmul(fc1, self.weights['wd1']), self.biases['bd1'])
            # ++++
            fc1 = self.BatchNorm_layer(fc1, 'fc1')
            fc1 = tf.nn.relu(fc1)
            fc1 = tf.nn.dropout(fc1, self.keep_prob)

        # Output class prediction
        self.logits = tf.add(tf.matmul(fc1, self.weights['out']), self.biases['bout'])

    def restoreSession(self):
        # Online Inference
        self._saver = tf.train.Saver()
        self._session = tf.InteractiveSession()
        init_op = tf.global_variables_initializer()
        self._session.run(init_op)
        # dirname = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
        # filename = os.path.join(dirname, 'checkpoints/checkpoint')
        # ckpt = tf.train.get_checkpoint_state(os.path.dirname(filename))
        # if ckpt and ckpt.model_checkpoint_path:
        #     self._saver.restore(self._session, ckpt.model_checkpoint_path)

    
    def infer(self, inp):
        logits, w = self._session.run([self.logits, self.inp_], feed_dict={self.X:inp, self.keep_prob:1.0})
        print(".. ", logits)
        print("********")
        print(w)
        return logits

    def image_conv(self, image):
        image = cv2.resize(image, (49,49)).copy()
        a,b,c = image.shape
        image = image.reshape(1,a,b,c)
        image = image.astype(np.float32) / 255.0
        return image