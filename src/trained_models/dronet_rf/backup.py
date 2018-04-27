import tensorflow as tf
import os, cv2, time
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
from klepto.archives import *
from tensorflow.python.training import moving_averages

def BatchNorm_layer(x, train, epsilon=0.001, decay=.99):
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
        if train:
            avg, var = tf.nn.moments(x, range(len(shape)-1))
            update_moving_avg = moving_averages.assign_moving_average(moving_avg, avg, decay)
            update_moving_var = moving_averages.assign_moving_average(moving_var, var, decay)
            control_inputs = [update_moving_avg, update_moving_var]
        else:
            avg = moving_avg
            var = moving_var
        with tf.control_dependencies(control_inputs):
            output = tf.nn.batch_normalization(x, avg, var, offset=beta, scale=gamma, variance_epsilon=epsilon)
    return output

def conv2d(x, W, b, training, strides):
    with tf.variable_scope('conv', reuse=tf.AUTO_REUSE):
        x = tf.nn.conv2d(x,W,strides=[1,strides,strides,1], padding='SAME')
        x = tf.nn.bias_add(x,b)
    return tf.nn.relu(x)

def maxpool2d(x, k=2):
    with tf.variable_scope('pool', reuse=tf.AUTO_REUSE):
        return tf.nn.max_pool(x, ksize=[1,k,k,1], strides=[1,k,k,1], padding='VALID')
    
def resBlock(x1, w_lis, b_lis, training, strides=2):
    with tf.variable_scope('conv1', reuse=tf.AUTO_REUSE):
        x2 = BatchNorm_layer(x1, training)
        x2 = tf.nn.relu(x2)
        x2 = conv2d(x2, w_lis[0], b_lis[0], training, strides)
        
    with tf.variable_scope('conv2', reuse=tf.AUTO_REUSE):
        x2 = BatchNorm_layer(x2, training)
        x2 = tf.nn.relu(x2)
        x2 = conv2d(x2, w_lis[1], b_lis[1], training, 1)

    with tf.variable_scope('conv3', reuse=tf.AUTO_REUSE):
        x1 = conv2d(x1, w_lis[2], b_lis[2], training, strides)
        
    with tf.variable_scope('output', reuse=tf.AUTO_REUSE):
        x3 = tf.add(x1,x2)
    
    return x3
        
# def safe_mkdir(path):
#     """ Create a directory if there isn't one already. """
#     try:
#         os.mkdir(path)
#     except OSError:
#         pass

class iisc_rf_net(object):
    def __init__(self):
        self.weight_scale = 0.05
        self.num_filters_convL = [32]
        self.num_filters_resBl = [32,64,128]
        self.hidden_dims = [200]
        # self.lr = 1e-2
        # self.batch_size = 128
        self.keep_prob = tf.Variable(0.5, dtype=tf.float32, trainable=False)
        # self.num_epochs = 50
        # self.gstep = tf.Variable(0, dtype=tf.int32, 
                                # trainable=False, name='global_step')
        self.n_classes = 3
        # self.skip_step = 20
        # self.n_test = 2500
        self.training = False
        # self.training_filenames = ["data/train_batch_200x200g_1.tfrecords","data/train_batch_200x200g_2.tfrecords","data/train_batch_200x200g_3.tfrecords",\
                              # "data/train_batch_200x200g_4.tfrecords","data/train_batch_200x200g_5.tfrecords"]
        # self.testing_filenames = ["data/valid_batch_200x200g.tfrecords"]
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

    # def _parse_function(self, data):
    #     features = {"image": tf.FixedLenFeature((), tf.string, default_value=""),\
    #               "label": tf.FixedLenFeature((), tf.int64, default_value=0)}
    #     parsed_features = tf.parse_single_example(data, features)
    #     return tf.decode_raw(parsed_features["image"], tf.uint8), parsed_features["label"]

    # def get_data(self):
    #     with tf.name_scope('data'):
            
    #         self.filenames = tf.placeholder(tf.string, shape=[None])
    #         dataset = tf.data.TFRecordDataset(self.filenames)
    #         dataset = dataset.map(self._parse_function)  # Parse the record into tensors.
    #         dataset = dataset.batch(self.batch_size)
    #         self.iterator = dataset.make_initializable_iterator()
            
    #         self.X, Y = self.iterator.get_next()
    #         self.X = tf.reshape(self.X, shape=[-1, 200, 200, 1])
    #         self.X = tf.divide(tf.cast(self.X, tf.float32), 255.0)
    #         self.label = tf.one_hot(Y,3)
            
    def inference(self):
        
        inp = self.X
        
        # Convolutional layers
        for i in range(len(self.num_filters_convL)):
            with tf.variable_scope('convL'+str(i+1), reuse=tf.AUTO_REUSE):
                conv = conv2d(inp, self.weights['wc'+str(i)], self.biases['bc'+str(i)], self.training, 2)
                inp = maxpool2d(conv, k=2)
            
        # Residual blocks
        for i in range(len(self.num_filters_resBl)):
            with tf.variable_scope('resBl'+str(i+1), reuse=tf.AUTO_REUSE):
                w_lis = []
                b_lis = []
                for j in range(3):
                    w_lis = w_lis + [self.weights['wc'+str(j+1)+'r'+str(i+1)]]
                    b_lis = b_lis + [self.biases['bc'+str(j+1)+'r'+str(i+1)]]
                inp = resBlock(inp, w_lis, b_lis, self.training)
                
        # Fully connnected layer
        with tf.variable_scope('fc1', reuse=tf.AUTO_REUSE):
	        # Reshape input
            fc1 = tf.layers.flatten(inp)
            fc1 = tf.nn.relu(fc1)
            fc1 = tf.nn.dropout(fc1, self.keep_prob)

        # Output class prediction
        self.logits = tf.add(tf.matmul(fc1, self.weights['wd1']), self.biases['bd1'])

        self.preds = tf.nn.softmax(self.logits)
        
    # def loss(self):
    #     '''
    #     define loss function
    #     use softmax cross entropy with logits as the loss function
    #     compute mean cross entropy, softmax is applied internally
    #     '''
    #     # 
    #     with tf.name_scope('loss'):
    #         entropy = tf.nn.softmax_cross_entropy_with_logits(labels=self.label, logits=self.logits)
    #         self.loss = tf.reduce_mean(entropy, name='loss')
    
    # def optimize(self):
        # '''
        # Define training op
        # using Adam Gradient Descent to minimize cost
        # '''
        # self.opt = tf.train.AdamOptimizer(self.lr).minimize(self.loss, 
        #                                         global_step=self.gstep)

    # def summary(self):
    #     '''
    #     Create summaries to write on TensorBoard
    #     '''
    #     with tf.name_scope('summaries'):
    #         tf.summary.scalar('loss', self.loss)
    #         tf.summary.scalar('accuracy', self.accuracy)
    #         tf.summary.histogram('histogram_loss', self.loss)
    #         self.summary_op = tf.summary.merge_all()
    
    # def eval(self):
    #     '''
    #     Count the number of right predictions in a batch
    #     '''
    #     with tf.name_scope('predict'):
    #         self.preds = tf.nn.softmax(self.logits)
    #         correct_preds = tf.equal(tf.argmax(self.preds, 1), tf.argmax(self.label, 1))
    #         self.accuracy = tf.reduce_sum(tf.cast(correct_preds, tf.float32))

    # def build(self):
    #     '''
    #     Build the computation graph
    #     '''
    #     self.get_data()
    #     self.inference()
    #     self.loss()
    #     self.optimize()
    #     self.eval()
    #     self.summary()

    # def train_one_epoch(self, sess, saver, writer, epoch, step):
    #     start_time = time.time()
    #     sess.run(self.iterator.initializer, feed_dict={self.filenames: self.training_filenames})
    #     self.training = True
    #     total_loss = 0
    #     n_batches = 0
    #     try:
    #         while True:
    #             _, l, summaries = sess.run([self.opt, self.loss, self.summary_op])
    #             writer.add_summary(summaries, global_step=step)
    #             if (step + 1) % self.skip_step == 0:
    #                 print('Loss at step {0}: {1}'.format(step, l))
    #             step += 1
    #             total_loss += l
    #             n_batches += 1
    #     except tf.errors.OutOfRangeError:
    #         pass
    #     saver.save(sess, 'checkpoints/rf_iisc_Dronet', step)
    #     print('Average loss at epoch {0}: {1}'.format(epoch, total_loss/n_batches))
    #     print('Took: {0} seconds'.format(time.time() - start_time))
    #     return step

    # def eval_once(self, sess, writer, epoch, step):
    #     start_time = time.time()
    #     sess.run(self.iterator.initializer, feed_dict={self.filenames: self.testing_filenames})
    #     self.training = False
    #     total_correct_preds = 0
    #     try:
    #         while True:
    #             accuracy_batch, summaries = sess.run([self.accuracy, self.summary_op])
    #             writer.add_summary(summaries, global_step=step)
    #             total_correct_preds += accuracy_batch
    #     except tf.errors.OutOfRangeError:
    #         pass

    #     print('Accuracy at epoch {0}: {1} '.format(epoch, total_correct_preds/self.n_test))
    #     print('Took: {0} seconds'.format(time.time() - start_time))

    # def train(self):
    #     '''
    #     The train function alternates between training one epoch and evaluating
    #     '''
    #     safe_mkdir('checkpoints')
    #     safe_mkdir('checkpoints/rf_iisc_Dronet')
    #     writer = tf.summary.FileWriter('./graphs/rf_iisc_Dronet', tf.get_default_graph())

    #     with tf.Session() as sess:
    #         sess.run(tf.global_variables_initializer())
    #         saver = tf.train.Saver()
    #         ckpt = tf.train.get_checkpoint_state(os.path.dirname('tf_code/checkpoints/rf_iisc_Dronet/checkpoint'))
    #         if ckpt and ckpt.model_checkpoint_path:
    #             saver.restore(sess, ckpt.model_checkpoint_path)
    #         step = self.gstep.eval()

    #         for epoch in range(self.num_epochs):
    #             step = self.train_one_epoch(sess, saver, writer, epoch, step)
    #             self.eval_once(sess, writer, epoch, step)
        # writer.close()

    def restoreSession(self):
		# Online Inference
		self._saver = tf.train.Saver()
		self._session = tf.InteractiveSession()
		init_op = tf.global_variables_initializer()
		self._session.run(init_op)
		print("############")
		ckpt = tf.train.get_checkpoint_state(os.path.dirname('/home/rbccps/catkin_ws/src/bebop_rf/src/tf_code/checkpoints/rf_iisc_Dronet/checkpoint'))
		if ckpt and ckpt.model_checkpoint_path:
			print("Restoring Model!!")
			self._saver.restore(self._session, ckpt.model_checkpoint_path)
		else:
			print("Model not restored!!!")
    
    def infer(self, inp):
        preds = self._session.run(self.preds, feed_dict={self.X:inp, self.keep_prob:1.0})
        return preds
