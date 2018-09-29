import tensorflow as tf
import numpy as np
from layers import conv, conv3d, max_pool, max_pool3d, fc, dropout

class BaseModel:
    def __init__(self, mode="train", graph=None):
        # the properties all models must have
        # self.mode = mode
        # self.graph = graph

        if mode=="test":
            # assert graph != None, "\033[31m in test mode, you must provide the graph\033[0m"
            self.set_is_training(False)
        else:
            self.set_is_training(True)

        self.layer = {}

    def create(self):
        pass

    def set_model_input(self, inputs = None):
        self._input = inputs

    def set_dropout(self, dropout_placeholder, keep_prob = 0.5):
        self._dropout_pl = dropout_placeholder
        self._keep_prob = keep_prob
    
    def set_train_placeholder(self, plhs=None):
        if not isinstance(plhs, list):
            plhs = [plhs]
        self._train_plhs = plhs
    
    def set_prediction_placeholder(self, plhs=None):
        if not isinstance(plhs, list):
            plhs = [plhs]
        self._predict_plhs = plhs
    
    def set_is_training(self, is_training=True):
        self.is_training = is_training
    
class BaseTwoStream(BaseModel):
    """ base of two-stream class """
    def __init__(self, num_class=5,
                 num_channels=20,
                 im_height=112, 
                 im_width=112,
                 learning_rate=0.0001,
                 mode="train",
                 graph=None
                 ):
        """
        Args:
            num_class (int): number of image classes
            num_channels (int): number of input channels
            im_height, im_width (int): size of input image
                               Can be unknown when testing.
            learning_rate (float): learning rate of training
        """

        BaseModel.__init__(self, mode=mode, graph=graph)
        self.learning_rate = learning_rate
        self.num_channels = num_channels
        self.im_height = im_height
        self.im_width = im_width
        self.num_class = num_class

    def _create_input(self):
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        self.image = tf.placeholder(
            tf.float32, name='image',
            shape=[None, self.im_height, self.im_width, self.num_channels])

        self.label = tf.placeholder(tf.int64, [None], 'label')

        self.set_model_input([self.image, self.keep_prob])
        self.set_dropout(self.keep_prob, keep_prob=0.5)
        self.set_train_placeholder([self.image, self.label])
        self.set_prediction_placeholder(self.image)

class TwoStream(BaseTwoStream):
    
    def _create_conv(self):
        self._create_input()
        arg_scope = tf.contrib.framework.arg_scope
        with arg_scope([conv], nl=tf.nn.relu,
                       trainable=True, mode=self.mode, graph=self.graph):
            conv1 = conv(self.image, 7, 96, 'conv1')
            mean1, var1 = tf.nn.moments(conv1, [0,1,2])
            conv1_bn = tf.nn.batch_normalization(conv1, mean1, var1, 0, 1, 1e-5)
            pool1 = max_pool(conv1_bn, 'pool1', padding='SAME')

            conv2 = conv(pool1, 5, 256, 'conv2')
            mean2, var2 = tf.nn.moments(conv2, [0,1,2])
            conv2_bn = tf.nn.batch_normalization(conv2, mean2, var2, 0, 1, 1e-5)
            pool2 = max_pool(conv2_bn, 'pool2', padding='SAME')

            conv3 = conv(pool2, 3, 512, 'conv3', stride = 1)

            conv4 = conv(conv3, 3, 512, 'conv4', stride = 1)

            conv5 = conv(conv4, 3, 512, 'conv5', stride = 1)
            pool5 = max_pool(conv5, 'pool5', padding='SAME')

            self.layer['conv1'] = conv1
            self.layer['conv2'] = conv2
            self.layer['conv3'] = conv3
            self.layer['conv4'] = conv4
            self.layer['pool5'] = pool5
            self.layer['conv_out'] = self.layer['conv5'] = conv5

        return pool5
    
    def create_model(self, mode="train", graph=None):
        self.mode = mode
        self.graph = graph
        print("\033[31m creating the model... \033[0m")
        conv_output = self._create_conv()
        # flatten the conv output
        shape = conv_output.get_shape().as_list()[1:]
        conv_out_flatten = tf.reshape(conv_output, [-1, int(np.prod(shape))])

        arg_scope = tf.contrib.framework.arg_scope
        with arg_scope([fc], trainable=True, mode=self.mode, graph=self.graph, nl=tf.nn.relu):
            fc6 = fc(conv_out_flatten, 4096, 'fc6')
            dropout_fc6 = dropout(fc6, self.keep_prob, self.is_training)

            fc7 = fc(dropout_fc6, 2048, 'fc7')
            dropout_fc7 = dropout(fc7, self.keep_prob, self.is_training)

            fc8 = fc(dropout_fc7, self.num_class, 'fc8')

        self.layer['fc6'] = fc6
        self.layer['fc7'] = fc7
        self.layer['output'] = self.layer['fc8'] = fc8
        self.output = fc8

    def loss(self):
        # get loss
        logit = self.output
        label = self.label
        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=label,logits=logit))
        return loss

    def train(self):
        # get train_op
        train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss())
        return train_op

    def accuracy(self):
        # get accuracy
        logit = self.output
        label = self.label
        correct_prediction = tf.equal(tf.argmax(logit, 1), label)
        acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        return acc

class BaseVGG(BaseModel):
    """ base of two-stream class """
    def __init__(self, num_class=5,
                 num_channels=3,
                 length = 16,
                 im_height=112, 
                 im_width=112,
                 learning_rate=0.001,
                 is_load=False,
                 pre_train_path=None,
                 is_rescale=False,
                 mode="train",
                 graph=None):
        """
        Args:
            num_class (int): number of image classes
            num_channels (int): number of input channels
            im_height, im_width (int): size of input image
                               Can be unknown when testing.
            learning_rate (float): learning rate of training
        """
        BaseModel.__init__(self, mode=mode, graph=graph)

        self.learning_rate = learning_rate
        self.length = length
        self.num_channels = num_channels
        self.im_height = im_height
        self.im_width = im_width
        self.num_class = num_class

    def _create_input(self):
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        self.image = tf.placeholder(
            tf.float32, name='image',
            shape=[None, self.length, self.im_height, self.im_width, self.num_channels])

        self.label = tf.placeholder(tf.int64, [None], 'label')

        self.set_model_input([self.image, self.keep_prob])
        self.set_dropout(self.keep_prob, keep_prob=0.5)
        self.set_train_placeholder([self.image, self.label])
        self.set_prediction_placeholder(self.image)

class VGG16(BaseVGG):
    
    def _create_conv(self):
        self._create_input()
        arg_scope = tf.contrib.framework.arg_scope
        with arg_scope([conv3d], nl=tf.nn.relu,
                       trainable=True, mode=self.mode, graph=self.graph):
            conv1 = conv3d(self.image, 3, 64, 'conv1', 'wc1', 'bc1')
            pool1 = max_pool3d(conv1, 'pool1', padding='SAME', filter_size=[1,2,2])

            conv2 = conv3d(pool1, 3, 128, 'conv2', 'wc2', 'bc2')
            pool2 = max_pool3d(conv2, 'pool2', padding='SAME')

            conv3a = conv3d(pool2, 3, 256, 'conv3a', 'wc3a', 'bc3a')
            conv3b = conv3d(conv3a, 3, 256, 'conv3b', 'wc3b', 'bc3b')
            pool3 = max_pool3d(conv3b, 'pool3', padding='SAME')

            conv4a = conv3d(pool3, 3, 256, 'conv4a', 'wc4a', 'bc4a')
            conv4b = conv3d(conv4a, 3, 256, 'conv4b', 'wc4b', 'bc4b')
            pool4 = max_pool3d(conv4b, 'pool4', padding='SAME')

            conv5a = conv3d(pool4, 3, 256, 'conv5a', 'wc5a', 'bc5a')
            conv5b = conv3d(conv5a, 3, 256, 'conv5b', 'wc5b', 'bc5b')
            pool5 = max_pool3d(conv5b, 'pool5', padding='SAME')

            self.layer['conv1'] = conv1
            self.layer['conv2'] = conv2
            self.layer['conv3'] = conv3b
            self.layer['conv4'] = conv4b
            self.layer['pool5'] = pool5
            self.layer['conv_out'] = self.layer['conv5'] = conv5b

        return pool5
    
    def create_model(self, mode="train", graph=None):
        self.mode = mode
        self.graph = graph
        print("\033[31m creating the model... \033[0m")
        conv_output = self._create_conv()
        # flatten the conv output
        shape = conv_output.get_shape().as_list()[1:]
        conv_out_flatten = tf.reshape(conv_output, [-1, int(np.prod(shape))])

        arg_scope = tf.contrib.framework.arg_scope
        with arg_scope([fc], trainable=True, mode=self.mode, graph=self.graph, nl=tf.nn.relu):
            fc6 = fc(conv_out_flatten, 4096, 'fc6', 'wd1', 'bd1')
            dropout_fc6 = dropout(fc6, self.keep_prob, self.is_training)

            fc7 = fc(dropout_fc6, 4096, 'fc7', 'wd2', 'bd2')
            dropout_fc7 = dropout(fc7, self.keep_prob, self.is_training)

            fc8 = fc(dropout_fc7, self.num_class, 'fc8', 'out_changed', 'out_chaged')

        self.layer['fc6'] = fc6
        self.layer['fc7'] = fc7
        self.layer['output'] = self.layer['fc8'] = fc8
        self.output = fc8

    def loss(self):
        # get loss
        logit = self.output
        label = self.label
        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=label,logits=logit))
        return loss

    def train(self):
        # get train_op
        train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss())
        return train_op

    def accuracy(self):
        # get accuracy
        logit = self.output
        label = self.label
        correct_prediction = tf.equal(tf.argmax(logit, 1), label)
        acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        return acc

        

