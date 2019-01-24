import tensorflow as tf

IMAGE_WIDTH, IMAGE_HEIGHT = 240, 180

def weight_variable(shape):
  initializer = tf.contrib.layers.xavier_initializer()
  return tf.get_variable('weights', 
                         shape=shape, 
                         initializer=initializer)

def bias_variable(shape):
  initializer = tf.constant_initializer(0.1)
  return tf.get_variable('biases', 
                         shape=shape, 
                         initializer=initializer)

def conv(inp, 
         in_channels, 
         out_channels, 
         kernel_size, 
         scope, 
         how='SAME', 
         stride=[1,1,1,1]):
  with tf.variable_scope(scope):
    W = weight_variable([kernel_size,
                         kernel_size, 
                         in_channels, 
                         out_channels])
    b = bias_variable([out_channels])
  ret = tf.nn.conv2d(inp, W, stride, how)
  ret = tf.nn.bias_add(ret, b)
  return tf.nn.relu(ret)

def pool(inp, size=[1,2,2,1]):
  return tf.nn.max_pool(inp, size, size, "SAME")

def fc(inp, 
       in_channels, 
       out_channels, 
       scope, 
       use_relu=True):
  with tf.variable_scope(scope):
    W = weight_variable([in_channels, out_channels])
    b = bias_variable([out_channels])
  res = tf.nn.bias_add(tf.matmul(inp, W), b)
  if use_relu:
    return tf.nn.relu(res)
  else:
    return res

class Network(object):
  
  def  __init__(self):
    self.image_ = tf.placeholder(tf.float32, shape=[None, IMAGE_HEIGHT, IMAGE_WIDTH, 1])
    self.label_ = tf.placeholder(tf.float32, shape=[None, 3])
    self.conv1 = conv(self.image_, 1, 16, 3, "conv1")
    self.pool1 = pool(self.conv1)
    self.conv2 = conv(self.pool1, 16, 32, 3, "conv2")
    self.pool2 = pool(self.conv2)
    self.conv3 = conv(self.pool2, 32, 64, 3, "conv3")
    self.pool3 = pool(self.conv3)
    self.conv4 = conv(self.pool3, 64, 64, 3, "conv4")
    self.pool4 = pool(self.conv4)
    self.conv5 = conv(self.pool4, 64, 64, 3, "conv5")
    self.pool5 = pool(self.conv5)
    self.conv6 = conv(self.pool5, 64, 64, 3, "conv6")
    self.pool6 = pool(self.conv6)
    print self.pool6.get_shape()
    self.pool6 = tf.reshape(self.pool6, [-1, 3*4*64])
    self.fc1 = fc(self.pool6, 3*4*64, 256, "fc1")
    self.fc2 = fc(self.fc1, 256, 64, "fc2")
    self.output = fc(self.fc2, 64, 3, "fc3", use_relu=False)
    self.loss = tf.reduce_mean(tf.square(self.label_ - self.output))
    self.train_step = tf.train.AdamOptimizer(1e-4).minimize(self.loss)
    self.saver = tf.train.Saver()
