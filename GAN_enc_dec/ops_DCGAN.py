#coding:utf-8
import os
import tensorflow as tf
from tensorflow.contrib import layers
from utils import *

ID_SIZE = 337
C_dim = 1
BATCH_SIZE = 64
checkpoint_dir = './ckpt/pre_train'#

try:
  image_summary = tf.image_summary
  scalar_summary = tf.scalar_summary
  histogram_summary = tf.histogram_summary
  merge_summary = tf.merge_summary
  SummaryWriter = tf.train.SummaryWriter
except:
  image_summary = tf.summary.image
  scalar_summary = tf.summary.scalar
  histogram_summary = tf.summary.histogram
  merge_summary = tf.summary.merge
  SummaryWriter = tf.summary.FileWriter
  
def bn(x,is_train):
    if is_train:        
        return tf.contrib.layers.batch_norm(x, decay = 0.9, epsilon = 1e-5, scale = True, 
                              is_training = is_train, 
                              updates_collections = None)
    else:
        return tf.contrib.layers.batch_norm(x, decay = 0.9, epsilon = 1e-5, scale = True, 
                              is_training = is_train, reuse = True, 
                              updates_collections = None)
def ln(x,is_train):
    if is_train: 
        return tf.contrib.layers.layer_norm(x)
    else:
        return tf.contrib.layers.layer_norm(x, trainable = is_train, reuse=True)    
def lrelu(x, leak=0.2, name="lrelu"):
  return tf.maximum(x, leak*x)
  
def conv2d(x,filters,kernel_size=5,strides=2):
    return layers.conv2d(x,filters,
             kernel_size=kernel_size,
             stride=strides, weights_initializer=tf.truncated_normal_initializer(stddev=0.02))
def deconv2d(x,filters,kernel_size=5,strides=2,padding='SAME'):
    return layers.conv2d_transpose(x,filters,
             kernel_size=kernel_size,
             padding=padding,
             stride=strides, weights_initializer=tf.random_normal_initializer(stddev=0.02))
def fully_connected(input, output_size, stddev=0.02):
    return layers.fully_connected(input, output_size, weights_initializer=tf.random_normal_initializer(stddev=stddev),activation_fn=None)
             
#Construction of generator and discriminator             
def encoder(image,is_train,reuse=False):
  with tf.variable_scope("encoder",reuse=reuse) as scope:
        h0 = lrelu(conv2d(image, filters=64))
        h1 = lrelu(bn(conv2d(h0, filters=64*2),is_train))
        h2 = lrelu(bn(conv2d(h1, filters=64*4),is_train))
        h3 = lrelu(bn(conv2d(h2, filters=64*8),is_train))
        #avg_pool = layers.avg_pool2d(h3,kernel_size=4,stride=4)
        h_id = fully_connected(tf.reshape(h3, [BATCH_SIZE, -1]),ID_SIZE)
        return h3, h_id

def discriminator(image,is_train,reuse=False):
    with tf.variable_scope("Discriminator",reuse=reuse) as scope:
        #print(tf.get_variable_scope().name,reuse)
        h0 = lrelu(conv2d(image, filters=64))
        h1 = lrelu(bn(conv2d(h0, filters=64*2),is_train))
        h2 = lrelu(bn(conv2d(h1, filters=64*4),is_train))
        h3 = lrelu(bn(conv2d(h2, filters=64*8),is_train))
        real = fully_connected(tf.reshape(h3, [BATCH_SIZE, -1]),1)
        h_id = fully_connected(tf.reshape(h3, [BATCH_SIZE, -1]),ID_SIZE)

        return tf.nn.sigmoid(h_id), h_id, tf.nn.sigmoid(real), real

def decoder(input_tensor,z,is_train,reuse=False):
  with tf.variable_scope("decoder",reuse=reuse) as scope:
    code = tf.reshape(input_tensor, [BATCH_SIZE, -1])
    gen_middle= tf.concat([code,z], 1) #tf 0.x，axis is before tensors。
    h0 = tf.nn.relu(bn(fully_connected(gen_middle,64*8*4*4),is_train))
    h1 = tf.nn.relu(bn(deconv2d(tf.reshape(h0,[-1,4,4,64*8]),filters=64*4),is_train))
    h2 = tf.nn.relu(bn(deconv2d(h1,filters=64*2),is_train))
    h3 = tf.nn.relu(bn(deconv2d(h2,filters=64*1),is_train))
    h4 = deconv2d(h3,filters = C_dim)
    
    generate = tf.nn.tanh(h4)
    return generate

def generator(image,z,is_train,reuse=False):
    with tf.variable_scope("Generator") as scope:
        if reuse:
            scope.reuse_variables() 
        encoded,enc_logit = encoder(image,is_train,reuse)
        return decoder(encoded,z,is_train,reuse),enc_logit



def load(sess, checkpoint_dir):
    import re
    print(" [*] Reading checkpoints...")
    #checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)
    
    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
      ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
      tf.train.Saver().restore(sess, os.path.join(checkpoint_dir, ckpt_name))
      counter = int(next(re.finditer("(\d+)(?!.*\d)",ckpt_name)).group(0))
      print(" [*] Success to read {}".format(ckpt_name))
      return True, counter
    else:
      print(" [*] Failed to find a checkpoint")
      return False, 0
