import math
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops

from utils import *

def batch_norm(x, name="batch_norm"):
    return tf.contrib.layers.batch_norm(x, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True, scope=name)

def instance_norm(input, name="instance_norm"):
    with tf.compat.v1.variable_scope(name):
        depth = input.shape[3]
        scale = tf.compat.v1.get_variable(name="scale", shape=[depth], initializer=tf.random_normal_initializer(1.0, 0.02))
        offset = tf.compat.v1.get_variable(name="offset", shape=[depth], initializer=tf.constant_initializer(0.0))
        mean, variance = tf.nn.moments(input, axes=[1,2])
        epsilon = 1e-5
        inv = tf.compat.v1.rsqrt(variance + epsilon)
        normalized = (input-mean)*inv
        return scale*normalized + offset

def conv2d(input_, output_dim, ks=4, s=2, stddev=0.02, padding='SAME', name="conv2d"):
    with tf.compat.v1.variable_scope(name):
        init=tf.compat.v1.random.truncated_normal([ks,ks,input_.shape[3],output_dim],stddev=stddev)
        w = tf.compat.v1.Variable(init,name="W")
        return tf.nn.conv2d(input_,w, strides=[1,s,s,1], padding=padding)

def deconv2d(input_, output_dim, ks=4, s=2, stddev=0.02, name="deconv2d"):
    with tf.compat.v1.variable_scope(name):
        # init=tf.compat.v1.random.truncated_normal([ks,ks,output_dim,input_.shape[3]],stddev=stddev)
        # w = tf.compat.v1.Variable(init,name="W")
        # return tf.nn.conv2d_transpose(input_,w,output_dim, strides=[1,s,s,1], padding='SAME')
        return tf.compat.v1.layers.conv2d_transpose(input_,output_dim,(ks,ks),(s,s),padding='SAME',kernel_initializer=tf.random_normal_initializer(1.0, 0.02))

def lrelu(x, leak=0.2, name="lrelu"):
    return tf.maximum(x, leak*x)

def linear(input_, output_size, scope=None, stddev=0.02, bias_start=0.0, with_w=False):

    with tf.compat.v1.variable_scope(scope or "Linear"):
        matrix = tf.compat.v1.get_variable("Matrix", [input_.get_shape()[-1], output_size], tf.float32,
                                 tf.random_normal_initializer(stddev=stddev))
        bias = tf.compat.v1.get_variable("bias", [output_size],
            initializer=tf.constant_initializer(bias_start))
        if with_w:
            return tf.matmul(input_, matrix) + bias, matrix, bias
        else:
            return tf.matmul(input_, matrix) + bias

def RCU_Block(image,output_dim,ks=3,s=1,stddev=0.02,padding='SAME',name="RCU_Block",is_training=True):
    with tf.name_scope(name):
        init_1 = tf.compat.v1.truncated_normal([ks,ks,image.shape[3],output_dim],stddev=stddev)
        init_2 = tf.compat.v1.truncated_normal([ks,ks,output_dim,output_dim],stddev=stddev)
        init_3 = tf.compat.v1.truncated_normal([ks,ks,output_dim,output_dim],stddev=stddev)
        W1 = tf.Variable(init_1,name='W1')
        W2 = tf.Variable(init_2,name='W2')
        W3 = tf.Variable(init_3,name='W3')
        b1 = tf.Variable(tf.constant(0.1,shape=[output_dim]),name='b1')
        b2 = tf.Variable(tf.constant(0.1,shape=[output_dim]),name='b2')
        b3 = tf.Variable(tf.constant(0.1,shape=[output_dim]),name='b3')
        conv_2d_1 = tf.nn.conv2d(image,W1,strides=[1,s,s,1],padding='SAME')
        conv_2d_b_1 = tf.nn.bias_add(conv_2d_1,b1)
        #conv_2d_bn_1 = tf.compat.v1.layers.batch_normalization(conv_2d_b_1,training=is_training)
        conv_2d_ReLU_1 = instance_norm(tf.nn.relu(conv_2d_b_1),name+'bn_1')

        conv_2d_2 = tf.nn.conv2d(conv_2d_ReLU_1,W2,strides=[1,s,s,1],padding='SAME')
        conv_2d_b_2 = tf.nn.bias_add(conv_2d_2,b2)
        #conv_2d_bn_2 = tf.compat.v1.layers.batch_normalization(conv_2d_b_2,training=is_training)
        conv_2d_ReLU_2 = instance_norm(tf.nn.relu(conv_2d_b_2),name=name+'bn_2')

        conv_2d_3 = tf.nn.conv2d(conv_2d_ReLU_2,W3,strides=[1,s,s,1],padding='SAME')
        conv_2d_b_3 = tf.nn.bias_add(conv_2d_3,b3)
        #conv_2d_bn_3 = tf.compat.v1.layers.batch_normalization(conv_2d_b_3,training=is_training)
        conv_2d_ReLU_3 = instance_norm(tf.nn.relu(conv_2d_b_3),name+'bn_3')

        return tf.add(conv_2d_ReLU_1,conv_2d_ReLU_3)

def CP_Block(image,output_dim,ks=3,s=2,stddev=0.02,padding='SAME',name="CP_Block"):
    with tf.name_scope(name):
        init = tf.compat.v1.truncated_normal([ks,ks,image.shape[3],output_dim],stddev=stddev)
        W = tf.Variable(init,name='W')

        return tf.nn.conv2d(image,W,strides=[1,s,s,1],padding='SAME')
