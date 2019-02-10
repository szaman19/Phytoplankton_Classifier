import tensorflow as tf
import os
import sys
import random 
import numpy as np 
from random import randint
from time import time 
from numpy.random import seed
from tensorflow import set_random_seed

def variable_summaries(var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
            tf.summary.scalar('stddev', stddev)
            tf.summary.scalar('max', tf.reduce_max(var))
            tf.summary.scalar('min', tf.reduce_min(var))
            tf.summary.histogram('histogram', var)

def create_weights(shape):
    return tf.Variable(tf.truncated_normal(shape,stddev=0.05))
def create_biases(size):
    return tf.Variable(tf.constant(0.05, shape=[size]))

def batch_norm():
    print("todo")
def create_conv_layer(input,
        num_input_channels,
        conv_filter_size,
        num_filters,
        name,pool,group_num=0,
        keep_prob = 0.3):
    shape = [conv_filter_size,conv_filter_size, num_input_channels, num_filters]

    with tf.name_scope(name):
        with tf.name_scope('weight'):
            weights = create_weights(shape)
            #variable_summaries(weights)
            #tf.summary.histogram('weights',weights)
        with tf.name_scope('bias'):
            biases = create_biases(num_filters)
            #variable_summaries(biases)
            #tf.summary.histogram('biases',biases)
        with tf.name_scope('layer'):
            layer = tf.nn.conv2d(input=input,filter=weights,strides=[1,1,1,1],padding='SAME',name=name)
            layer += biases
            #tf.summary.histogram('conv_layers',layer)

        if pool:
            max_pool_name = "maxpool"+str(group_num)
            with tf.name_scope(max_pool_name):
                layer = tf.nn.max_pool(value=layer,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
        layer = tf.nn.relu(layer)

        dropout_layer_name = 'dropout'+str(group_num)
        layer = tf.nn.dropout(layer,keep_prob=keep_prob, noise_shape=None, seed=3,name=dropout_layer_name)
        #tf.summary.histogram('activations',layer)
    return layer

def create_flat_layer(layer):
    layer_shape = layer.get_shape()
    num_features = layer_shape[1:4].num_elements()
    layer=tf.reshape(layer,[-1,num_features])
    return layer

def create_fc_layer(input,num_inputs,num_outputs,name,use_relu=True,use_leaky_relu=False):
    with tf.name_scope(name):
        with tf.name_scope('weight'):
            weights = create_weights(shape=[num_inputs, num_outputs])
        with tf.name_scope('biases'):
            biases = create_biases(num_outputs)
        with tf.name_scope('Wx_plus_b'):
            layer = tf.matmul(input, weights)+biases
        if use_relu:
            layer = tf.nn.relu(layer)
        elif use_leaky_relu:
            layer = tf.nn.leaky_relu(layer, alpha=0.2)
    return layer


