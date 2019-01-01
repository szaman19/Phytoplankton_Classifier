import tensorflow as tf
import os
import sys
import random 
import numpy as np 
from random import randint
from time import time 
from numpy.random import seed
from tensorflow import set_random_seed
'''
    The Dataset and Image Properties are written below. The code assumes the images are of constant size with correct buffers. 
    The preprocessing and input functions are defined in the next section.
'''
#The Classes that will be used to train and identify in the model 

#classes =['Asterionella','Aulocoseira','Colonial Cyanobacteria','Cryptomonas','Detritus','Dolichospermum','Filamentous cyanobacteria','Romeria','Staurastrum']
#classes = ["Ankistrodesmus","Aphanizomenon","Aphanothece","Asterionella","Aulocoseira","Auxospore","Ceratium","Chroococcus","Chroomonas","Ciliate","Cilliate","Closterium","Colonial Cyanobacteria","Cosmarium","Cryptomonas","Cymbella","Detritus","Diatoma","Dictyospaerium","Dinobryon","Dolichospermum","Eudorina","Euglena","Eunotia","Filamentous Cyanobacteria","Fragilaria","Gymnodinium","Heliozoan","Kirchneriella","Mallomonas","Merismopedia","Mesodinium","Micractinium","Microcystis","Mougeotia","Navicula","Nostoc","Oocystis","Pandorina","Pediastrum","Pennate diatom","Peridinium","Rhizosolenia","Romeria","Scenedesmus","Snowella","Sphaerocystis","Staurastrum","Staurodesmus","Stephanodiscus","Synedra","Synura","Tabellaria","Tetraspora","Trachelomonas","Ulnaria","Unidentified chlorophyte","Unidentified diatom","Uroglenopsis","Woronichinia","Zooplankton"]
classes = ["Colonial Cyanobacteria","Detritus"] 

#Get the number of classes
# - Used for the size of the final fully connedcted layer (Softmax)
# - To instantiate the Labels one-hot tensor 

num_classes= len(classes)

#The number of color channels that will be used. B&W = 1, Color = 3

num_channels = 3

#The image sized 

img_size = 256

'''
    The input pipeline map and preprocessing functions are defined in the next section. The map function is used to load 
    images that are saved as a binary format in .tfrecord file.

    The preprocessing function is used to add random perturbations to the images to add more generality to the network. 
'''

'''
    The computation graph is defined in the following section. 
'''
#Start the Graph 

session = tf.Session()

#Insert the input layer into the graph as a placeholder tensor. 
#The shape of the tensor is [batch_size, img_size, img_size, num_channels]

with tf.name_scope('input'):
    #Insert the input layer into the graph as a placeholder tensor. 
    #The shape of the tensor is [batch_size, img_size, img_size, num_channels]
    x = tf.placeholder(tf.float32, shape=[None, img_size, img_size, num_channels], name='input_images')
    #Labels
    #Holds the label of the input image as a one-hot tensor. 
    #The vector is 1 c num_classes, where the correct class is 1 and the rest is 0
    y_true = tf.placeholder(tf.float32, shape=[None,num_classes], name='label_true')

    #Gets the max of y_true
    y_true_cls = tf.argmax(y_true, dimension = 1)

    #Write summary for tensorboard 
    write_summary = tf.placeholder_with_default(False, shape=(), name='summary_flag')
    
    #Dropout rate for the FC layer. May or may not be used, depending on the architecture
    keep_prob = tf.placeholder(tf.float32, name = 'keep_prob')

#Graph Parameteres 

filter_size_conv1 = 3
num_filters_conv1 = 64

filter_size_conv2 = 3
num_filters_conv2 = 64

filter_size_conv3 = 3
num_filters_conv3 = 128

filter_size_conv4 = 3
num_filters_conv4 = 128

filter_size_conv5 = 3
num_filters_conv5 = 256

filter_size_conv6 = 3
num_filters_conv6 = 256

filter_size_conv7 = 3
num_filters_conv7 = 512

filter_size_conv8 = 3
num_filters_conv8 = 512

filter_size_conv9 = 3
num_filters_conv9 = 512

filter_size_conv10 = 3
num_filters_conv10 = 512


fc_layer_one = 4096
fc_layer_two = 8192

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

def create_conv_layer(input,
        num_input_channels,
        conv_filter_size,
        num_filters,
        name,pool,group_num=0):
    shape = [conv_filter_size,conv_filter_size, num_input_channels, num_filters]

    with tf.name_scope(name):
        with tf.name_scope('weight'):
            weights = create_weights(shape)
            #x = tf.cond(write_summary,lambda:print("Writing Summary"), lambda:print("Not writing summary"))
            #variable_summaries(weights)
            #tf.summary.histogram('weights',weights)
        with tf.name_scope('bias'):
            biases = create_biases(num_filters)
            #if write_summary:
            #    variable_summaries(biases)
            #variable_summaries(biases)
            #tf.summary.histogram('biases',biases)
        with tf.name_scope('layer'):
            layer = tf.nn.conv2d(input=input,filter=weights,strides=[1,1,1,1],padding='SAME',name=name)
            layer += biases
            #if write_summary:
            #tf.summary.histogram('conv_layer', layer)
            #tf.summary.histogram('conv_layers',layer)
        if pool:
            max_pool_name = "maxpool"+str(group_num)
            with tf.name_scope(max_pool_name):
                layer = tf.nn.max_pool(value=layer,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
        layer = tf.nn.relu(layer)
        #if write_summary:
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

layer_conv1 = create_conv_layer(input=x,
        num_input_channels=num_channels,
        conv_filter_size=filter_size_conv1,
        num_filters=num_filters_conv1,
        name="Convolutional_Layer_1",
        pool = False)

layer_conv2 = create_conv_layer(input=layer_conv1,
        num_input_channels=num_filters_conv1,
        conv_filter_size=filter_size_conv2,
        num_filters=num_filters_conv2,
        name="Convolutional_Layer_2",
        pool = True,group_num=1)

layer_conv3 = create_conv_layer(input=layer_conv2,
        num_input_channels=num_filters_conv2,
        conv_filter_size=filter_size_conv3,
        num_filters=num_filters_conv3,
        name="Convolutional_Layer_3",
        pool = False)

layer_conv4 = create_conv_layer(input=layer_conv3,
        num_input_channels=num_filters_conv3,
        conv_filter_size=filter_size_conv4,
        num_filters=num_filters_conv4,
        name="Convolutional_Layer_4",
        pool = True,group_num=2)

layer_conv5 = create_conv_layer(input=layer_conv4,
        num_input_channels=num_filters_conv4,
        conv_filter_size=filter_size_conv5,
        num_filters=num_filters_conv5,
        name="Convolutional_Layer_5",
        pool = False)

layer_conv6 = create_conv_layer(input=layer_conv5,
        num_input_channels=num_filters_conv5,
        conv_filter_size=filter_size_conv6,
        num_filters=num_filters_conv6,
        name="Convolutional_Layer_6",
        pool = True, group_num=3)

layer_conv7 = create_conv_layer(input=layer_conv6,
        num_input_channels=num_filters_conv6,
        conv_filter_size=filter_size_conv7,
        num_filters=num_filters_conv7,
        name="Convolutional_Layer_7",
        pool = False)

layer_conv8 = create_conv_layer(input=layer_conv7,
        num_input_channels=num_filters_conv7,
        conv_filter_size=filter_size_conv8,
        num_filters=num_filters_conv8,
        name="Convolutional_Layer_8",
        pool = True, group_num=4)

layer_conv9 = create_conv_layer(input=layer_conv8,
        num_input_channels=num_filters_conv8,
        conv_filter_size=filter_size_conv9,
        num_filters=num_filters_conv9,
        name="Convolutional_Layer_9",
        pool = False)

layer_conv10 = create_conv_layer(input=layer_conv9,
        num_input_channels=num_filters_conv9,
        conv_filter_size=filter_size_conv10,
        num_filters=num_filters_conv10,
        name="Convolutional_Layer_10",
        pool = True, group_num=5)

layer_flat = create_flat_layer(layer_conv10)

num_features = layer_flat.get_shape()[1:4].num_elements()
layer_fc1= create_fc_layer(input=layer_flat, num_inputs= num_features, num_outputs=fc_layer_one, use_relu=True,name="Fully_Connected_Layer_1")
layer_fc2= create_fc_layer(input=layer_fc1, num_inputs=fc_layer_one, num_outputs=fc_layer_two, use_relu=True, name="Fully_Connected_Layer_2")

with tf.name_scope('dropout'):
    keep_prob = tf.placeholder(tf.float32)
    dropout = tf.nn.dropout(x=layer_fc2, keep_prob=keep_prob, noise_shape=None, seed=3, name='dropout')


final_fc_layer = create_fc_layer(input = dropout, num_inputs=fc_layer_two, num_outputs=num_classes, use_relu=False, name = "Final_Layer")

with tf.name_scope('cross_entropy'):
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=final_fc_layer, labels=y_true)

with tf.name_scope('cost'):
    cost = tf.reduce_mean(cross_entropy)
tf.summary.scalar('cost',cost)
with tf.name_scope('train'):
    optimizer = tf.train.AdamOptimizer(learning_rate=8e-7).minimize(cost)

with tf.name_scope('accuracy'):
    with tf.name_scope('correct_prediction'):
        y_pred = tf.nn.softmax(final_fc_layer,name="prediction")
        y_pred_cls = tf.argmax(y_pred, dimension =1 )
        correct_prediction = tf.equal(y_pred_cls, y_true_cls)
    with tf.name_scope('accuracy'):
        accuracy=tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
tf.summary.scalar('accuracy', accuracy)

with tf.name_scope('confusion_matrix'):
    confusion = tf.confusion_matrix(y_true_cls, y_pred_cls,num_classes=num_classes)
    

merged = tf.summary.merge_all()

run_file = open('run_counter.txt','r')
run_counter = run_file.readlines()
run_counter = run_counter[-1].strip()
run_file.close()
train_file_dir = '/home/szaman5/Phytoplankton_Classifier/balanced_model/vgg/summary/train_'+run_counter
test_file_dir = '/home/szaman5/Phytoplankton_Classifier/balanced_model/vgg/summary/test_'+run_counter
train_writer = tf.summary.FileWriter(train_file_dir,session.graph)
test_writer = tf.summary.FileWriter(test_file_dir)
update_counter = int(run_counter) + 1
run_file = open('run_counter.txt', 'w')
run_file.write(str(update_counter)+"\n")
run_file.close()


saver = tf.train.Saver()
#confusion_matrix = None
session.run(tf.global_variables_initializer())

def _parser(example):
    feature_set={'label':tf.FixedLenSequenceFeature([],tf.float32,allow_missing=True),
                'image':tf.FixedLenFeature([],tf.string)}
    features = tf.parse_single_example(example,features=feature_set)

    image = tf.decode_raw(features['image'], tf.float32)
    image = tf.reshape (image, [256,256,3])
    label = tf.cast(features['label'], tf.float32)
    
    return image, label

def train():
    #training_filename = ["/home/szaman5/Phytoplankton_Classifier/complete_data/train.tfrecords"]
    validation_filename = ["/home/szaman5/Phytoplankton_Classifier/two_class_data/validation.tfrecords"]
    data_dir = "/home/szaman5/Phytoplankton_Classifier/two_class_data/"
    data_list = [data_dir + "train" + str(i)+".tfrecords" for i in range(100)]
   
    #dataset = tf.data.TFRecordDataset(filename)
    
    #dataset = dataset.map(_parser,num_parallel_calls=32)
    dataset = (tf.data.Dataset.from_tensor_slices(data_list).interleave(lambda x:tf.data.TFRecordDataset(x).map(_parser, num_parallel_calls = 48).prefetch(256),cycle_length=24,block_length=32)) 
    dataset = dataset.shuffle(buffer_size =2048)
    dataset = dataset.batch(32)
    dataset = dataset.prefetch(buffer_size = 512)
    
    iterator = dataset.make_initializable_iterator()

    next_element = iterator.get_next()

    filename = tf.placeholder(tf.string, shape=[None])
  
    val_dataset = tf.data.TFRecordDataset(filename)
    val_dataset = val_dataset.map(_parser, num_parallel_calls=20)
    val_dataset = val_dataset.shuffle(buffer_size = 2048)
    val_dataset = val_dataset.batch(64)
    val_dataset = val_dataset.prefetch(buffer_size = 100)

    val_iterator = val_dataset.make_initializable_iterator()
    next_val_element = val_iterator.get_next()
    NUM_EPOCHS = 40
    #summary = session.run(merged)
    #train_writer.add_summary(summary,0)
    confusion_matrix = 0
    #Confusion Matrix Image
    #conf_matrix = tf.placeholder(tf.float32, shape=[num_classes,num_classes,1], name='confustion_matrix')
   
    for epoch in range(NUM_EPOCHS):
        session.run(iterator.initializer)
        num_batches = 0
        epoch_train_accuracy = 0
        epoch_val_accuracy = 0
        epoch_train_cost = 0
        epoch_val_cost = 0
        while True:
            try:
                x_batch, y_true_batch = session.run(next_element)
                feed_dict_tr = {x: x_batch, y_true: y_true_batch, keep_prob:0.3}
                train,acc,t_cost,summary = session.run([optimizer,accuracy,cost,merged], feed_dict=feed_dict_tr)
                num_batches +=1
                epoch_train_accuracy += acc
                epoch_train_cost += t_cost
                #print(pred)
                #print(lab)
                train_writer.add_summary(summary,epoch+1)
            except tf.errors.OutOfRangeError as e:
                break
        session.run(val_iterator.initializer, feed_dict = {filename:validation_filename})
        valid_batches = 0
        while True:
            try:
                x_valid_batch, y_valid_batch = session.run(next_val_element)
                feed_dict_val = {x:x_valid_batch,y_true:y_valid_batch, keep_prob:1}
                val_loss,val_acc,summary,cm = session.run([cost,accuracy,merged,confusion], feed_dict_val)
                epoch_val_accuracy += val_acc
                epoch_val_cost += val_loss
                valid_batches +=1
                confusion_matrix +=cm
                test_writer.add_summary(summary,epoch+1)
            except tf.errors.OutOfRangeError as e:
                saver.save(session, "/home/szaman5/Phytoplankton_Classifier/balanced_model/vgg/")
                break
        
        #iprint(confusion_matrix) 
        confusion_matrix = tf.cast(confusion_matrix, tf.float32)
        confusion_matrix = tf.reshape(confusion_matrix,[1,num_classes,num_classes,1])
        #print(confusion_matrix.shape)
        #print(confusion_matrix)
        confusion_image = tf.summary.image(name="Confusion_Matrix", tensor=confusion_matrix)
        temp_im = session.run(confusion_image)
        test_writer.add_summary(temp_im)
        confusion_matrix = 0 
        msg = "Epoch {0} | Training accuracy {1:>6.4%}, Validation accuracy {2:>6.4%}, Training cost {3:>6.4}, Validation cos {4:>6.4%}"
        print(msg.format(epoch+1,epoch_train_accuracy/num_batches,epoch_val_accuracy/valid_batches,epoch_train_cost / num_batches,epoch_val_cost / valid_batches))
        



train()       
