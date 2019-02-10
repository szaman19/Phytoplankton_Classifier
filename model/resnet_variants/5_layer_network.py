import tensorflow as tf
import os
import sys
import random 
import numpy as np 
from random import randint
from time import time 
from numpy.random import seed
from tensorflow import set_random_seed
import convnet as cnn 

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

#Dropout rate for the FC layer. May or may not be used, depending on the architecture
keep_prob = tf.placeholder(tf.float32, name = 'keep_prob')

#Graph Parameteres 

filter_size_conv1 = 5
num_filters_conv1 = 32

filter_size_conv2 = 3
num_filters_conv2 = 32

filter_size_conv3 = 3
num_filters_conv3 = 64

filter_size_conv4 = 3
num_filters_conv4 = 64

filter_size_conv5 = 3
num_filters_conv5 = 128

fc_layer_one = 1024
fc_layer_two = 2048

layer_conv1 = cnn.create_conv_layer(input=x,
        num_input_channels=num_channels,
        conv_filter_size=filter_size_conv1,
        num_filters=num_filters_conv1,
        name="Convolutional_Layer_1",
        pool = True, group_num=1)

layer_conv2 = cnn.create_conv_layer(input=layer_conv1,
        num_input_channels=num_filters_conv1,
        conv_filter_size=filter_size_conv2,
        num_filters=num_filters_conv2,
        name="Convolutional_Layer_2",
        pool = True,group_num=2)

layer_conv3 = cnn.create_conv_layer(input=layer_conv2,
        num_input_channels=num_filters_conv2,
        conv_filter_size=filter_size_conv3,
        num_filters=num_filters_conv3,
        name="Convolutional_Layer_3",
        pool = True, group_num =3)

layer_conv4 = cnn.create_conv_layer(input=layer_conv3,
        num_input_channels=num_filters_conv3,
        conv_filter_size=filter_size_conv4,
        num_filters=num_filters_conv4,
        name="Convolutional_Layer_4",
        pool = True,group_num=4)

layer_conv5 = cnn.create_conv_layer(input=layer_conv4,
        num_input_channels=num_filters_conv4,
        conv_filter_size=filter_size_conv5,
        num_filters=num_filters_conv5,
        name="Convolutional_Layer_5",
        pool = True, group_num=5)

layer_flat = cnn.create_flat_layer(layer_conv5)

num_features = layer_flat.get_shape()[1:4].num_elements()
layer_fc1= cnn.create_fc_layer(input=layer_flat, num_inputs= num_features, num_outputs=fc_layer_one, use_relu=False, use_leaky_relu=True,name="Fully_Connected_Layer_1")
layer_fc2= cnn.create_fc_layer(input=layer_fc1, num_inputs=fc_layer_one, num_outputs=fc_layer_two, use_relu=True, name="Fully_Connected_Layer_2")

with tf.name_scope('dropout'):
    keep_prob = tf.placeholder(tf.float32)
    dropout = tf.nn.dropout(x=layer_fc2, keep_prob=keep_prob, noise_shape=None, seed=3, name='dropout')


final_fc_layer = cnn.create_fc_layer(input = dropout, num_inputs=fc_layer_two, num_outputs=num_classes, use_relu=False, name = "Final_Layer")

with tf.name_scope('cross_entropy'):
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=final_fc_layer, labels=y_true)

with tf.name_scope('cost'):
    cost = tf.reduce_mean(cross_entropy)
tf.summary.scalar('cost',cost)
with tf.name_scope('train'):
    optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)

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

run_file = open('shallow_run_counter.txt','r')
run_counter = run_file.readlines()
run_counter = run_counter[-1].strip()
run_file.close()
train_file_dir = '/home/szaman5/Phytoplankton_Classifier/trained_model/shallow/summary/train_'+run_counter
test_file_dir = '/home/szaman5/Phytoplankton_Classifier/trained_model/shallow/summary/test_'+run_counter
train_writer = tf.summary.FileWriter(train_file_dir,session.graph)
test_writer = tf.summary.FileWriter(test_file_dir)
update_counter = int(run_counter) + 1
run_file = open('shallow_run_counter.txt', 'w')
run_file.write(str(update_counter)+"\n")
run_file.close()


saver = tf.train.Saver()
session.run(tf.global_variables_initializer())

def _parser(example):
    feature_set={'label':tf.FixedLenSequenceFeature([],tf.float32,allow_missing=True),
                'image':tf.FixedLenFeature([],tf.string)}
    features = tf.parse_single_example(example,features=feature_set)

    image = tf.decode_raw(features['image'], tf.float32)
    image = tf.reshape (image, [256,256,3])
    label = tf.cast(features['label'], tf.float32)
    #rand = randint(0,4)
    #image = tf.image.rot90(image, rand)
    return image, label

def train():
    training_filename = ["/home/szaman5/Phytoplankton_Classifier/data/train.tfrecords"]
    validation_filename = ["/home/szaman5/Phytoplankton_Classifier/data/validation.tfrecords"]
    
    data_dir = "/home/szaman5/Phytoplankton_Classifier/two_class_data/"
    data_list = [data_dir + "train0v" + str(i)+".tfrecords" for i in range(8)]

    validation_list = [data_dir + "validation0v"+ str(i)+".tfrecords" for i in range(8)]

    filename = tf.placeholder(tf.string, shape=[None])

    #dataset = tf.data.TFRecordDataset(filename)
    
    dataset = (tf.data.Dataset.from_tensor_slices(data_list).interleave(lambda x:tf.data.TFRecordDataset(x).map(_parser, num_parallel_calls = 48).prefetch(256),cycle_length=24,block_length=32)) 
    dataset = dataset.shuffle(buffer_size =2048)
    dataset = dataset.batch(32)
    dataset = dataset.prefetch(buffer_size = 512)
    
 
    #dataset = dataset.map(_parser,num_parallel_calls=32)
    #dataset = dataset.shuffle(buffer_size = 6400)
    #dataset = dataset.batch(32)
    #dataset = dataset.prefetch(buffer_size = 3200)
    
    iterator = dataset.make_initializable_iterator()

    next_element = iterator.get_next()

    val_dataset = (tf.data.Dataset.from_tensor_slices(validation_list).interleave(lambda x:tf.data.TFRecordDataset(x).map(_parser, num_parallel_calls = 48).prefetch(256),cycle_length=24,block_length=32)) 
    val_dataset = val_dataset.shuffle(buffer_size = 2048)
    val_dataset = val_dataset.batch(64)
    val_dataset = val_dataset.prefetch(buffer_size = 100)

    val_iterator = val_dataset.make_initializable_iterator()
    next_val_element = val_iterator.get_next()
    
    NUM_EPOCHS = 150
    #summary = session.run(merged)
    #train_writer.add_summary(summary,0)
    predictions = []
    actual = []
    confusion_matrix = 0
    
    for epoch in range(NUM_EPOCHS):
        #session.run(iterator.initializer, feed_dict={filename:training_filename})
        session.run(iterator.initializer)
        num_batches = 0
        epoch_train_accuracy = 0
        epoch_val_accuracy = 0
        while True:
            try:
                x_batch, y_true_batch = session.run(next_element)
                feed_dict_tr = {x: x_batch, y_true: y_true_batch, keep_prob:0.3}

                train,acc,summary = session.run([optimizer,accuracy,merged], feed_dict=feed_dict_tr)
                num_batches +=1
                epoch_train_accuracy += acc
                train_writer.add_summary(summary,epoch+1)
            except tf.errors.OutOfRangeError as e:
                break
        #session.run(iterator.initializer, feed_dict = {filename:validation_filename})
        session.run(val_iterator.initializer)
        valid_batches = 0
        while True:
            try:
                x_valid_batch, y_valid_batch = session.run(next_val_element)
                feed_dict_val = {x:x_valid_batch,y_true:y_valid_batch, keep_prob:1}
                val_loss,val_acc,summary,cm = session.run([cost,accuracy,merged,confusion], feed_dict_val)
                epoch_val_accuracy += val_acc
                confusion_matrix +=cm
                valid_batches +=1
                test_writer.add_summary(summary,epoch+1)
            except tf.errors.OutOfRangeError as e:
                saver.save(session, "/home/szaman5/Phytoplankton_Classifier/trained_model/shallow/")
                break
        confusion_matrix = tf.cast(confusion_matrix, tf.float32)
        confusion_matrix = tf.reshape(confusion_matrix,[1,num_classes,num_classes,1])
        #print(confusion_matrix.shape)
        #print(confusion_matrix)
        confusion_image = tf.summary.image(name="Confusion_Matrix", tensor=confusion_matrix)
        temp_im = session.run(confusion_image)
        test_writer.add_summary(temp_im)
        confusion_matrix = 0 

        msg = "Epoch {0} | Training accuracy {1:>6.4%}, Validation accuracy {2:>6.4%}"
        print(msg.format(epoch+1,epoch_train_accuracy/num_batches,epoch_val_accuracy/valid_batches))
        



train() 
