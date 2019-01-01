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
classes = ["Asterionella","Aulocoseira","Colonial Cyanobacteria","Detritus","Dinobryon","Dolichospermum","Filamentous Cyanobacteria","Fragilaria","Mougeotia","Pennate diatom","Tabellaria"] 

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
    validation_filename = ["/home/szaman5/Phytoplankton_Classifier/complete_data/validation.tfrecords"]
    data_dir = "/home/szaman5/Phytoplankton_Classifier/complete_data/"
    data_list = [data_dir + "train" + str(i)+".tfrecords" for i in range(10)]



   
    #dataset = tf.data.TFRecordDataset(filename)
    
    dataset = (tf.data.Dataset.from_tensor_slices(data_list).interleave(lambda x:tf.data.TFRecordDataset(x).map(_parser, num_parallel_calls = 10),cycle_length=40,block_length=32))
    #dataset = dataset.map(_parser,num_parallel_calls=32)
    dataset = dataset.shuffle(buffer_size =65536)
    dataset = dataset.batch(256)
    dataset = dataset.prefetch(buffer_size = 512)
    
    iterator = dataset.make_initializable_iterator()

    next_element = iterator.get_next()
    

    filename = tf.placeholder(tf.string, shape=[None])

    val_dataset = tf.data.TFRecordDataset(filename)
    val_dataset = val_dataset.map(_parser, num_parallel_calls=20)
    val_dataset = val_dataset.shuffle(buffer_size = 2048)
    val_dataset = val_dataset.batch(2048)
    val_dataset = val_dataset.prefetch(buffer_size = 2048)

    val_iterator = val_dataset.make_initializable_iterator()
    next_val_element = val_iterator.get_next()

    NUM_EPOCHS = 200
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
        while True:
            try:
                x_batch, y_true_batch = session.run(next_element)
                feed_dict_tr = {x: x_batch, y_true: y_true_batch, keep_prob:0.3}
                results = session.run([y_true_cls], feed_dict=feed_dict_tr)
                print(results)
                num_batches +=1
                #epoch_train_accuracy += acc
                #print(pred)
                #print(lab)
                #train_writer.add_summary(summary,epoch+1)
            except tf.errors.OutOfRangeError as e:
                break
        session.run(val_iterator.initializer, feed_dict = {filename:validation_filename})
        valid_batches = 0
        while True:
            try:
                x_valid_batch, y_valid_batch = session.run(next_val_element)
                feed_dict_val = {x:x_valid_batch,y_true:y_valid_batch, keep_prob:1}
                res = session.run([y_true_cls], feed_dict_val)
                #epoch_val_accuracy += val_acc
                valid_batches +=1
                print(res)
                #confusion_matrix +=cm
            except tf.errors.OutOfRangeError as e:
                #saver.save(session, "/home/szaman5/Phytoplankton_Classifier/balanced_model/vgg/")
                break
        
        #iprint(confusion_matrix) 
        #confusion_matrix = tf.cast(confusion_matrix, tf.float32)
        #confusion_matrix = tf.reshape(confusion_matrix,[1,num_classes,num_classes,1])
        #print(confusion_matrix.shape)
        #print(confusion_matrix)
        #confusion_image = tf.summary.image(name="Confusion_Matrix", tensor=confusion_matrix)
        #temp_im = session.run(confusion_image)
        #test_writer.add_summary(temp_im)
        #confusion_matrix = 0 
        #msg = "Epoch {0} | Training accuracy {1:>6.4%}, Validation accuracy {2:>6.4%}"
        #print(msg.format(epoch+1,epoch_train_accuracy/num_batches,epoch_val_accuracy/valid_batches))
        



train() 
