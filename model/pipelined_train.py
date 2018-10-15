import tensorflow as tf
import numpy as np
import sys
import os
import time
import math
from resource import getrusage as resource_usage,RUSAGE_SELF
from PIL import Image
if len(sys.argv) < 2:
    print("Supply path to a .tfrecords file as an argument")
    #sys.exit()
    print("For testing purposes, hard coding the tfrecord file")
#filenames = [str(sys.argv[1])]

#for testing purposes hardcording the tf record file 
classes = ['Asterionella','Aulocoseira','Colonial Cyanobacteria','Cryptomonas','Detritus','Dolichospermum','Filamentious cyanobacteria','Romeria','Staurastrum']

def _parser(example):
    #print(example)
    #feature_set = {'train/image':tf.FixedLenFeature([],tf.string),
    #        'train/label':tf.FixedLenFeature([],tf.float32)}

    feature_set = {'train/label':tf.FixedLenSequenceFeature([],tf.float32,allow_missing=True),
                    'train/image':tf.FixedLenFeature([],tf.string)}
    features = tf.parse_single_example(example, features = feature_set)

    image = tf.decode_raw(features['train/image'],tf.float32)
    image = tf.reshape(image,[256,256,3])
    image = tf.image.rot90(image, 0)
    #image = tf.contrib.image.rotate(image,45)
    label = tf.cast(features['train/label'],tf.float32)
    
    #print(example.features_shape)
    #print(image)
    #print(label)
    #print(label)
    #return image
    print(image)
    return image,label

with tf.Session() as session:


    training_filename = ["/home/szaman5/Phytoplankton_Classifier/test_data/train.tfrecords"]
    validation_filename = ["/home/szaman5/Phytoplankton_Classifier/test_data/validation.tfrecords"]

    filename = tf.placeholder(tf.string, shape=[None])

    dataset = tf.data.TFRecordDataset(filename)
   
    #print(dataset)
    #print(dataset.output_shapes)
    dataset = dataset.map(_parser,num_parallel_calls=40)
    #dataset = dataset.shuffle(buffer_size = 100)
    #dataset = dataset.repeat(10)
    dataset = dataset.batch(1)
   
    #dataset = dataset.prefetch(buffer_size = 100)
    iterator = dataset.make_initializable_iterator()
    #session.run(iterator.initializer)
    #print(session.run([dataset.output_shapes,dataset.output_types]))
    next_element = iterator.get_next()
    counter = 0

    NUM_EPOCHS = 1

    for _ in range(NUM_EPOCHS):
        session.run(iterator.initializer, feed_dict={filename:training_filename})
        img, lbl = session.run(next_element)
        picture = img[0]*255 
        im = Image.fromarray(picture.astype('uint8'))
        print(im) 
        im.save('test_rotated.tiff')
        #im.save("test_image.tiff")
        #print(img[0]*255)
        #print(img[0].shape)
        #print(img[0].dtype)
       #while True:
        #    try:
        #        img,lbl = session.run(next_element)
        #        img

        #    except tf.errors.OutOfRangeError as e:
        #        print("Epoch Complete")
        #        break
        #session.run(iterator.initializer, feed_dict={filename:validation_filename})

        #while True:
        #    try:
        #        img,lbl = session.run(next_element)

        #    except tf.errors.OutOfRangeError as e:
        #        print("Validation complete")
        #        break

        counter +=1

    print(counter)
    #print(session.run(next_element))
    #print(session.run([images,labels]))
    session.close()


