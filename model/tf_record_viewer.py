import tensorflow as tf
import math
import random
import os
import numpy as np

with tf.Session() as session:
    reader = tf.TFRecordReader()

    filename_queue = tf.train.string_input_producer(['/home/szaman5/Phytoplankton_Classifier/data/train.tfrecords'],
        num_epochs=10)

    classes = ['Asterionella','Aulocoseira','Colonial Cyanobacteria','Cryptomonas','Detritus','Dolichospermum','Filamentious cyanobacteria','Romeria','Staurastrum']

    _, serialized_example = reader.read(filename_queue)

    print(serialized_example)
    feature_set = { 'train/label':tf.FixedLenFeature([len(classes)],tf.float32),
        'train/image':tf.FixedLenFeature([],tf.string)}

    features = tf.parse_single_example(serialized_example, features = feature_set)

    image = tf.decode_raw(features['train/image'],tf.float32)
    image = tf.reshape(image,[256,256,3])

    label = tf.cast(features['train/label'],tf.float32)

    print(label)

    print(image)
   # images, labels = tf.train.shuffle_batch([image, label], batch_size=10, capacity=30, num_threads=1, min_after_dequeue=10)

    init_op = tf.group(tf.global_variables_initializer(),tf.local_variables_initializer())

    session.run(init_op)
    coord = tf.train.Coordinator()

    threads = tf.train.start_queue_runners(coord=coord)
    counter = 0
    try:
        while True:
            img,lbl = session.run([image,label])
            #print(example)
            counter += 1
    except tf.errors.OutOfRangeError as e:
        coord.request_stop(e)
    finally:
        coord.request_stop()
        coord.join(threads)

    print(counter)
    
    session.close()


