import cv2
import os
import glob
from sklearn.utils import shuffle
import numpy as np
import multiprocessing
import tensorflow as tf
import imutils
import random
'''
Roadmap for loader:

1. Create a dictionary of classes
2. Read file names in to the different num_classes
'''



writers = []
test_writers  = []
for i in range(0,360,10):
    degree_writer = []
    degree_writer_tester = []
    for k in range(8):

        writer = tf.python_io.TFRecordWriter('/home/szaman5/Phytoplankton_Classifier/two_class_data/train'+str(i)+'v'+str(k)+'.tfrecords')
        val_writer = tf.python_io.TFRecordWriter('/home/szaman5/Phytoplankton_Classifier/two_class_data/validation'+str(i)+'v'+str(k)+'.tfrecords')
        degree_writer.append(writer)
        degree_writer_tester.append(val_writer)
    writers.append(degree_writer)
    test_writers.append(degree_writer_tester)
def load_train(train_path, image_size, classes):
    images = []
    labels = []
    img_names = []
    cls = []
    num_images = 0
    val_images = 0
    print('Going to read training images')
    for fields in classes:
        index = classes.index(fields)
        #print('Now going to read {} files (Index: {})'.format(fields, index))
        path = os.path.join(train_path, fields,"*.tif")
        files = glob.glob(path)
        total = len(files) 
        counter = 0
        num_images += total 
        msg = '{} : {}'
        print(msg.format(fields, str(total)))
        for fl in files:
            image = cv2.imread(fl)

            #For each image do 36 rotations 
            #for i in range(0,360,10):
            image = imutils.rotate_bound(image, i)
 
            image = cv2.resize(image, (image_size, image_size),0,0, cv2.INTER_LINEAR)
            image = image.astype(np.float32)
            image = np.multiply(image, 1.0 / 255.0)
            #images.append(image)
            label = np.zeros(len(classes))
            label[index] = 1.0
            #print(label)
            #labels.append(label)
            #flbase = os.path.basename(fl)
            #img_names.append(flbase)
            #cls.append(fields)
        #    print(label)

                #Reserve the last  30% of images for test set
                #print(i)
                #degree = i // 10
            if( counter > int(0.7 *total)):
                                        
                write_to_tf_records(image,label,'validation',0)
                val_images +=1
            else:
                write_to_tf_records(image,label,'training',0)
            num_images += 1
            counter +=1
        #images = np.array(images)  #problem is here
        #labels = np.array(labels)
        #img_names = np.array(img_names)
        #cls = np.array(cls)
   # print(val_images)
    print("Total Number of Images:", num_images)
    

def _float_feature(value):
  return tf.train.Feature(float_list=tf.train.FloatList(value=value))
def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def write_to_tf_records(image, label,val,degree):
    feature = {'label': _float_feature(label),
            'image': _bytes_feature(tf.compat.as_bytes(image.tostring())),}
    example = tf.train.Example(features=tf.train.Features(feature=feature))
    if val == 'validation':
        random_int = random.randrange(8)
        val_writer = test_writers[degree][random_int]
        val_writer.write(example.SerializeToString())
    else:
        #writer_num = random.randint(0,999)
        #print('*'*80)
        #print(len(writers))
        #print(degree)
        random_int = random.randrange(8)
        writer = writers[degree][random_int]
        writer.write(example.SerializeToString())
def main():
    classes = ["Colonial Cyanobacteria","Detritus"] 
    #classes = ['Asterionella','Aulocoseira','Colonial Cyanobacteria','Cryptomonas','Detritus','Dolichospermum','Filamentous cyanobacteria','Romeria','Staurastrum']
    os.chdir("../")
    path = os.getcwd()
    path = os.path.join(path, "Training_Data/")
    print(path)
    load_train(path,256, classes)
    writer.close()
main()
