import cv2
import os
import glob
from sklearn.utils import shuffle
import numpy as np
import multiprocessing
import tensorflow as tf
import imutils
'''
Roadmap for loader:

1. Create a dictionary of classes
2. Read file names in to the different num_classes
'''


writer = tf.python_io.TFRecordWriter('/home/szaman5/Phytoplankton_Classifier/complete_data/train.tfrecords')
val_writer = tf.python_io.TFRecordWriter('/home/szaman5/Phytoplankton_Classifier/complete_data/validation.tfrecords')

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
        total = len(files)*10 
        counter = 0
        num_images += total 
        msg = '{} : {}'
        print(msg.format(fields, str(total)))
        for fl in files:
            image = cv2.imread(fl)

            for i in range(0,360,36):
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
                if( counter > int(0.7 *total)):
                    write_to_tf_records(image,label,'validation')
                    val_images +=1
                else:
                    write_to_tf_records(image,label,'training')
                counter +=1
                num_images +=1
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

def write_to_tf_records(image, label,val):
    feature = {'label': _float_feature(label),
            'image': _bytes_feature(tf.compat.as_bytes(image.tostring())),}
    example = tf.train.Example(features=tf.train.Features(feature=feature))
    if val == 'validation':
        val_writer.write(example.SerializeToString())
    else:
        writer.write(example.SerializeToString())
def main():
    classes = ["Asterionella","Aulocoseira","Colonial Cyanobacteria","Detritus","Dinobryon","Dolichospermum","Filamentous Cyanobacteria","Fragilaria","Mougeotia","Pennate diatom","Tabellaria"] 
    #classes = ['Asterionella','Aulocoseira','Colonial Cyanobacteria','Cryptomonas','Detritus','Dolichospermum','Filamentous cyanobacteria','Romeria','Staurastrum']
    os.chdir("../")
    path = os.getcwd()
    path = os.path.join(path, "Training_Data/")
    print(path)
    load_train(path,256, classes)
    writer.close()
main()
