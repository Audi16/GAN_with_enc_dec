import os
import tensorflow as tf 
from PIL import Image
import numpy as np 
import cv2 as cv
import matplotlib.pyplot as plt

batch_size = 32

def data_augmentation(image, label, batch_size,crop_size):
    crop_image=tf.image.central_crop(image,0.96)        
    distorted_image = tf.random_crop(image, [crop_size, crop_size, 3])
    #distorted_image = tf.image.random_flip_up_down(distorted_image)
    #distorted_image = tf.image.random_brightness(distorted_image,max_delta=63)  
    #distorted_image = tf.image.random_contrast(distorted_image,lower=0.2, upper=1.8)  
    images, label_batch = tf.train.shuffle_batch([distorted_image, label],batch_size=batch_size,  
                                                 num_threads=16,capacity=50000,min_after_dequeue=10000)  
    #images, label_batch=tf.train.batch([distorted_image, label],batch_size=batch_size) 

def read_and_decode(filename):
    filename_queue = tf.train.string_input_producer([filename])
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue) 
    features = tf.parse_single_example(serialized_example,features={
						'id_label': tf.FixedLenFeature([], tf.int64),
						'img_raw' : tf.FixedLenFeature([], tf.string),
						})

    img = tf.decode_raw(features['img_raw'], tf.uint8)
    img = tf.reshape(img, [64, 64, 1])
    #img = tf.cast(img, tf.float32) * (1. / 255) - 0.5
    #img = tf.cast(img, tf.float32)/127.5 -1.
    id_label = tf.cast(features['id_label'], tf.int32)
    return img,id_label
 
img,label = read_and_decode("multiPie_nearfrontal.tfrecords")
img_batch, label_batch = tf.train.shuffle_batch([img, label],
                                                batch_size=batch_size, capacity=4000,
                                                min_after_dequeue=1000)

init = tf.initialize_all_variables()

with tf.Session() as sess:    
    sess.run(init)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord = coord) #
    for i in range(2):   
        images,labels = sess.run([img_batch,label_batch])#_batch
        flip_images = images[:,:,::-1,:] #flip the image in the left/right direction
        flip_labels = labels
        images = np.vstack((images,flip_images))
        labels = np.hstack((labels,flip_labels))
    print 'images:',type(images),images.shape,labels.shape
    cv.imshow('img',images[0,:,:,:]) #[0,:,:,:]
    cv.waitKey(0)
    cv.imshow('img',flip_images[0,:,:,:]) #[0,:,:,:]
    cv.waitKey(0)
    cv.imshow('img',images[batch_size,:,:,:]) #[0,:,:,:]
    cv.waitKey(0)
    print 'label is :',labels[0], labels[batch_size] 
    coord.request_stop()
    coord.join(threads)
