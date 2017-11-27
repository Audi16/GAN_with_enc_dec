#coding:utf-8
import os
import tensorflow as tf 
from PIL import Image
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

def get_batch(idx):
    temp = imglist[idx*batch_size:(idx+1)*batch_size]
    batch_images = np.zeros((batch_size,96,96,1))
    for i in range(temp.shape[0]):
        img = Image.open(image_path+temp[i,0]).convert('L') #读取灰度图
        array_img = np.array(img)
        #batch_images[i,:,:,0] = (array_img - array_img.mean())/array_img.std()
        batch_images[i,:,:,0] = array_img / 127.5 -1.
    batch_labels = temp[:,1].astype('int')
    temp=pd.DataFrame(temp)
    batch_dis_pose = temp[3].apply(lambda x:pose_dict[x]).values
    batch_gen_pose = temp[2].apply(lambda x:pose_dict[x]).values
    batch_gen_pose_onehot = onehot(batch_gen_pose,N_p)
    return batch_images,batch_labels-1,batch_dis_pose,batch_gen_pose,batch_gen_pose_onehot

def data_augmentation(image, label, batch_size,crop_size):  
        #数据扩充变换 
    crop_image=tf.image.central_crop(image,0.96)        
    distorted_image = tf.random_crop(image, [crop_size, crop_size, 3])#随机裁剪  
    #distorted_image = tf.image.random_flip_up_down(distorted_image)#上下随机翻转  
    #distorted_image = tf.image.random_brightness(distorted_image,max_delta=63)#亮度变化  
    #distorted_image = tf.image.random_contrast(distorted_image,lower=0.2, upper=1.8)#对比度变化  
  
    #生成batch  
    #shuffle_batch的参数：capacity用于定义shuttle的范围，如果是对整个训练数据集，获取batch，那么capacity就应该够大  
    #保证数据打的足够乱  
    images, label_batch = tf.train.shuffle_batch([distorted_image, label],batch_size=batch_size,  
                                                 num_threads=16,capacity=50000,min_after_dequeue=10000)  
    #images, label_batch=tf.train.batch([distorted_image, label],batch_size=batch_size) 

def read_and_decode(filename):
    #根据文件名生成一个队列
    filename_queue = tf.train.string_input_producer([filename])
    batch_images = np.zeros((batch_size,96,96,1))
    reader = tf.TFRecordReader()
    for i in range(batch_size):
		_, serialized_example = reader.read(filename_queue)   #返回文件名和文件
		features = tf.parse_single_example(serialized_example,
										   features={
											   'id_label': tf.FixedLenFeature([], tf.int64),
											   'dis_pose_label': tf.FixedLenFeature([], tf.int64),
											   'gen_pose_label': tf.FixedLenFeature([], tf.int64),
											   'img_raw' : tf.FixedLenFeature([], tf.string),
										   })

		img = tf.decode_raw(features['img_raw'], tf.uint8)
		img = tf.reshape(img, [100, 100, 1])
		#img = tf.cast(img, tf.float32) * (1. / 255) - 0.5
		img = tf.cast(img, tf.float32)/127.5 -1.
		batch_images[i,:,:,0] = img.eval()
		id_label = tf.cast(features['id_label'], tf.int32)
		d_pose_label = tf.cast(features['dis_pose_label'], tf.int32)
		g_pose_label = tf.cast(features['gen_pose_label'], tf.int32)
    return img,id_label,d_pose_label,g_pose_label
    
img,l1,l2,l3 = read_and_decode("multiPie_train.tfrecords")
'''
#使用shuffle_batch可以随机打乱输入
img_batch, label_batch = tf.train.shuffle_batch([img, label],
                                                batch_size=30, capacity=2000,
                                                min_after_dequeue=1000)
'''
init = tf.initialize_all_variables()

with tf.Session() as sess:    
    sess.run(init)
    threads = tf.train.start_queue_runners(sess=sess)    
    image = sess.run(img)   
    fig=plt.figure()     
    ax = fig.add_subplot(111)   
    ax.imshow(image,cmap="gray")    
    plt.show()  