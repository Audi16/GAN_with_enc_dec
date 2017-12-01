import os
import tensorflow as tf 
from PIL import Image
import numpy as np 

cwd = os.getcwd()
imglist = np.loadtxt('/home/audi/GANs/data/nearfrontal_64/nearFrontal2.txt',dtype=bytes).astype(str)
image_path = '/home/audi/GANs/data/'

def onehot(y, y_dim):
    y_vec = np.zeros((len(y), y_dim), dtype=np.float32)
    for i, label in enumerate(y):
        y_vec[i,y[i]] = 1.0
    
    return  y_vec

writer = tf.python_io.TFRecordWriter("multiPie_nearfrontal.tfrecords")
for i in range(imglist.shape[0]):
    img = Image.open(image_path+imglist[i,0]).convert('L') #grey 
    img = img.resize((64, 64))
    img_raw = img.tobytes()              #to bytes
    id_label = imglist[i,1].astype('int64')
    print(imglist[i,0],id_label-1)
    example = tf.train.Example(features=tf.train.Features(feature={
            "id_label": tf.train.Feature(int64_list=tf.train.Int64List(value=[id_label-1])),
            'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
        }))
    writer.write(example.SerializeToString()) 
print 'have read %d images...'%(i+1)	
writer.close()
