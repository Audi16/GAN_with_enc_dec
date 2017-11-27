# -*- coding: gbk -*- 
import os
import tensorflow as tf 
from PIL import Image
import numpy as np 
import pandas as pd

cwd = os.getcwd()
imglist = np.loadtxt('F:/data/multi-pie/img/multiPie_train.txt',dtype=bytes).astype(str)
image_path = ''
pose_dict = dict(zip(['90','80','130','140','51','50','41','190','200'],range(9)))
N_p = 9

def onehot(y, y_dim):
    y_vec = np.zeros((len(y), y_dim), dtype=np.float32)
    for i, label in enumerate(y):
        y_vec[i,y[i]] = 1.0
    
    return  y_vec

writer = tf.python_io.TFRecordWriter("multiPie_train.tfrecords")
temp=pd.DataFrame(imglist)
dis_pose = temp[2].apply(lambda x:pose_dict[x]).values
gen_pose = temp[3].apply(lambda x:pose_dict[x]).values
id_labels = imglist[:,1]
for i in range(imglist.shape[0]):
    print(imglist[i,0])
    img = Image.open(image_path+imglist[i,0]).convert('L') #读取灰度图
    img = img.resize((100, 100))
    img_raw = img.tobytes()              #将图片转化为原生bytes
    id_label = id_labels[i].astype('int').tolist()
    dis_pose_label = dis_pose[i].astype('int').tolist()
    print(dis_pose_label)
    gen_pose_label = gen_pose[i].astype('int').tolist()
    print(gen_pose_label)
    example = tf.train.Example(features=tf.train.Features(feature={
            "id_label": tf.train.Feature(int64_list=tf.train.Int64List(value=[id_label])),
            "dis_pose_label": tf.train.Feature(int64_list=tf.train.Int64List(value=[dis_pose_label])),
            "gen_pose_label": tf.train.Feature(int64_list=tf.train.Int64List(value=[gen_pose_label])),
            'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
        }))
    writer.write(example.SerializeToString())  #序列化为字符串
writer.close()