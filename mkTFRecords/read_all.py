import tensorflow as tf 
for s in tf.python_io.tf_record_iterator('multiPie_nearfrontal.tfrecords'):
   example = tf.train.Example()
   example.ParseFromString(s)
   image = example.features.feature['img_raw'].bytes_list.value
   label = example.features.feature['id_label'].int64_list.value
print type(image),type(label)
