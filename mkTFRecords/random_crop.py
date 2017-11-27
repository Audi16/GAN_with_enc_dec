import tensorflow as tf
import matplotlib.image as img
import matplotlib.pyplot as plt
sess = tf.InteractiveSession()
image = img.imread('F:/data/multi-pie/img/001_01_01_041_06.png')

reshaped_image = tf.cast(image,tf.float32)
size = tf.cast(tf.shape(reshaped_image).eval(),tf.int32)
distorted_image = tf.random_crop(reshaped_image,[96,96,3])
central_crop = tf.image.central_crop(reshaped_image,0.96)
print(tf.shape(reshaped_image).eval())
print(tf.shape(distorted_image).eval())

fig = plt.figure()
#fig1 = plt.figure()
ax = fig.add_subplot(121)
ax1 = fig.add_subplot(232)
ax2 = fig.add_subplot(233)
ax3 = fig.add_subplot(235)
ax4 = fig.add_subplot(236)
ax.imshow(sess.run(reshaped_image))
ax1.imshow(sess.run(central_crop))
ax2.imshow(sess.run(tf.random_crop(reshaped_image,[height,width,3],seed=0)))
ax3.imshow(sess.run(tf.random_crop(reshaped_image,[height,width,3],seed=1)))
ax4.imshow(sess.run(distorted_image))
plt.show()