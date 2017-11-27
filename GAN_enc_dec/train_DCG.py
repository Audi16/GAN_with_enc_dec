#coding:utf-8
import tensorflow as tf
from ops_DCGAN import *
from tensorflow.contrib import losses
import tensorflow.examples.tutorials.mnist.input_data as input_data
from PIL import  Image
#import pandas as pd
import tensorflow.contrib.slim as slim
#Parameters set
epoch = 500
LAMBDA = 10 # Gradient penalty lambda hyperparameter
LAMBDA_GAN = 1
LAMBDA_ID = 1
MODE = ''#wgan-gp
z_size = 50
CRITIC_ITERS = 5 # How many iterations to train the critic for
imglist = np.loadtxt('/home/root-gao/audi/GAN_enc_dec/nearfrontal_64/nearFrontal2.txt',dtype=bytes).astype(str)
image_path = '/home/root-gao/audi/GAN_enc_dec/'

#Placeholder set
inputs = tf.placeholder(tf.float32,[BATCH_SIZE,64,64,C_dim],name='real_images')
y = tf.placeholder(tf.int64, [BATCH_SIZE], name='labels')
z = tf.placeholder(tf.float32, [BATCH_SIZE,z_size], name='z')
#Fetch images
def get_batch_aug(idx):
    temp = imglist[idx*BATCH_SIZE/2:(idx+1)*BATCH_SIZE/2]
    batch_images = np.zeros((BATCH_SIZE,64,64,C_dim))
    for i in range(temp.shape[0]):
        img = Image.open(image_path+temp[i,0]).convert('L') #grey
        flip_img = img.transpose(Image.FLIP_LEFT_RIGHT)
        array_img = np.array(img)
        flip_array_img = np.array(flip_img)
        #batch_images[i,:,:,0] = (array_img - array_img.mean())/array_img.std()#Z-score
        batch_images[i,:,:,0] = array_img/127.5 - 1.
        batch_images[i+BATCH_SIZE/2,:,:,0] = flip_array_img/127.5 - 1.
    batch_labels = temp[:,1].astype('int')
    batch_labels = np.hstack((batch_labels,batch_labels))  #
    return batch_images,batch_labels-1
def get_batch(idx):
    temp = imglist[idx*BATCH_SIZE:(idx+1)*BATCH_SIZE]
    batch_images = np.zeros((BATCH_SIZE,64,64,C_dim))
    for i in range(temp.shape[0]):
        img = Image.open(image_path+temp[i,0]).convert('L') #grey
        array_img = np.array(img)
        #batch_images[i,:,:,0] = (array_img - array_img.mean())/array_img.std()#Z-score
        batch_images[i,:,:,0] = array_img/127.5 - 1.
    batch_labels = temp[:,1].astype('int')  
    return batch_images,batch_labels-1
#mkdir
def mkdirs(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)
#Logit set
G, enc_id = generator(inputs,z,is_train=True)###
D_id_,D_id,D_real_,D_real = discriminator(inputs,is_train=True)
D_G_id_,D_G_id,D_G_real_,D_G_real = discriminator(G,reuse=True,is_train=True)

d_id_sum = histogram_summary("d_id", D_id_)
d_real_sum = histogram_summary("d_real", D_real_)
d_G_id_sum = histogram_summary("d_g_id", D_G_id_)
d_G_real_sum = histogram_summary("d_g_real", D_G_real_)
enc_id_sum = histogram_summary("enc_id", enc_id)
G_sum = image_summary("G", G)


#pre-train G_enc
enc_loss = tf.reduce_mean(losses.sparse_softmax_cross_entropy(labels=y,logits=enc_id))
enc_loss_sum = scalar_summary("enc_loss", enc_loss)
#Loss function set
if MODE == 'wgan-gp':
    g_loss_gan = -tf.reduce_mean(D_G_real)
    d_loss_gan = tf.reduce_mean(D_G_real) - tf.reduce_mean(D_real)

    alpha = tf.random_uniform(
        shape=[BATCH_SIZE,1,1,1], #/len(DEVICES)
        minval=0.,
        maxval=1.
    )
    
    differences = G - inputs #differences = fake_data - real_data
    interpolates = inputs + (alpha*differences)
    _,_,_,D_p = discriminator(interpolates,reuse=True,is_train=True)
    print(D_p)
    #tf.NoGradient('AvgPoolGrad')
    gradients = tf.gradients(D_p, [interpolates],colocate_gradients_with_ops=True)[0]   #[0]decoded。。。
    print(gradients)
    slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), [1,2,3]))# reduce the dim to [batch_size]
    slopes_sum = histogram_summary("d_slopes", slopes)
    gradient_penalty = tf.reduce_mean((slopes-1.)**2)
    d_loss_gan += LAMBDA*gradient_penalty
else:
    d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_real, labels=tf.ones_like(D_real_)*0.9))#tf.slice(real_smooth,[0],smooth_i)
    d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_G_real, labels=tf.ones_like(D_real_)*0.1))#
    d_loss_gan = d_loss_real + d_loss_fake
    #d_loss_gan = -tf.reduce_mean(tf.log(tf.clip_by_value(D_real_,1e-10,1.0)) + tf.log(tf.clip_by_value(1. - D_G_real_,1e-10,1.0)))#tf.clip_by_value(y_conv,1e-10,1.0)
    g_loss_gan = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_G_real, labels=tf.ones_like(D_G_real_)*0.9))#tf.slice(real_smooth,[0],smooth_i)
    #g_loss_gan = -tf.reduce_mean(tf.log(tf.clip_by_value(D_G_real_,1e-10,1.0)))
    g_pix_loss = tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square(G - inputs), [1,2,3])))#L2(pixel-wise) loss between fake data and real data for Generator
    d_loss_real_flip = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_real, labels=tf.ones_like(D_real_)*0.1))
    d_loss_fake_flip = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_G_real, labels=tf.ones_like(D_real_)*0.9))#
    d_loss_gan_flip = d_loss_real + d_loss_fake
    g_loss_gan_flip = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_G_real, labels=tf.ones_like(D_G_real_)*0.1))

d_loss_id = tf.reduce_mean(losses.sparse_softmax_cross_entropy(labels=y,logits=D_id))
d_loss = LAMBDA_ID*d_loss_id + LAMBDA_GAN*d_loss_gan
d_loss_flip = LAMBDA_ID*d_loss_id + LAMBDA_GAN*d_loss_gan_flip
g_loss_id = tf.reduce_mean(losses.sparse_softmax_cross_entropy(labels=y,logits=D_G_id))
g_loss = LAMBDA_GAN*g_loss_gan + LAMBDA_ID*g_loss_id #+ LAMBDA_PIX*g_pix_loss
g_loss_flip = LAMBDA_GAN*g_loss_gan_flip + LAMBDA_ID*g_loss_id


d_loss_id_sum = scalar_summary("d_loss_id", d_loss_id)
d_loss_gan_sum = scalar_summary("d_loss_gan", d_loss_gan)
d_loss_sum = scalar_summary("d_loss", d_loss)
g_loss_id_sum = scalar_summary("g_loss_id", g_loss_id)
g_loss_gan_sum = scalar_summary("g_loss_gan", g_loss_gan)
g_loss_sum = scalar_summary("g_loss", g_loss)
#Variable set
t_vars = tf.trainable_variables()
d_vars = [var for var in t_vars if 'Discriminator' in var.name]
g_vars = [var for var in t_vars if 'Generator' in var.name]
g_enc_vars = [var for var in t_vars if 'Generator/encoder' in var.name] ###
g_dec_vars = [var for var in t_vars if 'Generator/decoder' in var.name] ###
#show all variables
slim.model_analyzer.analyze_vars(t_vars, print_info=True)

global_step = tf.Variable(0)
#pre-train G_enc
print('pre-training var_list:')
slim.model_analyzer.analyze_vars(g_enc_vars, print_info=True)
#Optimizer set
enc_optim = tf.train.AdamOptimizer(learning_rate=2e-4,beta1=0.5).minimize(enc_loss,var_list=g_enc_vars,global_step=global_step)
dec_optim = tf.train.AdamOptimizer(learning_rate=2e-4,beta1=0.5).minimize(g_loss,var_list=g_dec_vars,global_step=global_step)

if MODE == 'wgan-gp':
    d_optim = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5, beta2=0.9).minimize(d_loss,var_list=d_vars,global_step=global_step)#
    g_optim = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5, beta2=0.9).minimize(g_loss,var_list=g_vars,global_step=global_step)
else:
    d_optim = tf.train.AdamOptimizer(learning_rate=2e-4,beta1=0.5).minimize(d_loss,var_list=d_vars,global_step=global_step)
    g_optim = tf.train.AdamOptimizer(learning_rate=2e-4,beta1=0.5).minimize(g_loss,var_list=g_vars,global_step=global_step)
d_optim_flip = tf.train.AdamOptimizer(learning_rate=1e-4,beta1=0.5).minimize(d_loss_flip,var_list=d_vars,global_step=global_step)
g_optim_flip = tf.train.AdamOptimizer(learning_rate=2e-4,beta1=0.5).minimize(g_loss_flip,var_list=g_vars,global_step=global_step)

#Accuracy ops
enc_id_count = tf.reduce_sum(tf.cast(tf.equal(tf.argmax(enc_id,1), y),'int32'))
d_id_count = tf.reduce_sum(tf.cast(tf.equal(tf.argmax(D_id,1), y),'int32'))
g_id_count = tf.reduce_sum(tf.cast(tf.equal(tf.argmax(D_G_id,1), y),'int32'))
#Train
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1) ###only one GPU
run_config = tf.ConfigProto()
run_config.gpu_options.allow_growth=True
sess=tf.InteractiveSession(config=run_config)  
sess.run(tf.global_variables_initializer())   

g_sum = merge_summary([d_G_id_sum,d_G_real_sum,G_sum,g_loss_id_sum,g_loss_gan_sum, g_loss_sum])#
d_sum = merge_summary([d_id_sum,d_real_sum,d_loss_id_sum,d_loss_gan_sum,d_loss_sum])#,slopes_sum
enc_sum = merge_summary([enc_id_sum,enc_loss_sum])
mkdirs('./ckpt')
mkdirs("./logs")
writer = SummaryWriter("./logs", sess.graph)
saver = tf.train.Saver()#max_to_keep=1
sample_images,sample_labels=get_batch(0)
sample_z = np.random.uniform(-1, 1, [BATCH_SIZE, z_size]).astype(np.float32)
batch_idx=len(imglist) // BATCH_SIZE
batch_idx=batch_idx*2 #with augmentation
#
def pre_train(sess):
    counter = 0
    for i in range(epoch):
        #打乱排序
        seed = i
        np.random.seed(seed)
        np.random.shuffle(imglist)
        j=iter(range(batch_idx))
        try:
            while True:
                batch_images,batch_labels=get_batch_aug(next(j))
                #print batch_labels[31],batch_labels[63]
                batch_z = np.random.uniform(-1, 1, [BATCH_SIZE, z_size]).astype(np.float32)
                _,summary_str,loss,acc = sess.run([enc_optim,enc_sum,enc_loss,enc_id_count],
                        feed_dict={ 
                          inputs: batch_images,
                          y:batch_labels,
                          z:batch_z
                        })
                counter += 1
                writer.add_summary(summary_str, counter)
                if np.mod(counter, 10) == 1:
                    saver.save(sess,'./ckpt/model.ckpt',global_step=counter)
                    print('[Epoch-Batch]%d - %d: enc_loss=%f' % (i,counter,loss))
                    print('Acc.: %d / %d' % (acc,BATCH_SIZE))
        except StopIteration:
            pass

def triangle_train(sess):
    #load model trained before...
    _, counter = load(sess,checkpoint_dir)
    sample_counter = 0 
    if MODE == 'wgan-gp':
        disc_iters = CRITIC_ITERS
        enc_iters = 1
        dec_iters = 1
    else:
        disc_iters = 1
        enc_iters = 1
        dec_iters = 1
    # Train loop
    for i in range(epoch):
        #打乱排序
        seed = i
        np.random.seed(seed)
        np.random.shuffle(imglist)
        j=iter(range(batch_idx))
        try:
            while True:
                iteration = next(j)
                batch_images,batch_labels=get_batch_aug(iteration)
                batch_z = np.random.uniform(-1, 1, [BATCH_SIZE, z_size]).astype(np.float32)
                for s in range(enc_iters):
                    _,summary_str = sess.run([enc_optim,enc_sum],
                                feed_dict={ 
                                  inputs: batch_images,
                                  y:batch_labels,
                                  z:batch_z
                                })
                    counter += 1
                    writer.add_summary(summary_str, counter) 
                for t in range(disc_iters):
                    _,summary_str = sess.run([d_optim,d_sum],
                        feed_dict={
                          inputs: batch_images,
                          y:batch_labels,
                          z:batch_z
                        })
                    counter += 1
                    writer.add_summary(summary_str, counter)     
                for u in range(dec_iters):
                    _,summary_str = sess.run([dec_optim,g_sum],
                                feed_dict={ 
                                  inputs: batch_images,
                                  y:batch_labels,
                                  z:batch_z
                                })
                    counter += 1
                    writer.add_summary(summary_str, counter)            

                loss_pre = [enc_loss,d_loss_id,d_loss_gan,g_loss_id,g_loss_gan,enc_id_count,d_id_count,g_id_count]#,gradient_penalty
                el,dl1,dl2,gl1,gl2,e_id,d_id,g_id = sess.run(loss_pre,           
                        feed_dict={ 
                          inputs: batch_images,
                          y:batch_labels, 
                          z:batch_z
                        })#,gp
                sample_counter += 1
                #lr = sess.run(learning_rate)
                lr = 1e-4
                saver.save(sess,'./ckpt/model.ckpt',global_step=counter)
                print('[Epoch-Batch]%d - %d: Learning Rate=%f' % (i,counter,lr))
                print('enc_Loss. =%f'%(el))
                print('Loss. D:id=%f, gan=%f  G:id=%f, gan=%f' \
                    % (dl1,dl2,gl1,gl2))
                print('Acc. enc id:%d / %d, real id:%d / %d, fake id:%d / %d' % (e_id,BATCH_SIZE,d_id,BATCH_SIZE,g_id,BATCH_SIZE))
                #print(gp)
                if np.mod(sample_counter, 10) == 1:
                    samples= sess.run(G,
                        feed_dict={
                          inputs: sample_images,
                          z:sample_z
                        })
                    manifold_h = int(np.ceil(np.sqrt(samples.shape[0])))
                    manifold_w = int(np.floor(np.sqrt(samples.shape[0])))
                    mkdirs('./sample')
                    save_images(samples, [manifold_h, manifold_w],
                          '{}/train_{:02d}_{:04d}.png'.format('./sample', i, counter))
                    print('samples are saved!')
        except StopIteration:
            pass       
def train(sess):
    #load model trained before...
    _, counter = load(sess,checkpoint_dir)
    sample_counter = 0

    if MODE == 'wgan-gp':
        disc_iters = CRITIC_ITERS
        gen_iters = 1
    else:
        disc_iters = 1
        gen_iters = 2
    # Train loop
    for i in range(epoch):
        #打乱排序
        seed = i
        np.random.seed(seed)
        np.random.shuffle(imglist)
        j=iter(range(batch_idx))
        try:
            while True:
                iteration = next(j)
                batch_images,batch_labels=get_batch_aug(iteration)
                batch_z = np.random.uniform(-1, 1, [BATCH_SIZE, z_size]).astype(np.float32)
                for s in range(gen_iters):
                    if iteration > 0:
                        _,summary_str = sess.run([dec_optim,g_sum],
                                feed_dict={ 
                                  inputs: batch_images,
                                  y:batch_labels,
                                  z:batch_z
                                })
                        counter += 1
                        writer.add_summary(summary_str, counter)            
                for t in range(disc_iters):
                    #batch_images,batch_labels=get_batch_aug(next(j))
                    #batch_z = np.random.uniform(-1, 1, [BATCH_SIZE, z_size]).astype(np.float32)
                    #print(randi[0])
                    _,summary_str = sess.run([d_optim,d_sum],
                        feed_dict={
                          inputs: batch_images,
                          y:batch_labels,
                          z:batch_z
                        })
                    counter += 1
                    writer.add_summary(summary_str, counter)
                loss_pre = [d_loss_id,d_loss_gan,g_loss_id,g_loss_gan,d_id_count,g_id_count]#,gradient_penalty
                dl1,dl2,gl1,gl2,d_id,g_id = sess.run(loss_pre,           
                        feed_dict={ 
                          inputs: batch_images,
                          y:batch_labels, 
                          z:batch_z
                        })#,gp
                sample_counter += 1
                #lr = sess.run(learning_rate)
                lr = 1e-4
                saver.save(sess,'./ckpt/model.ckpt',global_step=counter)
                print('[Epoch-Batch]%d - %d: Learning Rate=%f' % (i,counter,lr))
                print('Loss. D:id=%f, gan=%f  G:id=%f, gan=%f' \
                    % (dl1,dl2,gl1,gl2))
                print('Acc. real id:%d / %d, fake id:%d / %d' % (d_id,BATCH_SIZE,g_id,BATCH_SIZE))
                #print(gp)
                if np.mod(sample_counter, 10) == 1:
                    samples= sess.run(G,
                        feed_dict={
                          inputs: sample_images,
                          z:sample_z
                        })
                    manifold_h = int(np.ceil(np.sqrt(samples.shape[0])))
                    manifold_w = int(np.floor(np.sqrt(samples.shape[0])))
                    mkdirs('./sample')
                    save_images(samples, [manifold_h, manifold_w],
                          '{}/train_{:02d}_{:04d}.png'.format('./sample', i, counter))
                    print('samples are saved!')
        except StopIteration:
            pass
train(sess)#pre_,triangle_train
sess.close()
