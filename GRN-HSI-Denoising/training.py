#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import os
import h5py
import re
import numpy as np
import tensorflow as tf
import model as model
from scipy import io
import time


##################### select GPU device ####################################
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
############################################################################

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('num_h5_file', 1,
                            """number of training h5 files.""")
tf.app.flags.DEFINE_integer('num_patches', 500,
                            """number of patches in each h5 file.""")
tf.app.flags.DEFINE_float('learning_rate', 0.0005,
                            """learning rate.""")
tf.app.flags.DEFINE_integer('epoch', 10,
                            """epoch.""")
tf.app.flags.DEFINE_integer('batch_size', 32,
                            """Batch size.""")
tf.app.flags.DEFINE_integer('num_channels', 31,
                            """Number of channels of the input.""")
tf.app.flags.DEFINE_integer('image_size', 64,
                            """Size of the images.""")
tf.app.flags.DEFINE_integer('label_size', 64,
                            """Size of the labels.""")
tf.app.flags.DEFINE_string("data_path", "./h5data/ICVL_stage1/", "The path of h5 files")

tf.app.flags.DEFINE_string("save_model_path", "./model/ICVL_stage1/", "The path of saving model")



# read h5 files
def read_data(file):
  with h5py.File(file, 'r') as hf:
    data = hf.get('data')
    label = hf.get('label')
    return np.array(data), np.array(label)

def compute_ssim_loss(pred, gt):
   value = 1.0 - tf.image.ssim(pred,gt, max_val=1.0)
   loss = tf.reduce_mean(value)   
   return loss

def compute_L1_loss(pred, gt):
   value = tf.abs(pred - gt)
   loss = tf.reduce_mean(value)
   return loss  

def compute_L2_loss(pred, gt):
   value = tf.square(pred - gt)
   loss = tf.reduce_mean(value)
   return loss 


def savemodel(save_model_path,training_error):
    return io.savemat(save_model_path + 'training_error.mat', 
                      {'training_error': training_error}) 


if __name__ == '__main__':

  images = tf.placeholder(tf.float32, shape=(None, FLAGS.image_size, FLAGS.image_size, FLAGS.num_channels))  # data
  labels = tf.placeholder(tf.float32, shape=(None, FLAGS.label_size, FLAGS.label_size, FLAGS.num_channels))  # detail layer
  
  outputs = model.Inference(images)      
  loss =   compute_L1_loss(labels, outputs)

  lr_ = FLAGS.learning_rate 
  lr  = tf.placeholder(tf.float32 ,shape = []) 
  
  all_vars = tf.trainable_variables() 
  print("Total parameters' number: %d" 
        %(np.sum([np.prod(v.get_shape().as_list()) for v in all_vars]))) 
  
  g_optim =  tf.train.AdamOptimizer(lr).minimize(loss) # Optimization method: Adam
  
  saver = tf.train.Saver(max_to_keep = 5)
  
  config = tf.ConfigProto()
  config.gpu_options.per_process_gpu_memory_fraction = 0.5 # GPU setting
  config.gpu_options.allow_growth = True
  
  
  data_path = FLAGS.data_path
  save_path = FLAGS.save_model_path 
  epoch = int(FLAGS.epoch) 

  with tf.Session(config=config) as sess:

    sess.run(tf.global_variables_initializer())
    

    validation_data_name = "validation.h5"
    validation_h5_data, validation_h5_label = read_data(data_path + validation_data_name)


    validation_data = validation_h5_data
    validation_data = np.transpose(validation_data, (0,2,3,1))   # image data
    validation_label = np.transpose(validation_h5_label, (0,2,3,1)) # label

    if tf.train.get_checkpoint_state(save_path):   # load previous trained model
      ckpt = tf.train.latest_checkpoint(save_path)
      saver.restore(sess, ckpt)
      ckpt_num = re.findall(r'(\w*[0-9]+)\w*',ckpt)
      start_point = int(ckpt_num[-1])
#      ckpt_num = re.findall(r"\d",ckpt)
#      if len(ckpt_num)==3:
#        start_point = 100*int(ckpt_num[0])+10*int(ckpt_num[1])+int(ckpt_num[2])
#      elif len(ckpt_num)==2:
#        start_point = 10*int(ckpt_num[0])+int(ckpt_num[1])
#      else:
#        start_point = int(ckpt_num[0]) 
        
      training_error = io.loadmat(save_path+'training_error.mat')['training_error']  
      training_error = list(training_error)
      print("Load success")
      print('start_point %d' % (start_point))
   
    else:
      print("re-training")
      start_point = 0 
      training_error = []

    start = time.time()  

    for j in range(start_point,epoch):   # epoch
      l = j - start_point
      Epoch = epoch - start_point
      if l+1 >(3*Epoch/6):  # reduce learning rate
        lr_ = 0.0001   #FLAGS.learning_rate*0.1 #0.1
      if l+1 >(5*Epoch/6):
        lr_ = 0.00001  #FLAGS.learning_rate*0.01 #0.01
#      if l+1 >(3*Epoch/4):
#        lr_ = 0.00001 

      Training_Loss = 0.
  
      for num in range(FLAGS.num_h5_file):    # h5 files
        train_data_name = "train" + str(num+1) + ".h5"
        train_h5_data, train_h5_label = read_data(data_path + train_data_name)

        train_data = np.transpose(train_h5_data, (0,2,3,1))   # image data
        train_label = np.transpose(train_h5_label, (0,2,3,1)) # label


        data_size = int( FLAGS.num_patches / FLAGS.batch_size )  # the number of batch
        for i in range(data_size):
          rand_index = np.arange(int(i*FLAGS.batch_size),int((i+1)*FLAGS.batch_size))   # batch
          batch_data = train_data[rand_index,:,:,:]   
          batch_label = train_label[rand_index,:,:,:]


          _,lossvalue = sess.run([g_optim,loss], feed_dict={images: batch_data, labels: batch_label, lr: lr_})
          Training_Loss += lossvalue  # training loss
  

      Training_Loss /=  (data_size * FLAGS.num_h5_file)
      
      training_error.append(Training_Loss)

      model_name = 'model-epoch'   # save model
      save_path_full = os.path.join(save_path, model_name)
      saver.save(sess, save_path_full, global_step = j+1)
      
      savemodel(save_path,np.array(training_error))

      Validation_Loss  = sess.run(loss,  feed_dict={images: validation_data, labels:validation_label})  # validation loss
      end = time.time()
      print ('%d epoch is finished, learning rate = %.4f, Training_Loss = %.4f, Validation_Loss = %.4f, runtime = %.1f s' 
             % (j+1, lr_, Training_Loss, Validation_Loss, (end-start)))
      start = time.time() 
    print('Training is finished.')
      