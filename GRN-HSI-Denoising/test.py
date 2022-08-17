# -*- coding: utf-8 -*-
"""
Created on Sat Apr 18 21:15:29 2020

@author: Xiangyong Cao
"""

import scipy.io
import os
import model as model
import tensorflow as tf
import skimage
import matplotlib.pyplot as plt
import matplotlib.image as img
import numpy as np
import h5py
from skimage.measure import compare_ssim
from skimage.measure import compare_psnr
import time
from Indexes import sam, ergas
#import cv2

##################### Select GPU device ###################################
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
###########################################################################

start = time.time()

model_path = './Pretrained_model/case1/'  # can be changed
data_path = './data/datasets/ICVL_test_pair_whole_image/stage1/icvl_50/'
result_path = './result/icvl_test/'


#model_path = './model/CAVE/'
#data_path = './data/CAVE_testing/cave_complex_case5/'
#result_path = './result/cave_complex5/'

height = 512
width  = 512
channels = 31

if __name__ == '__main__':
    
   imgName = os.listdir(data_path)
   filename = os.listdir(data_path)
   filename_ori = os.listdir(data_path)
   for i in range(len(filename)):
      filename[i] = data_path + filename[i]
      
   num_img = len(filename)
   PSNR = np.zeros([num_img,channels])
   SSIM = np.zeros([num_img,channels])
#   ERGAS = np.zeros([num_img])
   SAM = np.zeros([num_img])
   Name = []
    
   image = tf.placeholder(tf.float32, shape=(1, height, width, channels))   
   final = model.Inference(image)
   output = tf.clip_by_value(final, 0., 1.)
#   output = tf.squeeze(out)

   config = tf.ConfigProto()
   config.gpu_options.allow_growth=True
   saver = tf.train.Saver()
   
   with tf.Session(config=config) as sess: 
      with tf.device('/gpu:0'): 
          if tf.train.get_checkpoint_state(model_path):  
              ckpt = tf.train.latest_checkpoint(model_path)  # try your own model 
              saver.restore(sess, ckpt)
              print ("Loading model")
        
          for i in range(num_img):
              
              img_tmp = scipy.io.loadmat(filename[i])   # choose one dataset
              rain = img_tmp['input']
              label = img_tmp['gt'] 
              rain = np.expand_dims(rain,axis=0)
              
              derained = sess.run(output,feed_dict = {image:rain}) 
              
              index = imgName[i].rfind('.')
              name = imgName[i][:index]
              Name.append(name)
              mat_name = name + '_denoised'+'.mat'
              denoised_result = {}
              denoised_result['derained'] = derained[0,:,:,:]
              denoised_result['rain'] = rain[0,:,:,:]
              denoised_result['label'] = label              
              scipy.io.savemat(os.path.join(result_path, mat_name),denoised_result)
              
              for c in range(channels):
                  psnr_ori = compare_psnr(label[:,:,c],rain[0,:,:,c])
                  PSNR[i,c] = compare_psnr(label[:,:,c],derained[0,:,:,c])
                  SSIM[i,c] = compare_ssim(label[:,:,c],derained[0,:,:,c])
                   
#                  print(name + ': %d / %d images processed, psnr_ori=%f, psnr=%f, ssim=%f' 
#                        % (c+1,channels,psnr_ori, PSNR[i,c],SSIM[i,c]))     
              print('The %2d /% d test dataset: %30s: MPSNR = %.4f, MSSIM = %.4f' 
                    % (i+1,num_img, filename_ori[i], np.mean(PSNR[i,:]),np.mean(SSIM[i,:]))) 
#              ERGAS[i] = ergas(label,derained[0,:,:,:])
              SAM[i] = sam(label,derained[0,:,:,:])
          Measure = {}
          Measure['PSNR'] = PSNR
          Measure['SSIM'] = SSIM
#          Measure['ERGAS'] = ERGAS
          Measure['SAM'] = SAM
          Measure['Name'] = Name
          scipy.io.savemat(os.path.join(result_path, 'Measure.mat'),Measure)
          end = time.time()
          print('Testing is finished!')
          print('Mean PSNR = %.4f, Mean SSIM = %.4f, Mean SAM= %.4f, Time = %.4f' 
                % (np.mean(PSNR),np.mean(SSIM),np.mean(SAM),end-start))
