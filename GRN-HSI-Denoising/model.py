#!/usr/bin/env python2
# -*- coding: utf-8 -*-


import numpy as np
import tensorflow as tf



def spatialBlock(input_tensor):  
    _, _, _, in_channels = input_tensor.get_shape().as_list()    
   
    channels = in_channels//2


    theta = tf.layers.conv2d(input_tensor, channels, 1, padding="valid")
    theta = tf.reshape(theta, shape=[-1, tf.shape(input_tensor)[1] * tf.shape(input_tensor)[2], channels])

    phi =  tf.layers.conv2d(input_tensor, channels, 1, padding="valid")
    phi =  tf.reshape(phi, shape=[-1, tf.shape(input_tensor)[1] * tf.shape(input_tensor)[2], channels])


    g =  tf.layers.conv2d(input_tensor, channels, 1, padding="valid")
    g =  tf.reshape(g, shape=[-1, tf.shape(input_tensor)[1] * tf.shape(input_tensor)[2], channels])
  


    phi1 = tf.reshape(phi, shape=[-1,  tf.shape(phi)[1] *  tf.shape(phi)[2]])    
    phi1 = tf.nn.softmax(phi1, axis=-1)
    phi1 = tf.reshape(phi1, shape=[-1, tf.shape(phi)[1],  tf.shape(phi)[2]])   


    g1 = tf.reshape(g, shape=[-1,  tf.shape(g)[1] *  tf.shape(g)[2]])    
    g1 = tf.nn.softmax(g1, axis=-1)
    g1 = tf.reshape(g1, shape=[-1, tf.shape(g)[1],  tf.shape(g)[2]])     


    y = tf.matmul(phi1, g1, transpose_a=True)

#    y = y / tf.cast( tf.shape(input_tensor)[1] * tf.shape(input_tensor)[2],  tf.float32)

    
    y = tf.matmul(theta, y)

   
    F_s = tf.reshape(y, shape=[-1,  tf.shape(input_tensor)[1],  tf.shape(input_tensor)[2], channels])

    spatial_out = tf.layers.conv2d(F_s, in_channels, 1, padding='valid')      
      
     
    return spatial_out


    
    

def GloRe(X):
    
    imput_chancel = X.get_shape().as_list()[-1]
    inputs_shape = tf.shape(X)
    
    N =  imput_chancel//4
    C =  imput_chancel//2
    
    
    
    B = tf.layers.conv2d(X, N, 1, padding='valid')  
    B = tf.reshape(B, [inputs_shape[0], -1, N]) # [B, H*W, N]

    x_reduced =tf.layers.conv2d(X, C, 1, padding='valid')  
    x_reduced = tf.reshape(x_reduced, [inputs_shape[0], -1, C]) # [B,  H*W, C]
    x_reduced = tf.transpose(x_reduced, perm=[0, 2, 1])  # [B, C, H*W]

    # [B, C, H * W] * [B, H*W, N] —>#[B, C, N]
    v = tf.matmul(x_reduced, B) # [B, C, N]
    
    tmp = tf.reshape(v, shape=[-1,  N * C])    
    tmp = tf.nn.softmax(tmp, axis=-1)
    v = tf.reshape(tmp, shape=[-1, C,  N]) 
    
    v = tf.expand_dims(v, axis=1)  # [B, 1, C, N]

    def GCN(Vnode, nodeN, mid_chancel):

        net = tf.layers.conv2d(Vnode, N, 1, padding='valid')  # [B, 1, C, N]

        net = Vnode + net  #(I-Ag)V
        net = tf.nn.relu(net)

        net = tf.transpose(net, perm=[0, 3, 1, 2]) # [B, N, 1, C]

        net = tf.layers.conv2d(net, mid_chancel, 1, padding='valid') # [B, N, 1, C]

        return net

    z = GCN(v, N, C) # [B, N, 1, C]
    
#    z = z + tf.transpose(v, perm=[0, 3, 1, 2]) 
    
    z = tf.reshape(z, [inputs_shape[0], N, C])  # [B, N, C]

    # [B, H*W, N] * [B, N, C] => [B, H*W, C]
    y = tf.matmul(B, z) # [B, H*W, C]
    
    
    y = tf.expand_dims(y, axis=1)  #[B, 1, H*W, C]
    y = tf.reshape(y, [inputs_shape[0], inputs_shape[1], inputs_shape[2], C])  # [B, H, W, C]
    x_res = tf.layers.conv2d(y, imput_chancel, 1, padding='valid')        
          
    return  x_res




def blockE(_input): 
    _, _, _, channels = _input.get_shape().as_list()  
    

    input_tensor = _input
 
    conv1 = tf.layers.conv2d(input_tensor, channels, 3, padding="SAME",activation = tf.nn.relu)   
    
#    tmp = tf.concat([input_tensor, conv1],-1) 
    tmp = tf.add(input_tensor,conv1)
    
    conv2 = tf.layers.conv2d(tmp, channels, 3, padding="SAME",activation = tf.nn.relu)   

    tmp = tf.add(tmp,conv2)
#    tmp = tf.concat([input_tensor, conv1, conv2],-1)     
    conv3 = tf.layers.conv2d(tmp, channels, 3, padding="SAME",activation = tf.nn.relu)   

#    tmp = tf.concat([input_tensor, conv1, conv2, conv3],-1) 
    tmp = tf.add(tmp,conv3)      
    fuse = tf.layers.conv2d(tmp, channels, 1, padding="SAME")    
   
    fuse = fuse  + GloRe(fuse)

    return   fuse + _input


def blockD(_input): 
    _, _, _, channels = _input.get_shape().as_list()  
    

    input_tensor = _input +  spatialBlock(_input) 
 
    conv1 = tf.layers.conv2d(input_tensor, channels, 3, padding="SAME",activation = tf.nn.relu)   
    
#    tmp = tf.concat([input_tensor, conv1],-1)   
    tmp = tf.add(input_tensor,conv1)
    conv2 = tf.layers.conv2d(tmp, channels, 3, padding="SAME",activation = tf.nn.relu)   

#    tmp = tf.concat([input_tensor, conv1, conv2],-1) 
    tmp = tf.add(tmp,conv2)    
    conv3 = tf.layers.conv2d(tmp, channels, 3, padding="SAME",activation = tf.nn.relu)   

#    tmp = tf.concat([input_tensor, conv1, conv2, conv3],-1) 
    tmp = tf.add(input_tensor,conv3)      
    fuse = tf.layers.conv2d(tmp, channels, 1, padding="SAME")    


    return   fuse + _input



def Inference(images, channels = 64):
    
  inchannels = images.get_shape().as_list()[-1]  
  
  with tf.variable_scope('UNet', reuse=tf.AUTO_REUSE):       
         
     with tf.variable_scope('basic'):                     
         basic = tf.layers.conv2d(images, channels, 3, padding="SAME")       
         basic1 = tf.layers.conv2d(basic, channels, 3, padding="SAME")             
                 
     with tf.variable_scope('encoder0'):         
         encode0 =  blockE(basic1) 
         donw0 = tf.layers.conv2d(encode0, channels, 3, strides=2, padding="SAME",activation = tf.nn.relu)  
         
         
     with tf.variable_scope('encoder1'):              
         encode1 =  blockE(donw0) 
         donw1 = tf.layers.conv2d(encode1, channels, 3, strides=2, padding="SAME",activation = tf.nn.relu)  
        

               
     with tf.variable_scope('encoder2'): 
         encode2 =  blockE(donw1) 
         donw2 = tf.layers.conv2d(encode2, channels, 3, strides=2, padding="SAME",activation = tf.nn.relu)  
         



     with tf.variable_scope('middle'):         

         media_end =   blockE(donw2)

                                  
     with tf.variable_scope('decoder2'):              
          Deblock2 = tf.image.resize_images(media_end, [tf.shape(encode2)[1], tf.shape(encode2)[2]],method=1)
          Deblock2 = tf.concat([Deblock2, encode2] ,-1)
          Deblock2 = tf.layers.conv2d(Deblock2, channels, 1, padding="SAME",activation = tf.nn.relu) 
          Deblock2 = blockD(Deblock2)
         
         
     with tf.variable_scope('decoder1'):              
          Deblock1 = tf.image.resize_images(Deblock2, [tf.shape(encode1)[1], tf.shape(encode1)[2]],method=1) 
          Deblock1 = tf.concat([Deblock1, encode1] ,-1)         
          Deblock1 = tf.layers.conv2d(Deblock1, channels, 1, padding="SAME",activation = tf.nn.relu) 
          Deblock1 =  blockD(Deblock1)
         
         
     with tf.variable_scope('decoder0'):              
          Deblock0 = tf.image.resize_images(Deblock1, [tf.shape(encode0)[1], tf.shape(encode0)[2]],method=1)   
          Deblock0 = tf.concat([Deblock0, encode0, basic1] ,-1) 
          Deblock0 = tf.layers.conv2d(Deblock0, channels, 1, padding="SAME",activation = tf.nn.relu)                  
          Deblock0 =  blockD(Deblock0)    
 
            
     with tf.variable_scope('reconstruct'):       
         decoding_end = Deblock0 + basic
         res =  tf.layers.conv2d(decoding_end, inchannels, 3, padding = 'SAME')  
         out  = images + res    

  return  out





 
if __name__ == '__main__':
    tf.reset_default_graph()
    input_x = tf.placeholder(tf.float32, [10,101,101,1])
#    out = spatialBlock(input_x)
    out = Inference(input_x)   

    print(input_x.get_shape().as_list())

    
    all_vars = tf.trainable_variables()
    print("Total parameters' number: %d" 
         %(np.sum([np.prod(v.get_shape().as_list()) for v in all_vars])))  