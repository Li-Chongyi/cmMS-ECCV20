from utils import ( 
  imsave,
  prepare_data
)

import time
import os
import matplotlib.pyplot as plt
import re
import numpy as np
import tensorflow as tf
import scipy.io as scio
from ops import *

class T_CNN(object):

  def __init__(self,
               sess,
               image_height=224,
               image_width=224,
               label_height=224,
               label_width=224,
               batch_size=1,
               c_dim=3, 
               c_depth_dim=1,
               checkpoint_dir=None,
               ):
    self.sess = sess
    self.is_grayscale = (c_dim == 1)
    self.image_height = image_height
    self.image_width = image_width
    self.label_height = label_height
    self.label_width = label_width
    self.batch_size = batch_size
    self.dropout_keep_prob=0.5
    self.c_dim = c_dim
    self.df_dim = 64
    self.checkpoint_dir = checkpoint_dir
    self.c_depth_dim=c_depth_dim
    self.vgg_dir='vgg_pretrained/imagenet-vgg-verydeep-16.mat'
    data = scipy.io.loadmat(self.vgg_dir)
    self.weights = data['layers'][0]
    self.build_model()

  def build_model(self):
    self.images = tf.placeholder(tf.float32, [self.batch_size, self.image_height, self.image_width, self.c_dim], name='images')
    self.depth = tf.placeholder(tf.float32, [self.batch_size, self.image_height, self.image_width,self.c_depth_dim], name='depth')
    self.pred_h = self.model()
    self.saver = tf.train.Saver()
     
  def train(self, image_test, depth_test, test_image_name, config):


    # Stochastic gradient descent with the standard backpropagation,var_list=self.model_c_vars
    shape = image_test.shape
    expand_test = image_test[np.newaxis,:,:,:]
    expand_zero = np.zeros([self.batch_size-1,shape[0],shape[1],shape[2]])
    batch_test_image = np.append(expand_test,expand_zero,axis = 0)
    shape = image_test.shape
    shape1 = depth_test.shape
    expand_test1 = depth_test[np.newaxis,:,:]
    expand_zero1 = np.zeros([self.batch_size-1,shape1[0],shape1[1]])
    batch_test_depth1 = np.append(expand_test1,expand_zero1,axis = 0)
    batch_test_depth= batch_test_depth1.reshape(self.batch_size,shape1[0],shape1[1],1)
    counter = 0
    start_time = time.time()
    result_h  = self.sess.run(self.pred_h, feed_dict={self.images: batch_test_image, self.depth: batch_test_depth})

    _,h ,w , c = result_h.shape
    for id in range(0,1):
        result_h01 = result_h[id,:,:,:].reshape(h , w , 1)
        result_h0 = result_h01.squeeze()
        image_path0 = os.path.join(os.getcwd(), config.sample_dir)
        image_path = os.path.join(image_path0, test_image_name+'_out.png')
        imsave_lable(result_h0, image_path)



  def model(self):
    with tf.variable_scope("main") as scope:

## rgb_VGG
      parameter_mean=tf.constant([0.4076,0.4578,0.4850],name='parameter')

      image_input=tf.subtract(self.images, parameter_mean)

      conv1_vgg_224 = tf.nn.relu(conv2d_vgg(image_input, self.weights[0][0][0][2][0],name="conv2d_vgg_1"))
      conv1_vgg_224_2_temp = tf.nn.relu(conv2d_vgg(conv1_vgg_224, self.weights[2][0][0][2][0],name="conv1_vgg_224_2_temp"))
      conv1_vgg_224_2=tf.nn.relu(conv2d(conv1_vgg_224_2_temp, 64,32,k_h=1, k_w=1, d_h=1, d_w=1,name="conv1_vgg_224_2"))
      conv1_vgg_112 = max_pool_2x2(conv1_vgg_224_2_temp)

      conv2_vgg_112 = tf.nn.relu(conv2d_vgg(conv1_vgg_112, self.weights[5][0][0][2][0],name="conv2d_vgg_3"))
      conv2_vgg_112_2_temp = tf.nn.relu(conv2d_vgg(conv2_vgg_112, self.weights[7][0][0][2][0],name="conv2_vgg_112_2_temp"))
      conv2_vgg_112_2=tf.nn.relu(conv2d(conv2_vgg_112_2_temp, 128,64,k_h=1, k_w=1, d_h=1, d_w=1,name="conv2_vgg_112_2"))
      conv1_vgg_56 = max_pool_2x2(conv2_vgg_112_2_temp)
      
      conv2_vgg_56 = tf.nn.relu(conv2d_vgg(conv1_vgg_56, self.weights[10][0][0][2][0],name="conv2d_vgg_5"))
      conv3_vgg_56 = tf.nn.relu(conv2d_vgg(conv2_vgg_56, self.weights[12][0][0][2][0],name="conv2d_vgg_6"))
      conv3_vgg_56_2_temp = tf.nn.relu(conv2d_vgg(conv3_vgg_56, self.weights[14][0][0][2][0],name="conv3_vgg_56_2_temp"))
      conv3_vgg_56_2=tf.nn.relu(conv2d(conv3_vgg_56_2_temp, 256,128,k_h=1, k_w=1, d_h=1, d_w=1,name="conv3_vgg_56_2"))
      conv1_vgg_28 = max_pool_2x2(conv3_vgg_56_2_temp)

      conv2_vgg_28 = tf.nn.relu(conv2d_vgg(conv1_vgg_28, self.weights[17][0][0][2][0],name="conv2d_vgg_8"))
      conv3_vgg_28 = tf.nn.relu(conv2d_vgg(conv2_vgg_28, self.weights[19][0][0][2][0],name="conv2d_vgg_9"))
      conv3_vgg_28_2_temp = tf.nn.relu(conv2d_vgg(conv3_vgg_28, self.weights[21][0][0][2][0],name="conv3_vgg_28_2_temp"))
      conv3_vgg_28_2=tf.nn.relu(conv2d(conv3_vgg_28_2_temp, 512,256,k_h=1, k_w=1, d_h=1, d_w=1,name="conv3_vgg_28_2"))
      conv1_vgg_14 = max_pool_2x2(conv3_vgg_28_2_temp)
      
      conv2_vgg_14 = tf.nn.relu(conv2d_vgg(conv1_vgg_14, self.weights[24][0][0][2][0],name="conv2d_vgg_11"))
      conv3_vgg_14 = tf.nn.relu(conv2d_vgg(conv2_vgg_14, self.weights[26][0][0][2][0],name="conv2d_vgg_12"))
      conv4_vgg_14_temp = tf.nn.relu(conv2d_vgg(conv3_vgg_14, self.weights[28][0][0][2][0],name="conv4_vgg_14_temp"))
      conv4_vgg_14=tf.nn.relu(conv2d(conv4_vgg_14_temp, 512,256,k_h=1, k_w=1, d_h=1, d_w=1,name="conv4_vgg_14"))
## depth_VGG
#    
      depth_input_cont= tf.concat(axis = 3, values = [self.depth,self.depth,self.depth]) #
      depth_input=tf.subtract(depth_input_cont, parameter_mean)

      conv1_vgg_depth_224 = tf.nn.relu(conv2d_vgg(depth_input, self.weights[0][0][0][2][0],name="conv1_vgg_depth_224"))
      conv1_vgg_depth_224_2_temp = tf.nn.relu(conv2d_vgg(conv1_vgg_depth_224, self.weights[2][0][0][2][0],name="conv1_vgg_depth_224_2_temp"))
      conv1_vgg_depth_224_2=tf.nn.relu(conv2d(conv1_vgg_depth_224_2_temp, 64,32,k_h=1, k_w=1, d_h=1, d_w=1,name="conv1_vgg_depth_224_2"))
      conv1_vgg_depth_112 = max_pool_2x2(conv1_vgg_depth_224_2_temp)

      conv2_vgg_depth_112 = tf.nn.relu(conv2d_vgg(conv1_vgg_depth_112, self.weights[5][0][0][2][0],name="conv2_vgg_depth_112"))
      conv2_vgg_depth_112_2_temp = tf.nn.relu(conv2d_vgg(conv2_vgg_depth_112, self.weights[7][0][0][2][0],name="conv2_vgg_depth_112_2_temp"))
      conv2_vgg_depth_112_2=tf.nn.relu(conv2d(conv2_vgg_depth_112_2_temp, 128,64,k_h=1, k_w=1, d_h=1, d_w=1,name="conv2_vgg_depth_112_2"))
      conv1_vgg_depth_56 = max_pool_2x2(conv2_vgg_depth_112_2_temp)
      
      conv2_vgg_depth_56 = tf.nn.relu(conv2d_vgg(conv1_vgg_depth_56, self.weights[10][0][0][2][0],name="conv2_vgg_depth_56"))
      conv3_vgg_depth_56 = tf.nn.relu(conv2d_vgg(conv2_vgg_depth_56, self.weights[12][0][0][2][0],name="conv3_vgg_depth_56"))
      conv3_vgg_depth_56_2_temp = tf.nn.relu(conv2d_vgg(conv3_vgg_depth_56, self.weights[14][0][0][2][0],name="conv3_vgg_depth_56_2_temp"))
      conv3_vgg_depth_56_2=tf.nn.relu(conv2d(conv3_vgg_depth_56_2_temp, 256,128,k_h=1, k_w=1, d_h=1, d_w=1,name="conv3_vgg_depth_56_2"))
      conv1_vgg_depth_28 = max_pool_2x2(conv3_vgg_depth_56_2_temp)

      conv2_vgg_depth_28 = tf.nn.relu(conv2d_vgg(conv1_vgg_depth_28, self.weights[17][0][0][2][0],name="conv2_vgg_depth_28"))
      conv3_vgg_depth_28 = tf.nn.relu(conv2d_vgg(conv2_vgg_depth_28, self.weights[19][0][0][2][0],name="conv3_vgg_depth_28"))
      conv3_vgg_depth_28_2_temp = tf.nn.relu(conv2d_vgg(conv3_vgg_depth_28, self.weights[21][0][0][2][0],name="conv3_vgg_depth_28_2_temp"))
      conv3_vgg_depth_28_2=tf.nn.relu(conv2d(conv3_vgg_depth_28_2_temp, 512,256,k_h=1, k_w=1, d_h=1, d_w=1,name="conv3_vgg_depth_28_2"))
      conv1_vgg_depth_14 = max_pool_2x2(conv3_vgg_depth_28_2_temp)
      
      conv2_vgg_depth_14 = tf.nn.relu(conv2d_vgg(conv1_vgg_depth_14, self.weights[24][0][0][2][0],name="conv2_vgg_depth_14"))
      conv3_vgg_depth_14 = tf.nn.relu(conv2d_vgg(conv2_vgg_depth_14, self.weights[26][0][0][2][0],name="conv3_vgg_depth_14"))
      conv4_vgg_depth_14_temp = tf.nn.relu(conv2d_vgg(conv3_vgg_depth_14, self.weights[28][0][0][2][0],name="conv4_vgg_depth_14_temp"))
      conv4_vgg_depth_14=tf.nn.relu(conv2d(conv4_vgg_depth_14_temp, 512,256,k_h=1, k_w=1, d_h=1, d_w=1,name="conv4_vgg_depth_14"))
## RGBD
#224  
      conv1_vgg_rgbd_224a=tf.nn.relu(conv2d(conv1_vgg_depth_224_2, 32,32,k_h=7, k_w=7, d_h=1, d_w=1,name="conv1_vgg_rgbd_224a"))
      conv1_vgg_rgbd_224_a2=tf.nn.relu(conv2d(conv1_vgg_rgbd_224a, 32,32,k_h=5, k_w=5, d_h=1, d_w=1,name="conv1_vgg_rgbd_224_a2"))
      conv1_vgg_rgbd_224_a3=tf.nn.relu(conv2d(conv1_vgg_rgbd_224_a2, 32,32,k_h=3, k_w=3, d_h=1, d_w=1,name="conv1_vgg_rgbd_224_a3"))

      enhance_a224=conv2d(conv1_vgg_rgbd_224_a3, 32,32,k_h=3, k_w=3, d_h=1, d_w=1,name="enhance_a224")

      conv1_vgg_rgbd_224b=tf.nn.relu(conv2d(conv1_vgg_depth_224_2, 32,32,k_h=7, k_w=7, d_h=1, d_w=1,name="conv1_vgg_rgbd_224b"))
      conv1_vgg_rgbd_224_b2=tf.nn.relu(conv2d(conv1_vgg_rgbd_224b, 32,32,k_h=5, k_w=5, d_h=1, d_w=1,name="conv1_vgg_rgbd_224_b2"))
      conv1_vgg_rgbd_224_b3=tf.nn.relu(conv2d(conv1_vgg_rgbd_224_b2, 32,32,k_h=3, k_w=3, d_h=1, d_w=1,name="conv1_vgg_rgbd_224_b3"))

      enhance_b224=conv2d(conv1_vgg_rgbd_224_b3, 32,32,k_h=3, k_w=3, d_h=1, d_w=1,name="enhance_b224")

      conv1_vgg_rgbd_224_2=conv1_vgg_224_2*enhance_a224+enhance_b224
##############################################################################################
#112
      conv1_vgg_rgbd_112a=tf.nn.relu(conv2d(conv2_vgg_depth_112_2, 64,64,k_h=7, k_w=7, d_h=1, d_w=1,name="conv1_vgg_rgbd_112a"))
      conv1_vgg_rgbd_112_a2=tf.nn.relu(conv2d(conv1_vgg_rgbd_112a, 64,64,k_h=5, k_w=5, d_h=1, d_w=1,name="conv1_vgg_rgbd_112_a2"))
      conv1_vgg_rgbd_112_a3=tf.nn.relu(conv2d(conv1_vgg_rgbd_112_a2, 64,64,k_h=3, k_w=3, d_h=1, d_w=1,name="conv1_vgg_rgbd_112_a3"))

      enhance_a112=conv2d(conv1_vgg_rgbd_112_a3, 64,64,k_h=3, k_w=3, d_h=1, d_w=1,name="enhance_a112")

      conv1_vgg_rgbd_112b=tf.nn.relu(conv2d(conv2_vgg_depth_112_2, 64,64,k_h=7, k_w=7, d_h=1, d_w=1,name="conv1_vgg_rgbd_112b"))
      conv1_vgg_rgbd_112_b2=tf.nn.relu(conv2d(conv1_vgg_rgbd_112b, 64,64,k_h=5, k_w=5, d_h=1, d_w=1,name="conv1_vgg_rgbd_112_b2"))
      conv1_vgg_rgbd_112_b3=tf.nn.relu(conv2d(conv1_vgg_rgbd_112_b2, 64,64,k_h=3, k_w=3, d_h=1, d_w=1,name="conv1_vgg_rgbd_112_b3"))

      enhance_b112=conv2d(conv1_vgg_rgbd_112_b3, 64,64,k_h=3, k_w=3, d_h=1, d_w=1,name="enhance_b112")

      conv2_vgg_rgbd_112_2=conv2_vgg_112_2*enhance_a112+enhance_b112

##############################################################################################
#56
      conv1_vgg_rgbd_56a=tf.nn.relu(conv2d(conv3_vgg_depth_56_2, 128,128,k_h=7, k_w=7, d_h=1, d_w=1,name="conv1_vgg_rgbd_56a"))
      conv1_vgg_rgbd_56_a2=tf.nn.relu(conv2d(conv1_vgg_rgbd_56a, 128,128,k_h=5, k_w=5, d_h=1, d_w=1,name="conv1_vgg_rgbd_56_a2"))
      conv1_vgg_rgbd_56_a3=tf.nn.relu(conv2d(conv1_vgg_rgbd_56_a2, 128,128,k_h=3, k_w=3, d_h=1, d_w=1,name="conv1_vgg_rgbd_56_a3"))

      enhance_a56=conv2d(conv1_vgg_rgbd_56_a3, 128,128,k_h=3, k_w=3, d_h=1, d_w=1,name="enhance_a56")

      conv1_vgg_rgbd_56b=tf.nn.relu(conv2d(conv3_vgg_depth_56_2, 128,128,k_h=7, k_w=7, d_h=1, d_w=1,name="conv1_vgg_rgbd_56b"))
      conv1_vgg_rgbd_56_b2=tf.nn.relu(conv2d(conv1_vgg_rgbd_56b, 128,128,k_h=5, k_w=5, d_h=1, d_w=1,name="conv1_vgg_rgbd_56_b2"))
      conv1_vgg_rgbd_56_b3=tf.nn.relu(conv2d(conv1_vgg_rgbd_56_b2, 128,128,k_h=3, k_w=3, d_h=1, d_w=1,name="conv1_vgg_rgbd_56_b3"))

      enhance_b56=conv2d(conv1_vgg_rgbd_56_b3, 128,128,k_h=3, k_w=3, d_h=1, d_w=1,name="enhance_b56")

      conv3_vgg_rgbd_56_2=conv3_vgg_56_2*enhance_a56+enhance_b56

##############################################################################################
#28
      conv1_vgg_rgbd_28a=tf.nn.relu(conv2d(conv3_vgg_depth_28_2, 256,256,k_h=7, k_w=7, d_h=1, d_w=1,name="conv1_vgg_rgbd_28a"))
      conv1_vgg_rgbd_28_a2=tf.nn.relu(conv2d(conv1_vgg_rgbd_28a, 256,256,k_h=5, k_w=5, d_h=1, d_w=1,name="conv1_vgg_rgbd_28_a2"))
      conv1_vgg_rgbd_28_a3=tf.nn.relu(conv2d(conv1_vgg_rgbd_28_a2, 256,256,k_h=3, k_w=3, d_h=1, d_w=1,name="conv1_vgg_rgbd_28_a3"))

      enhance_a28=conv2d(conv1_vgg_rgbd_28_a3, 256,256,k_h=3, k_w=3, d_h=1, d_w=1,name="enhance_a28")

      conv1_vgg_rgbd_28b=tf.nn.relu(conv2d(conv3_vgg_depth_28_2, 256,256,k_h=7, k_w=7, d_h=1, d_w=1,name="conv1_vgg_rgbd_28b"))
      conv1_vgg_rgbd_28_b2=tf.nn.relu(conv2d(conv1_vgg_rgbd_28b, 256,256,k_h=5, k_w=5, d_h=1, d_w=1,name="conv1_vgg_rgbd_28_b2"))
      conv1_vgg_rgbd_28_b3=tf.nn.relu(conv2d(conv1_vgg_rgbd_28_b2, 256,256,k_h=3, k_w=3, d_h=1, d_w=1,name="conv1_vgg_rgbd_28_b3"))

      enhance_b28=conv2d(conv1_vgg_rgbd_28_b3, 256,256,k_h=3, k_w=3, d_h=1, d_w=1,name="enhance_b28")

      conv3_vgg_rgbd_28_2=conv3_vgg_28_2*enhance_a28+enhance_b28

##############################################################################################
#14
      conv1_vgg_rgbd_14a=tf.nn.relu(conv2d(conv4_vgg_depth_14, 256,256,k_h=7, k_w=7, d_h=1, d_w=1,name="conv1_vgg_rgbd_14a"))
      conv1_vgg_rgbd_14_a2=tf.nn.relu(conv2d(conv1_vgg_rgbd_14a, 256,256,k_h=5, k_w=5, d_h=1, d_w=1,name="conv1_vgg_rgbd_14_a2"))
      conv1_vgg_rgbd_14_a3=tf.nn.relu(conv2d(conv1_vgg_rgbd_14_a2, 256,256,k_h=3, k_w=3, d_h=1, d_w=1,name="conv1_vgg_rgbd_14_a3"))

      enhance_a14=conv2d(conv1_vgg_rgbd_14_a3, 256,256,k_h=3, k_w=3, d_h=1, d_w=1,name="enhance_a14")

      conv1_vgg_rgbd_14b=tf.nn.relu(conv2d(conv4_vgg_depth_14, 256,256,k_h=7, k_w=7, d_h=1, d_w=1,name="conv1_vgg_rgbd_14b"))
      conv1_vgg_rgbd_14_b2=tf.nn.relu(conv2d(conv1_vgg_rgbd_14b, 256,256,k_h=5, k_w=5, d_h=1, d_w=1,name="conv1_vgg_rgbd_14_b2"))
      conv1_vgg_rgbd_14_b3=tf.nn.relu(conv2d(conv1_vgg_rgbd_14_b2, 256,256,k_h=3, k_w=3, d_h=1, d_w=1,name="conv1_vgg_rgbd_14_b3"))

      enhance_b14=conv2d(conv1_vgg_rgbd_14_b3, 256,256,k_h=3, k_w=3, d_h=1, d_w=1,name="enhance_b14")

      conv4_vgg_rgbd_14=conv4_vgg_14*enhance_a14+enhance_b14
###############################################################################
# 14
## channle-on-channel-attention
      conc1_gate1 = tf.concat(axis = 3, values = [conv4_vgg_14,conv4_vgg_depth_14,conv4_vgg_rgbd_14]) #
      conv1_gate1=tf.nn.relu(conv2d(conc1_gate1, 256*3,256,k_h=3, k_w=3, d_h=1, d_w=1,name="conv1_gate1"))
      conv2_gate1=tf.nn.relu(conv2d(conv1_gate1, 256,256,k_h=3, k_w=3, d_h=1, d_w=1,name="conv2_gate1"))
      conv3_gate1=tf.nn.relu(conv2d(conv2_gate1, 256,256,k_h=3, k_w=3, d_h=1, d_w=1,name="conv3_gate1"))
      conv4_gate1=tf.nn.relu(conv2d(conv3_gate1, 256,256,k_h=3, k_w=3, d_h=1, d_w=1,name="conv4_gate1"))
      conv5_gate1=tf.nn.relu(conv2d(conv4_gate1, 256,256,k_h=3, k_w=3, d_h=1, d_w=1,name="conv5_gate1"))
      weights_gate1 =tf.nn.sigmoid(conv2d(conv5_gate1, 256,256*3, k_h=3, k_w=3, d_h=1, d_w=1,name="weights_gate1"))
      weight1_rgb=weights_gate1[:,:,:,0:256]  
      weight1_depth=weights_gate1[:,:,:,256:512]
      weight1_rgbd=weights_gate1[:,:,:,512:768]
      gata_output14=tf.add(tf.add(tf.multiply(weight1_rgb,conv4_vgg_14),tf.multiply(weight1_depth,conv4_vgg_depth_14)),tf.multiply(weight1_rgbd,conv4_vgg_rgbd_14))
      gata_output14_conv = tf.nn.relu(conv2d(gata_output14, 256,256,k_h=3, k_w=3, d_h=1, d_w=1,name="gata_output14_conv")) 
      channle_atten_rgb_14 = self.Squeeze_excitation_layer(conv4_vgg_14, out_dim=256, ratio=16, layer_name='channle_atten_rgb_14')
      channle_atten_rgb_14_conv = tf.nn.relu(conv2d(channle_atten_rgb_14, 256,256/2,k_h=1, k_w=1, d_h=1, d_w=1,name="channle_atten_rgb_14_conv"))
      channle_atten_depth_14 = self.Squeeze_excitation_layer(conv4_vgg_depth_14, out_dim=256, ratio=16, layer_name='channle_atten_depth_14')
      channle_atten_depth_14_conv = tf.nn.relu(conv2d(channle_atten_depth_14, 256,256/2,k_h=1, k_w=1, d_h=1, d_w=1,name="channle_atten_depth_14_conv"))      
      channle_atten_rgbd_14 = self.Squeeze_excitation_layer(conv4_vgg_rgbd_14, out_dim=256, ratio=16, layer_name='channle_atten_rgbd_14')
      channle_atten_rgbd_14_conv = tf.nn.relu(conv2d(channle_atten_rgbd_14, 256,256/2,k_h=1, k_w=1, d_h=1, d_w=1,name="channle_atten_rgbd_14_conv")) 
      conc_1 = tf.concat(axis = 3, values = [channle_atten_rgb_14_conv,channle_atten_depth_14_conv,channle_atten_rgbd_14_conv]) #
      channle_atten_2_14_temp = self.Squeeze_excitation_layer(conc_1, out_dim=384, ratio=16, layer_name='channle_atten_2_14')
      channle_atten_2_14_conv = tf.nn.relu(conv2d(channle_atten_2_14_temp, 384,256,k_h=3, k_w=3, d_h=1, d_w=1,name="channle_atten_2_14_conv"))
      channle_atten_2_14 = tf.concat(axis = 3, values = [gata_output14_conv,channle_atten_2_14_conv])
# edge-attention
      conv1_edg_14 = tf.nn.relu(conv2d(tf.concat(axis = 3, values = [conv4_vgg_14,conv4_vgg_depth_14,conv4_vgg_rgbd_14]), 256,256,k_h=3, k_w=3, d_h=1, d_w=1,name="conv1_edg_14"))
      conv2_edg_14 = tf.nn.relu(conv2d(conv1_edg_14, 256,256,k_h=3, k_w=3, d_h=1, d_w=1,name="conv2_edg_14"))
      edge_14 = conv2d(conv2_edg_14, 256,1,k_h=3, k_w=3, d_h=1, d_w=1,name="edge_14")
      conc_1_refine=tf.add(channle_atten_2_14,tf.multiply(channle_atten_2_14, tf.sigmoid(edge_14)))
## saliency 14
      conc_1_refine_f =tf.nn.relu(conv2d(conc_1_refine, 256,256,k_h=3, k_w=3, d_h=1, d_w=1,name="conc_1_refine_f"))
      saliency1_14 = tf.nn.relu(conv2d(conc_1_refine_f, 256,256,k_h=3, k_w=3, d_h=1, d_w=1,name="saliency1_14"))
      saliency_14 = conv2d(saliency1_14, 256,1,k_h=3, k_w=3, d_h=1, d_w=1,name="saliency_14")
      saliency_14_2up=tf.image.resize_bilinear(saliency_14,[28,28])
      conc_1_refine_f_2up_1 =tf.image.resize_bilinear(conc_1_refine,[28,28])
      conc_1_refine_f_2up_2=tf.nn.relu(conv2d(conc_1_refine_f_2up_1, 256,256,k_h=3, k_w=3, d_h=1, d_w=1,name="conc_1_refine_f_2up_2"))
      conc_1_refine_f_2up=tf.nn.relu(conv2d(conc_1_refine_f_2up_2, 256,256,k_h=3, k_w=3, d_h=1, d_w=1,name="conc_1_refine_f_2up"))
# 28
# 
## channle-on-channel-attention
#
      conc1_gate2 = tf.concat(axis = 3, values = [conv3_vgg_28_2,conv3_vgg_depth_28_2,conv3_vgg_rgbd_28_2,conc_1_refine_f_2up]) #
      conv1_gate2=tf.nn.relu(conv2d(conc1_gate2, 256*3,256,k_h=3, k_w=3, d_h=1, d_w=1,name="conv1_gate2"))
      conv2_gate2=tf.nn.relu(conv2d(conv1_gate2, 256,256,k_h=3, k_w=3, d_h=1, d_w=1,name="conv2_gate2"))
      conv3_gate2=tf.nn.relu(conv2d(conv2_gate2, 256,256,k_h=3, k_w=3, d_h=1, d_w=1,name="conv3_gate2"))
      conv4_gate2=tf.nn.relu(conv2d(conv3_gate2, 256,256,k_h=3, k_w=3, d_h=1, d_w=1,name="conv4_gate2"))
      conv5_gate2=tf.nn.relu(conv2d(conv4_gate2, 256,256,k_h=3, k_w=3, d_h=1, d_w=1,name="conv5_gate2"))
      weights_gate2 =tf.nn.sigmoid(conv2d(conv5_gate2, 256,256*4, k_h=3, k_w=3, d_h=1, d_w=1,name="weights_gate2"))
      weight2_rgb=weights_gate2[:,:,:,0:256]
      weight2_depth=weights_gate2[:,:,:,256:512]
      weight2_rgbd=weights_gate2[:,:,:,512:768]
      weight2_feature=weights_gate2[:,:,:,768:1024]
      gata_output28=tf.add(tf.add(tf.add(tf.multiply(weight2_rgb,conv3_vgg_28_2),tf.multiply(weight2_depth,conv3_vgg_depth_28_2)),tf.multiply(weight2_rgbd,conv3_vgg_rgbd_28_2)),tf.multiply(weight2_feature,conc_1_refine_f_2up))
      gata_output28_conv = tf.nn.relu(conv2d(gata_output28, 256,256,k_h=3, k_w=3, d_h=1, d_w=1,name="gata_output28_conv"))
      channle_atten_rgb_28 = self.Squeeze_excitation_layer(conv3_vgg_28_2, out_dim=256, ratio=16, layer_name='channle_atten_rgb_28')
      channle_atten_rgb_28_conv = tf.nn.relu(conv2d(channle_atten_rgb_28, 256,256/2,k_h=1, k_w=1, d_h=1, d_w=1,name="channle_atten_rgb_28_conv"))
      channle_atten_depth_28 = self.Squeeze_excitation_layer(conv3_vgg_depth_28_2, out_dim=256, ratio=16, layer_name='channle_atten_depth_28')
      channle_atten_depth_28_conv = tf.nn.relu(conv2d(channle_atten_depth_28, 256,256/2,k_h=1, k_w=1, d_h=1, d_w=1,name="channle_atten_depth_28_conv"))
      channle_atten_rgbd_28 = self.Squeeze_excitation_layer(conv3_vgg_rgbd_28_2, out_dim=256, ratio=16, layer_name='channle_atten_rgbd_28')
      channle_atten_rgbd_28_conv = tf.nn.relu(conv2d(channle_atten_rgbd_28, 256,256/2,k_h=1, k_w=1, d_h=1, d_w=1,name="channle_atten_rgbd_28_conv"))
      channle_atten_14_28 = self.Squeeze_excitation_layer(conc_1_refine_f_2up, out_dim=256, ratio=16, layer_name='channle_atten_14_28')
      channle_atten_14_28_conv = tf.nn.relu(conv2d(channle_atten_14_28, 256,256/2,k_h=1, k_w=1, d_h=1, d_w=1,name="channle_atten_14_28_conv"))
      conc_2 = tf.concat(axis = 3, values = [channle_atten_rgb_28_conv,channle_atten_depth_28_conv,channle_atten_rgbd_28_conv,channle_atten_14_28_conv]) #
      channle_atten_2_28_temp = self.Squeeze_excitation_layer(conc_2, out_dim=512, ratio=16, layer_name='channle_atten_2_28')
      channle_atten_2_28_temp_conv = tf.nn.relu(conv2d(channle_atten_2_28_temp, 512,256,k_h=3, k_w=3, d_h=1, d_w=1,name="channle_atten_2_28_temp_conv"))
      channle_atten_2_28 = tf.concat(axis = 3, values = [gata_output28_conv,channle_atten_2_28_temp_conv])
# spacial-attention
      conc_2_atten=tf.add(channle_atten_2_28,tf.multiply(channle_atten_2_28, tf.sigmoid(saliency_14_2up)))
# edge-attention
      conv1_edg_28 = tf.nn.relu(conv2d(tf.concat(axis = 3, values = [conv3_vgg_28_2,conv3_vgg_depth_28_2,conv3_vgg_rgbd_28_2,conc_1_refine_f_2up]), 256,256,k_h=3, k_w=3, d_h=1, d_w=1,name="conv1_edg_28"))
      conv2_edg_28 = tf.nn.relu(conv2d(conv1_edg_28, 256,256,k_h=3, k_w=3, d_h=1, d_w=1,name="conv2_edg_28"))
      edge_28 = conv2d(conv2_edg_28, 256,1,k_h=3, k_w=3, d_h=1, d_w=1,name="edge_28")
      conc_2_refine=tf.add(conc_2_atten, tf.multiply(conc_2_atten,tf.sigmoid(edge_28)))
## saliency 28
      conc_2_refine_f =tf.nn.relu(conv2d(conc_2_refine, 256,256,k_h=3, k_w=3, d_h=1, d_w=1,name="conc_2_refine_f"))
      saliency1_28 = tf.nn.relu(conv2d(conc_2_refine_f, 256,256,k_h=3, k_w=3, d_h=1, d_w=1,name="saliency1_28"))
      saliency_28 = conv2d(saliency1_28, 256,1,k_h=3, k_w=3, d_h=1, d_w=1,name="saliency_28")
      saliency_28_2up=tf.image.resize_bilinear(saliency_28,[56,56])
      conc_2_refine_f_2up_1 =tf.image.resize_bilinear(conc_2_refine,[56,56])
      conc_2_refine_f_2up_2=tf.nn.relu(conv2d(conc_2_refine_f_2up_1, 256,256,k_h=3, k_w=3, d_h=1, d_w=1,name="conc_2_refine_f_2up_2"))
      conc_2_refine_f_2up=tf.nn.relu(conv2d(conc_2_refine_f_2up_2, 256,128,k_h=3, k_w=3, d_h=1, d_w=1,name="conc_2_refine_f_2up"))
# 56
## channle-on-channel-attention
#
      conc1_gate3 = tf.concat(axis = 3, values = [conv3_vgg_56_2,conv3_vgg_depth_56_2,conv3_vgg_rgbd_56_2,conc_2_refine_f_2up]) #
      conv1_gate3=tf.nn.relu(conv2d(conc1_gate3, 128*3,128,k_h=3, k_w=3, d_h=1, d_w=1,name="conv1_gate3"))
      conv2_gate3=tf.nn.relu(conv2d(conv1_gate3, 128,128,k_h=3, k_w=3, d_h=1, d_w=1,name="conv2_gate3"))
      conv3_gate3=tf.nn.relu(conv2d(conv2_gate3, 128,128,k_h=3, k_w=3, d_h=1, d_w=1,name="conv3_gate3"))
      conv4_gate3=tf.nn.relu(conv2d(conv3_gate3, 128,128,k_h=3, k_w=3, d_h=1, d_w=1,name="conv4_gate3"))
      conv5_gate3=tf.nn.relu(conv2d(conv4_gate3, 128,128,k_h=3, k_w=3, d_h=1, d_w=1,name="conv5_gate3"))
      weights_gate3 =tf.nn.sigmoid(conv2d(conv5_gate3, 128,128*4, k_h=3, k_w=3, d_h=1, d_w=1,name="weights_gate3"))
      weight3_rgb=weights_gate3[:,:,:,0:128]
      weight3_depth=weights_gate3[:,:,:,128:256]
      weight3_rgbd=weights_gate3[:,:,:,256:384]
      weight3_feature=weights_gate3[:,:,:,384:512]
      gata_output56=tf.add(tf.add(tf.add(tf.multiply(weight3_rgb,conv3_vgg_56_2),tf.multiply(weight3_depth,conv3_vgg_depth_56_2)),tf.multiply(weight3_rgbd,conv3_vgg_rgbd_56_2)),tf.multiply(weight3_feature,conc_2_refine_f_2up))
      gata_output56_conv = tf.nn.relu(conv2d(gata_output56, 128,128,k_h=3, k_w=3, d_h=1, d_w=1,name="gata_output56_conv"))
      channle_atten_rgb_56 = self.Squeeze_excitation_layer(conv3_vgg_56_2, out_dim=128, ratio=4, layer_name='channle_atten_rgb_56')
      channle_atten_rgb_56_conv = tf.nn.relu(conv2d(channle_atten_rgb_56, 128,128/2,k_h=1, k_w=1, d_h=1, d_w=1,name="channle_atten_rgb_56_conv"))
      channle_atten_depth_56 = self.Squeeze_excitation_layer(conv3_vgg_depth_56_2, out_dim=128, ratio=4, layer_name='channle_atten_depth_56')
      channle_atten_depth_56_conv = tf.nn.relu(conv2d(channle_atten_depth_56, 128,128/2,k_h=1, k_w=1, d_h=1, d_w=1,name="channle_atten_depth_56_conv"))
      channle_atten_rgbd_56 = self.Squeeze_excitation_layer(conv3_vgg_rgbd_56_2, out_dim=128, ratio=4, layer_name='channle_atten_rgbd_56')
      channle_atten_rgbd_56_conv = tf.nn.relu(conv2d(channle_atten_rgbd_56, 128,128/2,k_h=1, k_w=1, d_h=1, d_w=1,name="channle_atten_rgbd_56_conv"))
      channle_atten_28_56 = self.Squeeze_excitation_layer(conc_2_refine_f_2up, out_dim=128, ratio=4, layer_name='channle_atten_28_56')
      channle_atten_28_56_conv = tf.nn.relu(conv2d(channle_atten_28_56, 128,128/2,k_h=1, k_w=1, d_h=1, d_w=1,name="channle_atten_28_56_conv"))      
      conc_3 = tf.concat(axis = 3, values = [channle_atten_rgb_56_conv,channle_atten_depth_56_conv,channle_atten_rgbd_56_conv,channle_atten_28_56_conv]) #
      channle_atten_2_56_temp = self.Squeeze_excitation_layer(conc_3, out_dim=256, ratio=4, layer_name='channle_atten_2_56')
      channle_atten_2_56_temp_conv = tf.nn.relu(conv2d(channle_atten_2_56_temp, 128,128,k_h=3, k_w=3, d_h=1, d_w=1,name="channle_atten_2_56_temp_conv"))
      channle_atten_2_56 = tf.concat(axis = 3, values = [gata_output56_conv,channle_atten_2_56_temp_conv])
# spacial-attention
      conc_3_atten=tf.add(channle_atten_2_56,tf.multiply(channle_atten_2_56, tf.sigmoid(saliency_28_2up)))
# edge-attention
      conv1_edg_56 = tf.nn.relu(conv2d(tf.concat(axis = 3, values = [conv3_vgg_56_2,conv3_vgg_depth_56_2,conv3_vgg_rgbd_56_2,conc_2_refine_f_2up]), 128,128,k_h=3, k_w=3, d_h=1, d_w=1,name="conv1_edg_56"))
      conv2_edg_56 = tf.nn.relu(conv2d(conv1_edg_56, 128,128,k_h=3, k_w=3, d_h=1, d_w=1,name="conv2_edg_56"))
      edge_56 = conv2d(conv2_edg_56, 128,1,k_h=3, k_w=3, d_h=1, d_w=1,name="edge_56")
      conc_3_refine=tf.add(conc_3_atten,tf.multiply(conc_3_atten,tf.sigmoid(edge_56)))
## saliency 56
      conc_3_refine_f =tf.nn.relu(conv2d(conc_3_refine, 128,128,k_h=3, k_w=3, d_h=1, d_w=1,name="conc_3_refine_f"))
      saliency1_56 = tf.nn.relu(conv2d(conc_3_refine_f, 128,128,k_h=3, k_w=3, d_h=1, d_w=1,name="saliency1_56"))
      saliency_56 = conv2d(saliency1_56, 128,1,k_h=3, k_w=3, d_h=1, d_w=1,name="saliency_56")
      saliency_56_2up=tf.image.resize_bilinear(saliency_56,[112,112])
      conc_3_refine_f_2up_1 =tf.image.resize_bilinear(conc_3_refine,[112,112])
      conc_3_refine_f_2up_2=tf.nn.relu(conv2d(conc_3_refine_f_2up_1, 128,128,k_h=3, k_w=3, d_h=1, d_w=1,name="conc_3_refine_f_2up_2"))
      conc_3_refine_f_2up=tf.nn.relu(conv2d(conc_3_refine_f_2up_2, 128,64,k_h=3, k_w=3, d_h=1, d_w=1,name="conc_3_refine_f_2up"))
# 112
# 
## channle-on-channel-attention
#
      conc1_gate4 = tf.concat(axis = 3, values = [conv2_vgg_112_2,conv2_vgg_depth_112_2,conv2_vgg_rgbd_112_2,conc_3_refine_f_2up]) #
      conv1_gate4=tf.nn.relu(conv2d(conc1_gate4, 64*3,64,k_h=3, k_w=3, d_h=1, d_w=1,name="conv1_gate4"))
      conv2_gate4=tf.nn.relu(conv2d(conv1_gate4, 64,64,k_h=3, k_w=3, d_h=1, d_w=1,name="conv2_gate4"))
      conv3_gate4=tf.nn.relu(conv2d(conv2_gate4, 64,64,k_h=3, k_w=3, d_h=1, d_w=1,name="conv3_gate4"))
      conv4_gate4=tf.nn.relu(conv2d(conv3_gate4, 64,64,k_h=3, k_w=3, d_h=1, d_w=1,name="conv4_gate4"))
      conv5_gate4=tf.nn.relu(conv2d(conv4_gate4, 64,64,k_h=3, k_w=3, d_h=1, d_w=1,name="conv5_gate4"))
      weights_gate4 =tf.nn.sigmoid(conv2d(conv5_gate4, 64,64*4, k_h=3, k_w=3, d_h=1, d_w=1,name="weights_gate4"))
      weight4_rgb=weights_gate4[:,:,:,0:64]
      weight4_depth=weights_gate4[:,:,:,64:128]
      weight4_rgbd=weights_gate4[:,:,:,128:192]
      weight4_feature=weights_gate4[:,:,:,192:256]
      gata_output128=tf.add(tf.add(tf.add(tf.multiply(weight4_rgb,conv2_vgg_112_2),tf.multiply(weight4_depth,conv2_vgg_depth_112_2)),tf.multiply(weight4_rgbd,conv2_vgg_rgbd_112_2)),tf.multiply(weight4_feature,conc_3_refine_f_2up))
      gata_output128_conv = tf.nn.relu(conv2d(gata_output128, 64,64,k_h=3, k_w=3, d_h=1, d_w=1,name="gata_output128_conv"))
      channle_atten_rgb_112 = self.Squeeze_excitation_layer(conv2_vgg_112_2, out_dim=64, ratio=4, layer_name='channle_atten_rgb_112')
      channle_atten_rgb_112_conv = tf.nn.relu(conv2d(channle_atten_rgb_112, 64,64/2,k_h=1, k_w=1, d_h=1, d_w=1,name="channle_atten_rgb_112_conv"))      
      channle_atten_depth_112 = self.Squeeze_excitation_layer(conv2_vgg_depth_112_2, out_dim=64, ratio=4, layer_name='channle_atten_depth_112')
      channle_atten_depth_112_conv = tf.nn.relu(conv2d(channle_atten_depth_112, 64,64/2,k_h=1, k_w=1, d_h=1, d_w=1,name="channle_atten_depth_112_conv")) 
      channle_atten_rgbd_112 = self.Squeeze_excitation_layer(conv2_vgg_rgbd_112_2, out_dim=64, ratio=4, layer_name='channle_atten_rgbd_112')
      channle_atten_rgbd_112_conv = tf.nn.relu(conv2d(channle_atten_rgbd_112, 64,64/2,k_h=1, k_w=1, d_h=1, d_w=1,name="channle_atten_rgbd_112_conv"))       
      channle_atten_56_112 = self.Squeeze_excitation_layer(conc_3_refine_f_2up, out_dim=64, ratio=4, layer_name='channle_atten_56_112')
      channle_atten_56_112_conv = tf.nn.relu(conv2d(channle_atten_56_112, 64,64/2,k_h=1, k_w=1, d_h=1, d_w=1,name="channle_atten_56_112_conv"))             
      conc_4 = tf.concat(axis = 3, values = [channle_atten_rgb_112_conv,channle_atten_depth_112_conv,channle_atten_rgbd_112_conv,channle_atten_56_112_conv]) #
      channle_atten_2_112_temp = self.Squeeze_excitation_layer(conc_4, out_dim=128, ratio=4, layer_name='channle_atten_2_112')
      channle_atten_2_112_temp_conv = tf.nn.relu(conv2d(channle_atten_2_112_temp, 64,64,k_h=3, k_w=3, d_h=1, d_w=1,name="channle_atten_2_112_temp_conv"))
      channle_atten_2_112 = tf.concat(axis = 3, values = [gata_output128_conv,channle_atten_2_112_temp_conv])
# spacial-attention
# 
      conc_4_atten=tf.add(channle_atten_2_112,tf.multiply(channle_atten_2_112, tf.sigmoid(saliency_56_2up)))
# edge-attention
      conv1_edg_112 = tf.nn.relu(conv2d(tf.concat(axis = 3, values = [conv2_vgg_112_2,conv2_vgg_depth_112_2,conv2_vgg_rgbd_112_2,conc_3_refine_f_2up]), 128,128,k_h=3, k_w=3, d_h=1, d_w=1,name="conv1_edg_112"))
      conv2_edg_112 = tf.nn.relu(conv2d(conv1_edg_112, 64,64,k_h=3, k_w=3, d_h=1, d_w=1,name="conv2_edg_112"))
      edge_112 = conv2d(conv2_edg_112, 64,1,k_h=3, k_w=3, d_h=1, d_w=1,name="edge_112")
      conc_4_refine=tf.add(conc_4_atten,tf.multiply(conc_4_atten,tf.sigmoid(edge_112)))
## saliency 112
      conc_4_refine_f =tf.nn.relu(conv2d(conc_4_refine, 64,64,k_h=3, k_w=3, d_h=1, d_w=1,name="conc_4_refine_f"))
      saliency1_112 = tf.nn.relu(conv2d(conc_4_refine_f, 64,64,k_h=3, k_w=3, d_h=1, d_w=1,name="saliency1_112"))
      saliency_112 = conv2d(saliency1_112, 64,1,k_h=3, k_w=3, d_h=1, d_w=1,name="saliency_112")
      saliency_112_2up=tf.image.resize_bilinear(saliency_112,[224,224])
      conc_4_refine_f_2up_1 =tf.image.resize_bilinear(conc_4_refine,[224,224])
      conc_4_refine_f_2up_2=tf.nn.relu(conv2d(conc_4_refine_f_2up_1, 64,64,k_h=3, k_w=3, d_h=1, d_w=1,name="conc_4_refine_f_2up_2"))
      conc_4_refine_f_2up=tf.nn.relu(conv2d(conc_4_refine_f_2up_2, 64,32,k_h=3, k_w=3, d_h=1, d_w=1,name="conc_4_refine_f_2up"))
# 224
## channle-on-channel-attention
      conc1_gate5 = tf.concat(axis = 3, values = [conv1_vgg_224_2,conv1_vgg_depth_224_2,conv1_vgg_rgbd_224_2,conc_4_refine_f_2up]) #
      conv1_gate5=tf.nn.relu(conv2d(conc1_gate5, 32*3,32,k_h=3, k_w=3, d_h=1, d_w=1,name="conv1_gate5"))
      conv2_gate5=tf.nn.relu(conv2d(conv1_gate5, 32,32,k_h=3, k_w=3, d_h=1, d_w=1,name="conv2_gate5"))
      conv3_gate5=tf.nn.relu(conv2d(conv2_gate5, 32,32,k_h=3, k_w=3, d_h=1, d_w=1,name="conv3_gate5"))
      conv4_gate5=tf.nn.relu(conv2d(conv3_gate5, 32,32,k_h=3, k_w=3, d_h=1, d_w=1,name="conv4_gate5"))
      conv5_gate5=tf.nn.relu(conv2d(conv4_gate5, 32,32,k_h=3, k_w=3, d_h=1, d_w=1,name="conv5_gate5"))
      weights_gate5 =tf.nn.sigmoid(conv2d(conv5_gate5, 32,32*4, k_h=3, k_w=3, d_h=1, d_w=1,name="weights_gate5"))
      weight5_rgb=weights_gate5[:,:,:,0:32]
      weight5_depth=weights_gate5[:,:,:,32:64]
      weight5_rgbd=weights_gate5[:,:,:,64:96]
      weight5_feature=weights_gate5[:,:,:,96:128]
      gata_output224=tf.add(tf.add(tf.add(tf.multiply(weight5_rgb,conv1_vgg_224_2),tf.multiply(weight5_depth,conv1_vgg_depth_224_2)),tf.multiply(weight5_rgbd,conv1_vgg_rgbd_224_2)),tf.multiply(weight5_feature,conc_4_refine_f_2up))
      gata_output224_conv = tf.nn.relu(conv2d(gata_output224, 32,32,k_h=3, k_w=3, d_h=1, d_w=1,name="gata_output224_conv"))
      channle_atten_rgb_224 = self.Squeeze_excitation_layer(conv1_vgg_224_2, out_dim=32, ratio=4, layer_name='channle_atten_rgb_224')
      channle_atten_rgb_224_conv = tf.nn.relu(conv2d(channle_atten_rgb_224, 32,32/2,k_h=1, k_w=1, d_h=1, d_w=1,name="channle_atten_rgb_224_conv"))      
      channle_atten_depth_224 = self.Squeeze_excitation_layer(conv1_vgg_depth_224_2, out_dim=32, ratio=4, layer_name='channle_atten_depth_224')
      channle_atten_depth_224_conv = tf.nn.relu(conv2d(channle_atten_depth_224, 32,32/2,k_h=1, k_w=1, d_h=1, d_w=1,name="channle_atten_depth_224_conv")) 
      channle_atten_rgbd_224 = self.Squeeze_excitation_layer(conv1_vgg_rgbd_224_2, out_dim=32, ratio=4, layer_name='channle_atten_rgbd_224')
      channle_atten_rgbd_224_conv = tf.nn.relu(conv2d(channle_atten_rgbd_224, 32,32/2,k_h=1, k_w=1, d_h=1, d_w=1,name="channle_atten_rgbd_224_conv")) 
      channle_atten_112_224 = self.Squeeze_excitation_layer(conc_4_refine_f_2up, out_dim=32, ratio=4, layer_name='channle_atten_112_224')
      channle_atten_112_224_conv = tf.nn.relu(conv2d(channle_atten_112_224, 32,32/2,k_h=1, k_w=1, d_h=1, d_w=1,name="channle_atten_112_224_conv")) 
      conc_5 = tf.concat(axis = 3, values = [channle_atten_rgb_224_conv,channle_atten_depth_224_conv,channle_atten_rgbd_224_conv,channle_atten_112_224_conv]) #
      channle_atten_2_224_temp = self.Squeeze_excitation_layer(conc_5, out_dim=64, ratio=4, layer_name='channle_atten_2_224')
      channle_atten_2_224_temp_conv = tf.nn.relu(conv2d(channle_atten_2_224_temp, 32,32,k_h=3, k_w=3, d_h=1, d_w=1,name="channle_atten_2_224_temp_conv"))
      channle_atten_2_224 =tf.concat(axis = 3, values = [gata_output224_conv,channle_atten_2_224_temp_conv])
# spacial-attention
      conc_5_atten=tf.add(channle_atten_2_224,tf.multiply(channle_atten_2_224, tf.sigmoid(saliency_112_2up)))
# edge-attention
      conv1_edg_224 = tf.nn.relu(conv2d(tf.concat(axis = 3, values = [conv1_vgg_224_2,conv1_vgg_depth_224_2,conv1_vgg_rgbd_224_2,conc_4_refine_f_2up]), 32,32,k_h=3, k_w=3, d_h=1, d_w=1,name="conv1_edg_224"))
      conv2_edg_224 = tf.nn.relu(conv2d(conv1_edg_224, 32,32,k_h=3, k_w=3, d_h=1, d_w=1,name="conv2_edg_224"))
      edge_224 = conv2d(conv2_edg_224, 32,1,k_h=3, k_w=3, d_h=1, d_w=1,name="edge_224")
      conc_5_refine=tf.add(conc_5_atten,tf.multiply(conc_5_atten, tf.sigmoid(edge_224)))
## saliency 224
      conc_5_refine_f =tf.nn.relu(conv2d(conc_5_refine, 32,32,k_h=3, k_w=3, d_h=1, d_w=1,name="conc_5_refine_f"))
      saliency1_224 = tf.nn.relu(conv2d(conc_5_refine_f, 32,32,k_h=3, k_w=3, d_h=1, d_w=1,name="saliency1_224"))
      saliency_224 = conv2d(saliency1_224, 32,1,k_h=3, k_w=3, d_h=1, d_w=1,name="saliency_224")
    return tf.nn.sigmoid(saliency_224)
    



  def save(self, checkpoint_dir, step):
    model_name = "coarse.model"
    model_dir = "%s_%s" % ("coarse", self.label_height)
    checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    self.saver.save(self.sess,
                    os.path.join(checkpoint_dir, model_name),
                    global_step=step)

  def load(self, checkpoint_dir):
    print(" [*] Reading checkpoints...")
    model_dir = "%s_%s" % ("coarse", self.label_height)
    checkpoint_dir = os.path.join(checkpoint_dir, model_dir)
    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
        ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
        self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
        return True
    else:
        return False

  def Squeeze_excitation_layer(self, input_x, out_dim, ratio, layer_name):
    with tf.name_scope(layer_name) :
        squeeze = Global_Average_Pooling(input_x)

        excitation = Fully_connected(squeeze, units=out_dim / ratio, layer_name=layer_name+'_fully_connected1')
        excitation = Relu(excitation)
        excitation = Fully_connected(excitation, units=out_dim, layer_name=layer_name+'_fully_connected2')
        excitation = Sigmoid(excitation)

        excitation = tf.reshape(excitation, [-1,1,1,out_dim])

        scale = input_x * excitation

        return scale 