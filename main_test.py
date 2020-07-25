from model import T_CNN
from utils import *
import numpy as np
import tensorflow as tf

import pprint
import os
from tensorflow.contrib.slim import nets
slim = tf.contrib.slim
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
 

 
flags = tf.app.flags
flags.DEFINE_integer("epoch", 120, "Number of epoch [120]")
flags.DEFINE_integer("batch_size", 1, "The size of batch images [128]")
flags.DEFINE_integer("image_height", 224, "The size of image to use [230]")
flags.DEFINE_integer("image_width", 224, "The size of image to use [310]")
flags.DEFINE_integer("label_height", 224, "The size of label to produce [230]")
flags.DEFINE_integer("label_width", 224, "The size of label to produce [310]")
flags.DEFINE_float("learning_rate", 0.001, "The learning rate of gradient descent algorithm [1e-4]")
flags.DEFINE_float("beta1", 0.5, "Momentum term of adam [0.5]")
flags.DEFINE_integer("c_dim", 3, "Dimension of image color. [3]")
flags.DEFINE_integer("c_depth_dim", 1, "Dimension of depth. [1]")
# flags.DEFINE_integer("stride", 14, "The size of stride to apply input image [14]")model_dehaze_depth
flags.DEFINE_string("checkpoint_dir", "checkpoint", "Name of checkpoint directory [checkpoint]")
flags.DEFINE_string("sample_dir", "sample", "Name of sample directory [sample]")
flags.DEFINE_string("test_data_dir", "test", "Name of sample directory [test]")
flags.DEFINE_boolean("is_train", False, "True for training, False for testing [True]")
FLAGS = flags.FLAGS

pp = pprint.PrettyPrinter()

def main(_):
  pp.pprint(flags.FLAGS.__flags)

  if not os.path.exists(FLAGS.checkpoint_dir):
    os.makedirs(FLAGS.checkpoint_dir)
  if not os.path.exists(FLAGS.sample_dir): 
    os.makedirs(FLAGS.sample_dir)

  #load test data
  filenames = os.listdir('test_real')
  data_dir = os.path.join(os.getcwd(), 'test_real')
  data = sorted(glob.glob(os.path.join(data_dir, "*.png")))
  test_data_list = data + sorted(glob.glob(os.path.join(data_dir, "*.jpg")))+sorted(glob.glob(os.path.join(data_dir, "*.bmp")))

  filenames1 = os.listdir('depth_real')
  data_dir1 = os.path.join(os.getcwd(), 'depth_real')
  data1 = sorted(glob.glob(os.path.join(data_dir1, "*.png")))
  test_data_list1 = data1 + sorted(glob.glob(os.path.join(data_dir1, "*.jpg")))+sorted(glob.glob(os.path.join(data_dir1, "*.bmp")))

  # init session
  sess = tf.Session()

  # build the graph
  srcnn = T_CNN(sess,
                batch_size=FLAGS.batch_size,
                c_dim=FLAGS.c_dim,
                c_depth_dim=FLAGS.c_depth_dim,
                )

  #load checkpoint file
  srcnn.load(checkpoint_dir = FLAGS.checkpoint_dir)
  print(" Reading checkpoints...")
  for ide in range(0,len(test_data_list)):
    image_test =  get_image(test_data_list[ide],is_grayscale=False)
    depth_test =  get_image(test_data_list1[ide],is_grayscale=False)
    test_image_name = test_data_list[ide]
    srcnn.train(image_test, depth_test, test_image_name, FLAGS)

if __name__ == '__main__':
  tf.app.run()
