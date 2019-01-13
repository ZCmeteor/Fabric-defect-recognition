
"""
@author: 11420
"""
import os
import sys
import time
import copy
import numpy as np
import tensorflow as tf
import lzfnet_path
from scipy.misc import imread, imresize
from tf_cnnvis import *




mean = np.load("D:/Weights/img_mean.npy").transpose((1, 2, 0))

tf.reset_default_graph()

X = tf.placeholder(tf.float32, shape = [None, 224, 224, 3]) 
y_ = tf.placeholder(tf.float32, shape = [None, 2]) 

class LZFNet:
    def __init__(self, imgs,weights=None, sess=None):
        self.parameters = []
        self.imgs = imgs
        self.convlayers()
        self.fc_layers()
        self.probs = self.fc8
        if weights is not None and sess is not None:
            self.load_weights("D:/Weights/LZFNet.npz", sess)

    def saver(self):
        return tf.train.Saver()

    def maxpool(self,name,input_data, trainable):
        out = tf.nn.max_pool(input_data,[1,2,2,1],[1,2,2,1],padding="SAME",name=name)
        return out

    def conv(self,name, input_data, out_channel, trainable = True):
        in_channel = input_data.get_shape()[-1]
        with tf.variable_scope(name):
            kernel = tf.get_variable("weights", [3, 3, in_channel, out_channel], dtype=tf.float32,trainable=trainable)
            biases = tf.get_variable("biases", [out_channel], dtype=tf.float32,trainable=trainable)
            conv_res = tf.nn.conv2d(input_data, kernel, [1, 1, 1, 1], padding="SAME")
            res = tf.nn.bias_add(conv_res, biases)
            out = tf.nn.relu(res, name=name)
        self.parameters += [kernel, biases]
        return out

    def fc(self,name,input_data,out_channel,trainable = True):
        shape = input_data.get_shape().as_list()
        if len(shape) == 4:
            size = shape[-1] * shape[-2] * shape[-3]
        else:size = shape[1]
        input_data_flat = tf.reshape(input_data,[-1,size])
        with tf.variable_scope(name):
            weights = tf.get_variable(name="weights",shape=[size,out_channel],dtype=tf.float32,trainable = trainable)
            biases = tf.get_variable(name="biases",shape=[out_channel],dtype=tf.float32,trainable = trainable)
            res = tf.matmul(input_data_flat,weights)
            out = tf.nn.relu(tf.nn.bias_add(res,biases))
        self.parameters += [weights, biases]
        return out

    def convlayers(self):
               
        self.conv1_1 = self.conv("conv1_1",self.imgs,64,trainable=True) #trainable=False=True
        self.pool1 = self.maxpool("pool1",self.conv1_1,trainable=True)                    
        self.conv2 = self.conv("conv2",self.pool1,64,trainable=True)        
        self.pool2 = self.maxpool("pool2",self.conv2,trainable=True)       
        self.conv3 = self.conv("conv3",self.pool2,128,trainable=True)
        self.conv4 = self.conv("conv4",self.conv3,128,trainable=True)      
        self.pool3 = self.maxpool("pool3",self.conv4,trainable=True)       
        self.conv5 = self.conv("conv5",self.pool3,256,trainable=True)
        self.conv6 = self.conv("conv6",self.conv5,256,trainable=True)
        self.conv7 = self.conv("conv7",self.conv6,256,trainable=True)
        self.pool4 = self.maxpool("pool4",self.conv7,trainable=True)       
        self.conv8 = self.conv("conv8",self.pool4,512,trainable=True)
        self.conv9 = self.conv("conv9",self.conv8,64,trainable=True)
        self.pool5 = self.maxpool("pool5",self.conv9,trainable=True)

    def fc_layers(self):

        self.fc7 = self.fc("fc7", self.pool5, 2048,trainable=True)
        self.fc8 = self.fc("fc8", self.fc7, 2)

    def load_weights(self, weight_file, sess):
        weights = np.load(weight_file)
        keys = sorted(weights.keys())
        for i, k in enumerate(keys):          
                sess.run(self.parameters[i].assign(weights[k]))
        print("-----------all done---------------")
        
        
if __name__ == '__main__':
    sess = tf.Session()
    imgs = tf.placeholder(tf.float32, [None, 224, 224, 3])
    LZFNet = LZFNet(X, 'LZFNet.npz', sess)
    

"""  
    #####################
    ### visualization ###
    #####################
"""

# reading sample image
im = np.expand_dims(imresize(imresize(imread(os.path.join("D:/","Deep Code","Typical-model", "examples", "images.jpg")), (256, 256)) - mean, 
                             (224, 224)), axis = 0)


sess = tf.Session(graph=tf.get_default_graph())
sess.run(tf.global_variables_initializer())

# activation visualization
layers = ['r', 'p', 'c']

start = time.time()
with sess.as_default():

    is_success = activation_visualization(sess_graph_path = None, value_feed_dict = {X : im}, 
                                          layers=layers, path_logdir=os.path.join("D:/","Deep Code","Typical-model","Log","LZFNet"), 
                                          path_outdir=os.path.join("D:/","Deep Code","Typical-model","Output","LZFNet"))
start = time.time() - start
print("Total Time = %f" % (start))

# deconv visualization
layers = ['r', 'p', 'c']

start = time.time()
with sess.as_default():
    is_success = deconv_visualization(sess_graph_path = None, value_feed_dict = {X : im}, 
                                      layers=layers, path_logdir=os.path.join("D:/","Deep Code","Typical-model","Log","LZFNet"), 
                                      path_outdir=os.path.join("D:/","Deep Code","Typical-model","Output","LZFNet"))
start = time.time() - start
print("Total Time = %f" % (start))




sess.close()


