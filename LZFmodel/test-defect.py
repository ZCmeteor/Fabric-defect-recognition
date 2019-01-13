# -*- coding: utf-8 -*-
"""
Created on Sat May  5 18:26:00 2018

@author: 11420
"""

import tensorflow as tf
from scipy.misc import imread, imresize
import LZFNet as model

imgs = tf.placeholder(tf.float32, [None, 224, 224, 3])
sess = tf.Session()
LZF = model.LZFNet(imgs)
fc3_normal_and_defect = LZF.probs
saver = LZF.saver()
saver.restore(sess, 'D:/Deep Code/(LZFNet)/model/LZFmodel06/')

import os
for root, sub_folders, files in os.walk('D:/Deep Code/(LZFNet)/test/'):
    i = 0
    normal = 0
    defect = 0
    for name in files:
        i += 1
        filepath = os.path.join(root, name)

        try:
            img1 = imread(filepath, mode='RGB')
            img1 = imresize(img1, (224, 224))
        except:
            print("remove", filepath)

        prob = sess.run(fc3_normal_and_defect, feed_dict={LZF.imgs: [img1]})
        import numpy as np
        max_index = np.argmax(prob)
        if max_index == 0:
            normal += 1
        else:
            defect += 1
        if i % 50 == 0:
            acc = (defect * 1.)/(normal + defect)
            print(acc)
            print("-----------img number is %d------------" % i)