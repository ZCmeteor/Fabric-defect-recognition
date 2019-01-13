# -*- coding: utf-8 -*-
"""
Created on Sat May  5 18:26:00 2018

@author: 11420
"""

import tensorflow as tf
from scipy.misc import imread, imresize
import VGG16_model as model

imgs = tf.placeholder(tf.float32, [None, 224, 224, 3])
sess = tf.Session()
vgg = model.vgg16(imgs)
fc3_defect_and_normal = vgg.probs
saver = vgg.saver()
saver.restore(sess, 'D:/Deep Code/defect-vs-normal/model/stripe01/')

import os
for root, sub_folders, files in os.walk('D:/Deep Code/defect-vs-normal/test/'):
    i = 0
    defect = 0
    normal = 0
    for name in files:
        i += 1
        filepath = os.path.join(root, name)

        try:
            img1 = imread(filepath, mode='RGB')
            img1 = imresize(img1, (224, 224))
        except:
            print("remove", filepath)

        prob = sess.run(fc3_defect_and_normal, feed_dict={vgg.imgs: [img1]})
        import numpy as np
        max_index = np.argmax(prob)
        if max_index == 0:
            defect += 1
        else:
            normal += 1
        if i % 50 == 0:
            acc = (defect * 1.)/(normal + defect)
            print(acc)
            print("-----------img number is %d------------" % i)