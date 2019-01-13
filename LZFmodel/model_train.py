# -*- coding: utf-8 -*-
"""
Created on Sat May  5 18:09:00 2018

@author: 11420
"""

import numpy as np
import tensorflow as tf
import LZFNet_model as model
import Preprocessing as reader2


if __name__ == '__main__':

    X_train, y_train = reader2.get_file("D:/Deep Code/(LZFNet)/train/")

    image_batch, label_batch = reader2.get_batch(X_train, y_train, 224, 224, 100, 512) 

    x_imgs = tf.placeholder(tf.float32, [None, 224, 224, 3])
    y_imgs = tf.placeholder(tf.int32, [None, 2])

    LZFNet = model.LZFNet(x_imgs)
    fc3_normal_and_defect = LZFNet.probs
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=fc3_normal_and_defect, labels=y_imgs))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-4).minimize(loss)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    saver = LZFNet.saver()

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord, sess=sess)

    import time
    start_time = time.time()

    for i in range(2000):  

            image, label = sess.run([image_batch, label_batch])
            labels = reader2.onehot(label)

            sess.run(optimizer, feed_dict={x_imgs: image, y_imgs: labels})
            loss_record = sess.run(loss, feed_dict={x_imgs: image, y_imgs: labels})
            print("now the loss is %f " % loss_record)
            end_time = time.time()
            print('time: ', (end_time - start_time))
            start_time = end_time
            print("----------epoch %d is finished---------------" % i)

    saver.save(sess, "D:/Deep Code/(LZFNet)/model/")
    print("Optimization Finished!")