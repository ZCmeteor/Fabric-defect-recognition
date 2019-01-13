# -*- coding: utf-8 -*-
"""
Created on Sat May  5 18:09:00 2018

@author: 11420
"""

import numpy as np
import tensorflow as tf
import VGG16_model as model
import create_and_read_TFRecord2 as reader2

if __name__ == '__main__':

    X_train, y_train = reader2.get_file("D:/Deep Code/defect-vs-normal/train/")
    image_batch, label_batch = reader2.get_batch(X_train, y_train, 224, 224, 100, 512)

    x_imgs = tf.placeholder(tf.float32, [None, 224, 224, 3])
    y_imgs = tf.placeholder(tf.int32, [None, 2])

    vgg = model.vgg16(x_imgs)
    fc3_defect_and_normal = vgg.probs
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=fc3_defect_and_normal, labels=y_imgs))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-4).minimize(loss)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    vgg.load_weights('D:/Deep Code/Course/defect-vs-normal/vgg16_weights.npz', sess)
    saver = vgg.saver()

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

    saver.save(sess, "D:/Deep Code/defect-vs-normal/model/")
    print("Optimization Finished!")