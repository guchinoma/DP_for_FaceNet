from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time

import numpy as np
import pandas as pd
import tensorflow as tf
import facenet
from pulp import *

import os

class FastGradientSign_AdvGen:

    def __init__(self, cmd_args, input_x_shape, config):
        self.cmd_args = cmd_args
        self.input_x_shape = input_x_shape
        self.config = config

    def __len__(self):
        return len(self.input_x_shape)

    # This program is just for generation adversarial examples.
    def run_queue_for_lfw(self, input_dict, num_classes, lfw_list):
        graph = input_dict["graph"]
        images = input_dict["x"]
        raw_images = input_dict["x_raw"]
        labels = input_dict["y_"]
        logits = input_dict["y_conv"]

        adversarial_perturbation_min = self.config.getfloat(
            'main', 'adversarial_perturbation_min')
        adversarial_perturbation_max = self.config.getfloat(
            'main', 'adversarial_perturbation_max')
        adversarial_perturbation_steps = self.config.getfloat(
            'main', 'adversarial_perturbation_steps')
        perturbation_const = self.config.getfloat(
            'main', 'perturbation_const')

        with graph.as_default():

            init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

            #Restore the moving average version of the learned variables for eval.

            y_ = tf.one_hot(indices=tf.cast(labels, "int64"), 
                depth=num_classes, 
                on_value=1.0, 
                off_value=0.0)

            anchor, positive, negative = tf.unpack(tf.reshape(logits, [-1,3,128]), 3, 1)
            triplet_loss = facenet.triplet_loss(anchor, positive, negative, 0.2)
            regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
            total_loss = tf.add_n([cross_entropy_mean] + regularization_losses, name='total_loss')

            #cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits, y_)
            grad = tf.gradients(total_loss, images)
            
            print("succeed in calculating loss")

            with tf.Session() as sess:

                # Start the queue runners.

                sess.run(init_op)

                #coord = tf.train.Coordinator()
                #try:
                    #threads = []
                    #for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
                        #threads.extend(qr.create_threads(sess, coord=coord, daemon=True,
                                                         #start=True))
                    
                print("succeed in producing threads")

                sample_count = int(self.config.get('main', 'num_examples_per_epoch_eval'))
 
                print("succeed in initializing adv_diff or all kinds of stuffs")

                while step < sample_count and not coord.should_stop():

                    print("Initialising")

                    raw_images_val, images_val, labels_val, triplet_loss_val, grad_val = sess.run([raw_images, images, labels, total_loss, grad[0]])
    
                    print("succeeded in sess.run()")

 
                    step += 1


                    #for perturbation in np.linspace(adversarial_perturbation_min, 
                                                    #adversarial_perturbation_max, 
                                                    #adversarial_perturbation_steps):

                    #filename_1 = os.path.join("/Adversarial_data", str(perturbation))
                    filename_1 = os.path.join("/Users/KeiNomaguchi/Adversarial_data", str(perturbation_const))

                    if os.path.exists(filename_1):
                        filename_2 = filename_1
                    else:
                        os.makedirs(filename_1)
                        filename_2 = filename_1

                    for i in range(len(images_val)):
                        image = raw_images_val[i]
                        true_label = labels_val[i]

                        filename_3 = os.path.join(filename_2, lfw_list[true_label])

                        if os.path.exists(filename_3):
                            filename_4 = filename_3

                        else:
                            os.makedirs(filename_3)
                            filename_4 = filename_3

                        grad_sign = np.sign(grad_val[i])

                        adv_image = perturbation_const * grad_sign + image
                        print("adversarial example is generated")
                        adv_image_reshaped = np.reshape(adv_image, np.insert(adv_image.shape, 0 , 1))

                        output_image = tf.cast(adv_image, tf.uint8)
                        print("output:%s" % output_image)
                        filename_jpeg = '%s-%03d.jpeg' % (lfw_list[true_label], i)
                        filename_jpeg_2 = os.path.join(filename_4, filename_jpeg)
                        with open(filename_jpeg_2, 'wb') as f1, open('file_name_adv.txt', 'r') as f2:
                            f1.write(sess.run(tf.image.encode_jpeg(output_image)))
                            f2.write(filename_jpeg_2)

                        print("succeeded in generating adversarial examples")

                #except Exception as e:
                    #coord.request_stop(e)
                    #print("failed to load")
                #coord.request_stop()
                #coord.join(threads, stop_grace_period_secs=10)
