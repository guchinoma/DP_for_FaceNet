from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gzip
import os
import sys
import time
import pickle
from ConfigParser import SafeConfigParser

import numpy as np
from six.moves import urllib
from six.moves import xrange
import tensorflow as tf

#from adversarial.fastgradientsign_advgen_triplet_losses import FastGradientSign_AdvGen
from adversarial.triplet_loss_advgen import TripletLoss_AdvGen

import facenet

from models.nn4 import inference

import tensorflow.contrib.slim as slim


tf.app.flags.DEFINE_string('config_path', './config/lfw.conf', 'Application configuration file.')
tf.app.flags.DEFINE_boolean('restore_checkpoint', False, 'Skip training, restore from checkpoint.')
tf.app.flags.DEFINE_boolean('test', False, 'Test run with a fraction of the data.')
cmd_args = tf.app.flags.FLAGS

def main(argv=None):

    print("Debug")

    config = SafeConfigParser()
    config.read(cmd_args.config_path)

    with tf.Graph().as_default() as g:

        global_step = tf.Variable(0, trainable=False)

        #saver = tf.train.Saver(tf.global_variable())

        print("Now it is the time to load from the downloaded one")

        #############################
        image_size = config.getint("main", "image_size")
        batch_size = config.getint("main", "batch_size")
        # batch means the subset of the learning data.
        max_nrof_epochs = config.getint("main", "max_nrof_epochs")
        random_crop = True
        random_flip = False
        nrof_preprocess_threads = config.getint("main", "nrof_preprocess_threads")
        ############################     

        # Get the directory
        train_set = facenet.get_dataset(config.get("data", "work_directory"))
        
        # Get image lists and label lists
        image_list, label_list = facenet.get_image_paths_and_labels(train_set)
        
        # What the image is and label is 
        images_raw, labels_eval = facenet.read_and_augument_data(image_list, label_list, image_size, batch_size, max_nrof_epochs, random_crop, random_flip, nrof_preprocess_threads)

        print("tensor:%s" % images_raw)

        num_classes = len(os.listdir(config.get("data", "work_directory")))

        print("classes:%s" % num_classes)

        lfw_list = []

        lfw_classes = os.listdir(config.get("data", "work_directory"))
        lfw_classes.sort()

        for i in lfw_classes:
            lfw_list.append(i)

        keep_probability = 1.0

        # logits_eval is like this
        #
        #     1    [........]
        #     2    [........]
        #     3    [........]
        #     .        .
        #     .        .
        #     .        .
        #     128  [........]
        #
        #     128 x 10(num_channel)
        #
        # This goes to
        # cross_entropy = tf.nn.softmax_cross_entrpy_with_logts(logits......)
        
        prelogits, _ = inference(images_raw, keep_probability, phase_train=True, weight_decay=0.0)
        
        # logits_single is like this
        #
        #     1    [..................................]
        #
        #          |----------------------------------|
        #                           a
        #
        #     1 x 10(num_channel) with whitening(but idk why)
        #
        # This goes to
        # pred_logit = sess.run(logits_single, feed_dict={x:raw_image_reshaped})
        # (12/24) Changed the shape of logits because there is something wrong in model function. Should be changed 

        print("Done")

        pre_embeddings = slim.fully_connected(prelogits, 128, activation_fn=None, scope='Embeddings', reuse=False)
        logits_eval = tf.nn.l2_normalize(pre_embeddings, 1, 1e-10, name='embeddings')


    input_dict = {
        "graph": g,
        "x": images_raw,
        "x_raw": images_raw,
        "y_": labels_eval,
        "y_conv": logits_eval,
        #"y_conv_single": logits_single,
        #"adv_image_placeholder": x,
        #"keep_prob": None,
    }

    print("succees in generating images")

    output_dir = config.get("main", "image_output_path")

    #fastgradientsign_advgen = FastGradientSign_AdvGen(cmd_args, [1, 24, 24, 3], config)
    #fastgradientsign_advgen.run_queue_for_lfw(input_dict, num_classes, lfw_list)
    tripletloss_advgen = TripletLoss_AdvGen(cmd_args, [1, 24, 24, 3], config)
    tripletloss_advgen.run_queue_for_lfw(input_dict, num_classes, lfw_list, output_dir)

if __name__ == '__main__':

    tf.app.run()
