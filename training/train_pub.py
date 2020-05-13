# -*- coding: utf-8 -*-
"""
Created on Mon Aug 20 21:48:47 2018

@author: yy
"""

import datetime
import os
import sys
sys.path.append(".")
import keras.backend.tensorflow_backend as TK
import keras.backend as K
import numpy as np
import tensorflow as tf
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.optimizers import Adam

from mtcnn_model.mtcnn_model import p_net, r_net, o_net
from config import LABEL_MAP, MODEL_WEIGHT_SAVE_DIR, LOG_DIR

MODES = ['label', 'bbox', 'landmark']   # not used

NEGATIVE = TK.constant(LABEL_MAP['0'])  # not used
POSITIVE = TK.constant(LABEL_MAP['1'])
PARTIAL = TK.constant(LABEL_MAP['-1'])
LANDMARK = TK.constant(LABEL_MAP['-2'])
num_keep_radio = 0.7


def cal_mask(label_true, _type='label'):    # label_true: ground truth labels
    """
    Calculate mask as filter to get particular examples with respect to different training target
    :param label_true: ground truth tensor
    :param _type: 'label' 'bbox' or 'landmark'('label' by default)
    :return: mask
    """
    def true_func():
        return 0

    def false_func():
        return 1

    label_true_int32 = tf.cast(label_true, dtype=tf.int32)
    if _type == 'label':        # need pos and neg examples when training label
        label_filtered = tf.map_fn(lambda x: tf.cond(tf.equal(x[0], x[1]), true_func, false_func), label_true_int32)
    elif _type == 'bbox':       # need pos and part examples when training label
        label_filtered = tf.map_fn(lambda x: tf.cond(tf.equal(x[0], 1), true_func, false_func), label_true_int32)
    elif _type == 'landmark':   # need landmark examples when training label
        label_filtered = tf.map_fn(lambda x: tf.cond(tf.logical_and(tf.equal(x[0], 1), tf.equal(x[1], 1)),
                                                     false_func, true_func), label_true_int32)
    else:
        raise ValueError('Unknown type of: {} while calculate mask'.format(_type))

    mask = tf.cast(label_filtered, dtype=tf.int32)
    return mask


def label_ohem(label_true, label_pred):
    label_int = cal_mask(label_true, 'label')

    num_cls_prob = tf.size(label_pred)
    print('num_cls_prob: ', num_cls_prob)
    cls_prob_reshape = tf.reshape(label_pred, [num_cls_prob, -1])
    print('label_pred shape: ', tf.shape(label_pred))
    num_row = tf.shape(label_pred)[0]
    num_row = tf.to_int32(num_row)
    row = tf.range(num_row) * 2
    indices_ = row + label_int
    label_prob = tf.squeeze(tf.gather(cls_prob_reshape, indices_))
    loss = -tf.log(label_prob + 1e-10)

    valid_inds = cal_mask(label_true, 'label')
    num_valid = tf.reduce_sum(valid_inds)       # get the num of valid samples

    keep_num = tf.cast(tf.cast(num_valid, dtype=tf.float32) * num_keep_radio, dtype=tf.int32)
    # set 0 to invalid sample
    loss = loss * tf.cast(valid_inds, dtype=tf.float32)
    loss, _ = tf.nn.top_k(loss, k=keep_num)     # get top k losses
    return tf.reduce_mean(loss)


def bbox_ohem(label_true, bbox_true, bbox_pred):
    mask = cal_mask(label_true, 'bbox')
    num = tf.reduce_sum(mask)
    keep_num = tf.cast(num, dtype=tf.int32)

    bbox_true1 = tf.boolean_mask(bbox_true, mask, axis=0)   # get valid b-boxes
    bbox_pred1 = tf.boolean_mask(bbox_pred, mask, axis=0)

    square_error = tf.square(bbox_pred1 - bbox_true1)
    square_error = tf.reduce_sum(square_error, axis=1)

    _, k_index = tf.nn.top_k(square_error, k=keep_num)
    square_error = tf.gather(square_error, k_index)

    return tf.reduce_mean(square_error)


def landmark_ohem(label_true, landmark_true, landmark_pred):
    mask = cal_mask(label_true, 'landmark')
    num = tf.reduce_sum(mask)
    keep_num = tf.cast(num, dtype=tf.int32)

    landmark_true1 = tf.boolean_mask(landmark_true, mask)
    landmark_pred1 = tf.boolean_mask(landmark_pred, mask)

    square_error = tf.square(landmark_pred1 - landmark_true1)
    square_error = tf.reduce_sum(square_error, axis=1)

    _, k_index = tf.nn.top_k(square_error, k=keep_num)
    square_error = tf.gather(square_error, k_index)

    return tf.reduce_mean(square_error)


def _loss_func(y_true, y_pred):     # calculate total loss
    # labels_true = y_true[:, :1]
    # bbox_true = y_true[:, 2:6]
    # landmark_true = y_true[:, 6:]
    #
    # labels_pred = y_pred[:, :1]
    # bbox_pred = y_pred[:, 2:6]
    # landmark_pred = y_pred[:, 6:]
    #
    # label_loss = label_ohem(labels_true, labels_pred)
    # bbox_loss = bbox_ohem(labels_true, bbox_true, bbox_pred)
    # landmark_loss = landmark_ohem(labels_true, landmark_true, landmark_pred)

    zero_index = K.zeros_like(y_true[:, 0])
    ones_index = K.ones_like(y_true[:, 0])

    # Classifier
    labels = y_true[:, 0]
    class_preds = y_pred[:, 0]
    bi_crossentropy_loss = -labels * K.log(class_preds) - (1 - labels) * K.log(1 - class_preds)

    classify_valid_index = tf.where(K.less(y_true[:, 0], 0), zero_index, ones_index)
    classify_keep_num = K.cast(tf.reduce_sum(classify_valid_index) * num_keep_radio, dtype=tf.int32)
    # For classification problem, only pick 70% of the valid samples.

    classify_loss_sum = bi_crossentropy_loss * classify_valid_index
    classify_loss_sum_filtered, _ = tf.nn.top_k(classify_loss_sum, k=classify_keep_num)
    classify_loss = K.mean(classify_loss_sum_filtered)

    # Bounding box regressor
    rois = y_true[:, 1: 5]
    roi_preds = y_pred[:, 1: 5]
    # roi_raw_mean_square_error = K.sum(K.square(rois - roi_preds), axis = 1) # mse
    roi_raw_smooth_l1_loss = K.mean(tf.where(K.abs(rois - roi_preds) < 1, 0.5 * K.square(rois - roi_preds),
                                             K.abs(rois - roi_preds) - 0.5))  # L1 Smooth Loss

    roi_valid_index = tf.where(K.equal(K.abs(y_true[:, 0]), 1), ones_index, zero_index)
    roi_keep_num = K.cast(tf.reduce_sum(roi_valid_index), dtype=tf.int32)

    # roi_valid_mean_square_error = roi_raw_mean_square_error * roi_valid_index
    # roi_filtered_mean_square_error, _ = tf.nn.top_k(roi_valid_mean_square_error, k = roi_keep_num)
    # roi_loss = K.mean(roi_filtered_mean_square_error)
    roi_valid_smooth_l1_loss = roi_raw_smooth_l1_loss * roi_valid_index
    roi_filtered_smooth_l1_loss, _ = tf.nn.top_k(roi_valid_smooth_l1_loss, k=roi_keep_num)
    roi_loss = K.mean(roi_filtered_smooth_l1_loss)

    loss = classify_loss * 1 + roi_loss * 0.5

    return loss

    # return label_loss + bbox_loss * 0.5 + landmark_loss * 0.5


def _onet_loss_func(y_true, y_pred):
    labels_true = y_true[:, :2]
    bbox_true = y_true[:, 2:6]
    landmark_true = y_true[:, 6:]

    labels_pred = y_pred[:, :2]
    bbox_pred = y_pred[:, 2:6]
    landmark_pred = y_pred[:, 6:]

    label_loss = label_ohem(labels_true, labels_pred)
    bbox_loss = bbox_ohem(labels_true, bbox_true, bbox_pred)
    landmark_loss = landmark_ohem(labels_true, landmark_true, landmark_pred)

    return label_loss + bbox_loss * 0.5 + landmark_loss     # ratio is different



def train_p_net_with_data_generator(data_gen, steps_per_epoch, initial_epoch=0, epochs=1000, lr=0.001,
                                    callbacks=None, weights_file=None):
    _p_net = p_net(training=True)
    _p_net.summary()
    if weights_file is not None:
        _p_net.load_weights(weights_file)

    _p_net.compile(Adam(lr=lr), loss=_loss_func, metrics=['accuracy'])

    _p_net.fit_generator(data_gen,
                         steps_per_epoch=steps_per_epoch,
                         initial_epoch=initial_epoch,
                         epochs=epochs,
                         callbacks=callbacks)
    return _p_net

def train_r_net_with_data_generator(data_gen, steps_per_epoch, initial_epoch=0, epochs=1000, lr=0.001,
                                    callbacks=None, weights_file=None):
    _r_net = r_net(training=True)
    _r_net.summary()
    if weights_file is not None:
        _r_net.load_weights(weights_file)

    _r_net.compile(Adam(lr=lr), loss=_loss_func, metrics=['accuracy'])

    _r_net.fit_generator(data_gen,
                         steps_per_epoch=steps_per_epoch,
                         initial_epoch=initial_epoch,
                         epochs=epochs,
                         callbacks=callbacks)
    return _r_net

def train_o_net_with_data_generator(data_gen, steps_per_epoch, initial_epoch=0, epochs=1000, lr=0.001,
                                    callbacks=None, weights_file=None):
    _o_net = o_net(training=True)
    _o_net.summary()
    if weights_file is not None:
        _o_net.load_weights(weights_file)

    _o_net.compile(Adam(lr=lr), loss=_onet_loss_func, metrics=['accuracy'])

    _o_net.fit_generator(data_gen,
                         steps_per_epoch=steps_per_epoch,
                         initial_epoch=initial_epoch,
                         epochs=epochs,
                         callbacks=callbacks)
    return _o_net


def create_callbacks_model_file(net_name, epochs=1):
    filename = datetime.datetime.now().strftime('%Y%m%d_%H%M%S.%f')
    
    log_dir = "{}/{}/{}".format(LOG_DIR, net_name, filename)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    tensor_board = TensorBoard(log_dir=log_dir)
    model_file_path = '{}/{}.h5'.format(MODEL_WEIGHT_SAVE_DIR, net_name+'_weight')

    checkpoint = ModelCheckpoint(model_file_path, verbose=0, save_weights_only=True, period=epochs)
    return [checkpoint, tensor_board], model_file_path

