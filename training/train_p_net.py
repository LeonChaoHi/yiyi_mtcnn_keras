# -*- coding: utf-8 -*-
"""
Created on Mon Aug 20 23:02:01 2018

@author: yy
"""
import os
import sys
# sys.path.append(".")

from config import NET_SIZE, BATCH_SIZE, PNET_EPOCHS, PNET_LEARNING_RATE, GAN_DATA_ROOT_DIR

from training.load_data import _load_dataset, DataGenerator
from training.train_pub import create_callbacks_model_file, train_p_net_with_data_generator

def train_with_data_generator(dataset_root_dir = GAN_DATA_ROOT_DIR, weights_file=None):
    net_name = 'p_net'
    batch_size = BATCH_SIZE
    epochs = PNET_EPOCHS
    learning_rate = PNET_LEARNING_RATE
    
    dataset_dir = os.path.join(dataset_root_dir, net_name)
    pos_dataset_path = os.path.join(dataset_dir, 'pos_shuffle.h5')
    neg_dataset_path = os.path.join(dataset_dir, 'neg_shuffle.h5')
    part_dataset_path = os.path.join(dataset_dir, 'part_shuffle.h5')
    landmarks_dataset_path = os.path.join(dataset_dir, 'landmarks_shuffle.h5')

    # data generator
    data_generator = DataGenerator(pos_dataset_path, neg_dataset_path, part_dataset_path, landmarks_dataset_path,
                                   batch_size, im_size=NET_SIZE['p_net'])
    data_gen = data_generator.generate()
    steps_per_epoch = data_generator.steps_per_epoch()

    callbacks, model_file = create_callbacks_model_file(net_name, 1)  # [checkpoint, tensor_board], model_file_path

    # call p-net training function
    _p_net = train_p_net_with_data_generator(data_gen, steps_per_epoch,
                                             initial_epoch=0,
                                             epochs=epochs,
                                             lr=learning_rate,
                                             callbacks=callbacks,
                                             weights_file=weights_file)
    _p_net.save_weights(model_file)


if __name__ == '__main__':

    train_with_data_generator()
