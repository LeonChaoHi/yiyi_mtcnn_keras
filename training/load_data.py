# -*- coding: utf-8 -*-
"""
Created on Mon Aug 20 21:53:08 2018

@author: yy
"""

import os
import pickle
import random

import cv2
import h5py
import numpy as np
import sys
sys.path.append(".")
from utils import load_dict_from_hdf5
from config import LABEL_MAP


class DataGenerator:
    def __init__(self, pos_dataset_path, neg_dataset_path, part_dataset_path, batch_size, im_size):
        self.im_size = im_size
        
        self.pos_file = h5py.File(pos_dataset_path, 'r')
        self.neg_file = h5py.File(neg_dataset_path, 'r')
        self.part_file = h5py.File(part_dataset_path, 'r')

        self.batch_size = batch_size
        
        pos_part_radio = 1.0/5  # samples ratio: neg pos part landmark = 3:1:1:2
        neg_radio = 3.0/5

        self.pos_part_batch_size = int(np.ceil(self.batch_size*pos_part_radio))
        self.neg_batch_size = int(np.ceil(self.batch_size*neg_radio))

        print('pos_part_batch_size---:', self.pos_part_batch_size)
        print('neg_batch_size---:', self.neg_batch_size)

        self.pos_part_start = 0
        self.neg_start = 0
        self.flag = 0

    def _load_pos_dataset(self, end):
        im_batch = self.pos_file['ims'][self.pos_part_start:end]
        labels_batch = self.pos_file['labels'][self.pos_part_start:end]
        bboxes_batch = self.pos_file['bbox'][self.pos_part_start:end]
        return im_batch, labels_batch, bboxes_batch
    
    def _load_neg_dataset(self, end):
        im_batch = self.neg_file['ims'][self.neg_start:end]
        labels_batch = self.neg_file['labels'][self.neg_start:end]
        bboxes_batch = np.zeros((self.neg_batch_size, 4), np.float32)
        return im_batch, labels_batch, bboxes_batch
    
    def _load_part_dataset(self, end):
        im_batch = self.part_file['ims'][self.pos_part_start:end]
        labels_batch = self.part_file['labels'][self.pos_part_start:end]
        bboxes_batch = self.part_file['bbox'][self.pos_part_start:end]
        return im_batch, labels_batch, bboxes_batch

    def generate(self):         # define a generator
        while 1:
            
            pos_part_end = self.pos_part_start + self.pos_part_batch_size
            neg_end = self.neg_start + self.neg_batch_size
            # landmark_end = self.landmark_start + self.landmark_batch_size
            
            # load data with particular batch size into tensors (numpy nd-array)
            im_batch1, labels_batch1, bboxes_batch1 = self._load_pos_dataset(pos_part_end)
            im_batch2, labels_batch2, bboxes_batch2 = self._load_part_dataset(pos_part_end)
            im_batch3, labels_batch3, bboxes_batch3 = self._load_neg_dataset(neg_end)

            x_batch = np.concatenate((im_batch1, im_batch2, im_batch3), axis=0)  # concatenate img tensors
            x_batch = _process_im(x_batch)  # normalize img to [-1, 1]

            label_batch = np.concatenate((labels_batch1, labels_batch2, labels_batch3), axis=0)
            label_batch = label_batch[:, np.newaxis]

            bbox_batch = np.concatenate((bboxes_batch1, bboxes_batch2, bboxes_batch3), axis=0)

            label_batch_shape = label_batch.shape
            bbox_batch_shape = bbox_batch.shape

            # check if 3 annotation tensors have the same batch size
            this_batch_size = max(label_batch_shape[0], bbox_batch_shape[0])
            if label_batch_shape[0] != bbox_batch_shape[0]:
                self.flag = 1
                new_start = random.randrange(1, min(self.pos_part_batch_size, self.neg_batch_size))     # (64, 192)
                
                if label_batch_shape[0] != this_batch_size:
                    self.pos_part_start = int(new_start)
                    self.neg_start = int(new_start)
                if bbox_batch_shape[0] != this_batch_size:
                    self.pos_part_start = int(new_start)
                continue
            
            y_batch = np.concatenate((label_batch, bbox_batch), axis=1)     # shape: [batch_size, 5]

            yield x_batch, y_batch

            self.pos_part_start = pos_part_end      # update start point
            self.neg_start = neg_end

    def steps_per_epoch(self):               # get steps needed to run per epoch depending on batch size and data number
        pos_len = len(self.pos_file['ims'][:])
        neg_len = len(self.neg_file['ims'][:])
        part_len = len(self.part_file['ims'][:])

        pos_total_step = int(pos_len / self.pos_part_batch_size)
        neg_total_step = int(neg_len / self.neg_batch_size)
        part_total_step = int(part_len / self.pos_part_batch_size)

        total_len = min(pos_total_step, neg_total_step, part_total_step)
        
        print('pos_total_step, neg_total_step, part_total_step, landmark_total_step----:', pos_total_step,
              neg_total_step, part_total_step, 0)

        return total_len-12         # TODO: why -12???

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.pos_file.close()
        self.neg_file.close()
        self.part_file.close()


def _load_dataset(dataset_path):    # Unused
    ext = dataset_path.split(os.extsep)[-1]         # get file extension
    if ext == 'pkl':
        with open(dataset_path, 'rb') as f:
            dataset = pickle.load(f)
    elif ext == 'h5':
        dataset = load_dict_from_hdf5(dataset_path)
    else:
        raise ValueError('Unsupported file type, only *.pkl and *.h5 are supported now.')
    return dataset


def _process_im(im):
    return (im.astype(np.float32) - 127.5) / 128        # process training image data


def _process_label(labels):
    label = []
    for ll in labels:
        label.append(LABEL_MAP.get(str(ll)))
    return label
