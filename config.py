# -*- coding: utf-8 -*-
"""
Created on Sat Aug 18 10:28:23 2018

@author: yy
"""

NET_SIZE = {
    'p_net': 12,
    'r_net': 24,
    'o_net': 48,
}

NET_NAMES = ['p_net', 'r_net', 'o_net']

LABEL_MAP = {'0': [1, 0], '1': [0, 1], '-1': [0, 0], '-2': [1, 1]}      # neg pos part landmark respectively

GAN_DATA_ROOT_DIR = '/Users/leon/PycharmProjects/BISHE/my_face_detector/data_set'

WIDER_FACE_IMG_DIR = '/Users/leon/Downloads/WIDER_FACE/WIDER_train/images'
WIDER_FACE_ANNO_FILE = '/Users/leon/Downloads/WIDER_FACE/wider_face_split/wider_face_train_bbx_gt.txt'


CELEBA_IMG_DIR = 'G:/BaiduNetdiskDownload/CelebA/Img/img_celeba.7z/img_celeba/'
# G:/BaiduNetdiskDownload/CelebA/Img/img_align_celeba/
CELEBA_ANNO_DIR = 'G:/BaiduNetdiskDownload/CelebA/Anno/'
CELEBA_ANNO_LANDMARKS_FILE = CELEBA_ANNO_DIR + 'list_landmarks_celeba.txt'
CELEBA_ANNO_BBOX_FILE = CELEBA_ANNO_DIR + 'list_bbox_celeba.txt'


MODEL_WEIGHT_SAVE_DIR = '/Users/leon/PycharmProjects/BISHE/my_face_detector/model_weight'
LOG_DIR = '/Users/leon/PycharmProjects/BISHE/my_face_detector/log'
BATCH_SIZE = 64*5       # 64*7 3+1+1+2

#p_net
PNET_VERSION = 2

PNET_EPOCHS = 1
PNET_LEARNING_RATE = 0.001

PNET_CELEBA_NUM = 100002
PNET_WIDEFACE_NUM = 20002

PNET_BACKGRAND_NEG_NUM = 50
PNET_FACE_NEG_NUM = 5       # num for each pic, not for entire dataset
PNET_POS_PART_NUM = 20

#r_net
RNET_VERSION = 1

RNET_EPOCHS = 1
RNET_LEARNING_RATE = 0.001

RNET_CELEBA_NUM = 100002
RNET_WIDEFACE_NUM = 20002

RNET_BACKGRAND_NEG_NUM = 50
RNET_FACE_NEG_NUM = 5
RNET_POS_PART_NUM = 20

#o_net
ONET_EPOCHS= 1
ONET_LEARNING_RATE = 0.001

ONET_CELEBA_NUM = 100002
ONET_WIDEFACE_NUM = 20002

ONET_BACKGRAND_NEG_NUM = 50
ONET_FACE_NEG_NUM = 5
ONET_POS_PART_NUM = 20

#gen_middle_wideface_data
DETECT_WIDEFACE_NUM = 2     # TODO: Changeable param of OHEM(From github comments)


#test
TEST_INPUT_IMG_DIR = '/Users/leon/PycharmProjects/BISHE/my_face_detector/yiyi_mtcnn_keras/test_input_img'
TEST_OUTPUT_IMG_DIR = '/Users/leon/PycharmProjects/BISHE/my_face_detector/yiyi_mtcnn_keras/test_output_img'









