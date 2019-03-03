from __future__ import absolute_import, division, print_function

import math
import os
import argparse
import numpy as np
import scipy.misc
import tensorflow as tf
from tensorflow.contrib import layers
from tensorflow.contrib import losses
from tensorflow.contrib.framework import arg_scope
from scipy.misc import imsave
from rae import RAE

import dataLoader
from random import shuffle
import time

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='RAE', help='AE / RAE')
parser.add_argument('--nepoch', type=int, default=10000, help='the number of epochs for training')
parser.add_argument('--batchSize', type=int, default=8, help='input batch size')
parser.add_argument('--learningRate', type=float, default=1e-3, help='learning rate')
parser.add_argument('--inpChannel', type=int, default=7, help='input channel number')
parser.add_argument('--seqLen', type=int, default=7, help='length of training sequence')
parser.add_argument('--scaleW', type=int, default=800, help='the width of the image')
parser.add_argument('--scaleH', type=int, default=425, help='the height of the image')
parser.add_argument('--cropW', type=int, default=128, help='the width of the depth image')
parser.add_argument('--cropH', type=int, default=128, help='the height of the depth image')

parser.add_argument('--store', action='store_false', help='store weights')
parser.add_argument('--saveModelFreq', type=int, default=1000, help='number of iteration to save weights')
parser.add_argument('--saveImgFreq', type=int, default=500, help='number of iteration to save images')

parser.add_argument('--dataPath', required=True, help='path to dataset')
parser.add_argument('--outputPath', required=True, help='path to store images summary and model')
parser.add_argument('--pretrainPath', default='/path/to/pretrained/model', help='path to pretrained model')

parser.add_argument('--gpuId', type=int, default=0, help='the id of gpu used for training.')
parser.add_argument('--manualSeed', type=int, default = None, help = 'the random seed for pytorch')
parser.add_argument('--isRefine', action='store_true', help='whether to refine the network or train from scratch')
parser.add_argument('--epochId', default=0, help='the epoch id of the network load from the file for refinement')

parser.add_argument('--ws', type=float, default=0.8, help='coef of spatial L1 loss')
parser.add_argument('--wg', type=float, default=0.1, help='coef of gradient-domain L1 loss')
parser.add_argument('--wt', type=float, default=0.1, help='coef of temporal L1 loss')
parser.add_argument('--phase', type=str, default='train', help='train / test / predict')

opt = parser.parse_args()

# train on specific gpu
os.environ["CUDA_VISIBLE_DEVICES"]=str(opt.gpuId)

image_directory = os.path.join(opt.outputPath, 'results')
summary_directory = os.path.join(opt.outputPath, 'summary')
model_directory = os.path.join(opt.outputPath, 'model')
for p in [opt.outputPath, image_directory, summary_directory, model_directory]:
    if not os.path.exists(p):
        os.makedirs(p)

restore = opt.isRefine 

if __name__ == "__main__":

    assert opt.model in ['AE', 'RAE']

    # Load Data
    if opt.phase == 'train' or opt.phase == 'test':
        cropSize = (opt.cropH, opt.cropW)
    else:
        cropSize = (416, 800)
    data_loader = dataLoader.BatchLoader(dataRoot = opt.dataPath, inpChannel = opt.inpChannel, batchSize = opt.batchSize, seqLen = opt.seqLen, \
        scaleSize=(opt.scaleH, opt.scaleW), cropSize=cropSize, rseed=opt.manualSeed, phase=opt.phase)

    print('Data Loaded')

    start_time = time.strftime("%Y%m%d%H%M%S", time.localtime())

    # file directory
    model_name = opt.model 
    model_dir = os.path.join(model_directory, model_name)
    log_dir = os.path.join(summary_directory, (model_name +'_'+ start_time))
    img_dir = os.path.join(image_directory, (model_name +'_'+ start_time))
    
    enc_model_dir = os.path.join(model_dir, 'Enc.ckpt')
    dec_model_dir = os.path.join(model_dir, 'Dec.ckpt')

    # initialize model
    model = RAE(opt.model, opt.inpChannel, opt.batchSize, opt.seqLen, opt.learningRate, opt.ws, opt.wg, opt.wt, opt.phase, log_dir)

    # restore the model(parameter)
    ENC_saver = tf.train.Saver(var_list=model.Enc_params)
    DEC_saver = tf.train.Saver(var_list=model.Dec_params)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    if (restore == True) or (opt.phase == 'test') or (opt.phase == 'predict'):
        encPath = os.path.join(opt.pretrainPath, 'Enc.ckpt')
        ENC_saver.restore(model.sess, encPath)
        print("Encoder Model restored in file: %s" % encPath)
        decPath = os.path.join(opt.pretrainPath, 'Dec.ckpt')
        DEC_saver.restore(model.sess, decPath)
        print("Decoder Model restored in file: %s" % decPath)
        
    # start training
    j = 0
    k = 0
    for epoch in range(opt.nepoch):

        while True:
            
            inputNoiseList, imageNoiseOriginalList, inputCleanList, inputCleanOriginalList, isEpoch \
            	= data_loader.loadBatch()

            # update model
            if opt.phase == 'train':
                loss_summary = model.update_params(inputNoiseList, inputCleanList)
                print ("\nEpoch: ", epoch, " / ", opt.nepoch, " ; Iter: ", j, " / ", data_loader.count // opt.batchSize +1)
                # write log to tensorboard
                model.train_writer.add_summary(loss_summary, j+epoch*(data_loader.count // opt.batchSize +1))
            
                # Save weights
                if (k+epoch*(data_loader.count // opt.batchSize +1))% opt.saveModelFreq == opt.saveModelFreq-1:
                    if opt.store == True:
                        os.makedirs(os.path.join(model_dir, str(j)))
                        ENC_save_path = ENC_saver.save(model.sess, model_dir + '/{0}/Enc.ckpt'.format(j))
                        print("Encoder Model saved in file: %s" % ENC_save_path)
                        DEC_save_path = DEC_saver.save(model.sess, model_dir + '/{0}/Dec.ckpt'.format(j))
                        print("Decoder Model saved in file: %s" % DEC_save_path)

            # Save generated images
                if k%opt.saveImgFreq==0:
                    print ('\nsaving compact images')

                    imgs_folder = os.path.join(img_dir, 'imgs')
                    if not os.path.exists(imgs_folder):
                        os.makedirs(imgs_folder)
                    
                    model.generate_and_save_compact_images(opt.batchSize, opt.seqLen, inputNoiseList, inputCleanOriginalList, imgs_folder, epoch, j, opt.phase)
                
            else:
                print ('\nsaving compact images')
                imgs_folder = os.path.join(img_dir, 'imgs')
                if not os.path.exists(imgs_folder):
                    os.makedirs(imgs_folder)
                
                model.generate_and_save_long_videos(opt.batchSize, opt.seqLen, inputNoiseList, inputCleanOriginalList, imgs_folder, epoch, j, opt.phase)

            j += 1
            k += 1

            if isEpoch == True:
                break
