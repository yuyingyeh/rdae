from __future__ import absolute_import, division, print_function

import math
import numpy as np
import tensorflow as tf
from tensorflow.contrib import layers
from tensorflow.contrib.framework import arg_scope
from tensorflow.contrib import rnn

from utils import encoder, encoderRNN, decoder
from model import Model

class RAE(Model):

    def __init__(self, model, channel_num, batch_size, seq_len, learning_rate, ws, wg, wt, phase, sum_dir):
        if phase == 'train' or phase == 'test':
            self.inputNoiseList = [tf.placeholder(tf.float32, [batch_size, 128, 128, channel_num])\
                for _ in range(seq_len)]
            self.inputCleanList = [tf.placeholder(tf.float32, [batch_size, 128, 128, 3])\
                for _ in range(seq_len)]
        else:
            self.inputNoiseList = [tf.placeholder(tf.float32, [batch_size, 416, 800, channel_num])\
                for _ in range(seq_len)]
            self.inputCleanList = [tf.placeholder(tf.float32, [batch_size, 416, 800, 3])\
                for _ in range(seq_len)]

        with arg_scope([layers.conv2d],
                       activation_fn=tf.nn.leaky_relu,
                       #normalizer_fn=layers.batch_norm,
                       normalizer_params={'scale': True},
                       padding='SAME'):
            with tf.variable_scope("model") as scope: #Full VAEGAN structure
                if phase == 'train' or phase == 'test':
                    inpH, inpW = 128, 128
                else:
                    inpH, inpW = 416, 800
                if model == 'RAE':
                    with tf.name_scope("initalize_RNN_cell"):
                        cell1 = rnn.ConvLSTMCell(2, [inpH, inpW, 32], 32, [3,3], name = 'rnn1')
                        cell2 = rnn.ConvLSTMCell(2, [inpH/2, inpW/2, 43], 43, [3,3], name = 'rnn2')
                        cell3 = rnn.ConvLSTMCell(2, [inpH/4, inpW/4, 57], 57, [3,3], name = 'rnn3')
                        cell4 = rnn.ConvLSTMCell(2, [inpH/8, inpW/8, 76], 76, [3,3], name = 'rnn4')
                        cell5 = rnn.ConvLSTMCell(2, [inpH/16, inpW/16, 101], 101, [3,3], name = 'rnn5')
                        cell6 = rnn.ConvLSTMCell(2, [inpH/32, inpW/32, 101], 101, [3,3], name = 'rnn6')

                    # Encoder
                    l1, l2, l3, l4, l5, out = encoderRNN(self.inputNoiseList, batch_size, cell1, cell2, cell3, \
                        cell4, cell5, cell6, (inpH, inpW), reuse_vars=False)
                elif model == "AE":
                    l1, l2, l3, l4, l5, out = encoder(self.inputNoiseList, batch_size, reuse_vars=False)
                Enc_params_num       = len(tf.trainable_variables())

                # Decoder / Generator
                self.denoised_imgList   = decoder(l1, l2, l3, l4, l5, out, (inpH, inpW), reuse_vars=False)
                Enc_n_Dec_params_num = len(tf.trainable_variables())

        self.params     = tf.trainable_variables()
        self.Enc_params = self.params[:Enc_params_num]
        self.Dec_params = self.params[Enc_params_num:Enc_n_Dec_params_num]
        print(len(self.params))
        for var in self.params:
            print(var.name)

        self.Spatial_loss = self.__get_L1_loss(self.denoised_imgList, self.inputCleanList)
        Spatial_loss_sum  = tf.summary.scalar('Spatial_loss', self.Spatial_loss)
        self.Gradient_loss = self.__get_grad_L1_loss(self.denoised_imgList, self.inputCleanList)
        Gradient_loss_sum  = tf.summary.scalar('Gradient_loss', self.Gradient_loss)
        if model == 'RAE':
            self.Temporal_loss = self.__get_tem_L1_loss(self.denoised_imgList, self.inputCleanList)
            Temporal_loss_sum = tf.summary.scalar('Temporal_loss', self.Temporal_loss)
            # merge  summary for Tensorboard
            self.detached_loss_summary_merged          =  tf.summary.merge([Spatial_loss_sum, Gradient_loss_sum, Temporal_loss_sum])
            # loss function
            total_loss = ws * self.Spatial_loss + wg * self.Gradient_loss + wt * self.Temporal_loss

        elif model == 'AE':
            self.detached_loss_summary_merged          =  tf.summary.merge([Spatial_loss_sum, Gradient_loss_sum])
            # loss function
            total_loss = ws * self.Spatial_loss + wg * self.Gradient_loss

        # self.train     = layers.optimize_loss(total_loss, tf.train.get_or_create_global_step(\
        #     ), learning_rate=learning_rate, variables = self.params, optimizer='RMSProp', update_ops=[])

        self.train     = tf.train.AdamOptimizer(learning_rate=learning_rate, 
                                        beta1=0.9, 
                                        beta2=0.99, 
                                        epsilon=1e-08, 
                                        name='Adam').minimize(total_loss, var_list=self.params)

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        self.sess.run(tf.global_variables_initializer())
        #.replace('\\','/')
        self.train_writer =  tf.summary.FileWriter(sum_dir, self.sess.graph)

    def __preprocess(self, inp):
        return inp

    def __get_L1_loss(self, outputList, targetList):
        lossSum = 0
        for i, out in enumerate(outputList):
            out = self.__preprocess(out)
            target = self.__preprocess(targetList[i])
            lossSum += tf.reduce_mean(tf.abs(out - target))
        return lossSum/len(outputList)

    def __get_grad_L1_loss(self, outputList, targetList):
        lossSum = 0
        for i, out in enumerate(outputList):
            out = self.__preprocess(out)
            target = self.__preprocess(targetList[i])
            lossSum += tf.reduce_mean(tf.abs(self.__applyGrad(out) - self.__applyGrad(target)))
        return lossSum/len(outputList)

    def __get_tem_L1_loss(self, outputList, targetList):
        lossSum = 0
        for i, out in enumerate(outputList):
            out = self.__preprocess(out)
            target = self.__preprocess(targetList[i])
            if i == 0:
                outPrev = out 
                targetPrev = target
                continue
            outDiff = out - outPrev
            tarDiff = target - targetPrev
            lossSum += tf.reduce_mean(tf.abs(outDiff - tarDiff))
            outPrev = out 
            targetPrev = target
        return lossSum/(len(outputList)-1)

    def __applyGrad(self, inp):
        #print(self.LoG_filter())
        LoG_filter = tf.convert_to_tensor(self.LoG_filter(), np.float32) #[15 15]
        LoG_filter = tf.expand_dims(tf.expand_dims(LoG_filter, -1), -1)
        LoG_filter = tf.tile(LoG_filter, [1, 1, 3, 1])
        # Convolve.
        return tf.nn.conv2d(inp, LoG_filter, strides=[1, 1, 1, 1], padding="SAME")

    def LoG_filter(self):
        nx, ny = (15, 15)
        x = np.linspace(-7, 7, 15)
        y = np.linspace(-7, 7, 15)
        xv, yv = np.meshgrid(x, y)
        output = self.LoG(xv, yv, 1.5)
        return(output)

    def LoG(self, X, Y, sig):
        pi = math.pi
        return -1/(pi*sig**4)*(1-(X**2+Y**2)/(2*sig**2))*np.exp(-(X**2+Y**2)/(2*sig**2))

    def update_params(self, inputNoiseList, inputCleanList):
        noiseList = {i1: d1 for i1, d1 in zip(self.inputNoiseList, inputNoiseList)}
        cleanList = {i2: d2 for i2, d2 in zip(self.inputCleanList, inputCleanList)}
        _, loss_summary = self.sess.run([self.train, self.detached_loss_summary_merged], \
            {**noiseList, **cleanList})
        return loss_summary
