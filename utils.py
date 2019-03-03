import tensorflow as tf
from tensorflow.contrib import layers
from tensorflow.contrib import rnn
#import tensorflow.image as im

def encoderRNN(inputList, batchSize, cell1, cell2, cell3, cell4, cell5, cell6, cropSize, reuse_vars=True):
    '''Create encoder network.

    Args:
        input_tensor: a batch of images [batch_size, 128, 128, n]]
        inh1: hidden state [128, 128]

    Returns:
        neti_out(i=1~5) for skipped connection
        net6_out        for encoder output
    '''
    inpH, inpW = cropSize
    with tf.variable_scope("encoder", reuse=reuse_vars):
        net1_hList = []
        net2_hList = []
        net3_hList = []
        net4_hList = []
        net5_hList = []
        net6_hList = []

        for i, inp in enumerate(inputList):
            if i == 0:
                zeros1 = tf.zeros([2, batchSize, inpH, inpW, 32])
                state1 = rnn.LSTMStateTuple(zeros1[0], zeros1[1]) #[c:hidden state, h:output]
                zeros2 = tf.zeros([2, batchSize, inpH/2, inpW/2, 43])
                state2 = rnn.LSTMStateTuple(zeros2[0], zeros2[1]) #[c:hidden state, h:output]
                zeros3 = tf.zeros([2, batchSize, inpH/4, inpW/4, 57])
                state3 = rnn.LSTMStateTuple(zeros3[0], zeros3[1]) #[c:hidden state, h:output]
                zeros4 = tf.zeros([2, batchSize, inpH/8, inpW/8, 76])
                state4 = rnn.LSTMStateTuple(zeros4[0], zeros4[1]) #[c:hidden state, h:output]
                zeros5 = tf.zeros([2, batchSize, inpH/16, inpW/16, 101])
                state5 = rnn.LSTMStateTuple(zeros5[0], zeros5[1]) #[c:hidden state, h:output]
                zeros6 = tf.zeros([2, batchSize, inpH/32, inpW/32, 101])
                state6 = rnn.LSTMStateTuple(zeros6[0], zeros6[1]) #[c:hidden state, h:output]

            if i > 0:
                tf.get_variable_scope().reuse_variables()

            net1            = layers.conv2d(inp, 32, 3, stride=1, scope="Enc_conv_0") #[128,128,32]
            net1            = layers.conv2d(net1, 32, 3, stride=1, scope="Enc_conv_1") #[128,128,32]
            net1_h, state1  = cell1(net1, state1) #[128, 128, 32]
            net2            = layers.max_pool2d(net1_h, kernel_size=2, stride=2, scope="Enc_maxpool_1") #[64, 64, 32]
            net2            = layers.conv2d(net2, 43, 3, stride=1, scope="Enc_conv_2") #[64,64,43]
            net2_h, state2  = cell2(net2, state2) #[64, 64, 43]
            net3            = layers.max_pool2d(net2_h, kernel_size=2, stride=2, scope="Enc_maxpool_2") #[32, 32, 43]
            net3            = layers.conv2d(net3, 57, 3, stride=1, scope="Enc_conv_3") #[32,32,57]
            net3_h, state3  = cell3(net3, state3) #[32, 32, 57]
            net4            = layers.max_pool2d(net3_h, kernel_size=2, stride=2, scope="Enc_maxpool_3") #[16, 16, 57]
            net4            = layers.conv2d(net4, 76, 3, stride=1, scope="Enc_conv_4") #[16,16,76]
            net4_h, state4  = cell4(net4, state4) #[16, 16, 76]
            net5            = layers.max_pool2d(net4_h, kernel_size=2, stride=2, scope="Enc_maxpool_4") #[8, 8, 76]
            net5            = layers.conv2d(net5, 101, 3, stride=1, scope="Enc_conv_5") #[8,8,101]
            net5_h, state5  = cell5(net5, state5) #[8, 8, 101]
            net6            = layers.max_pool2d(net5_h, kernel_size=2, stride=2, scope="Enc_maxpool_5") #[4, 4, 101]
            net6            = layers.conv2d(net6, 101, 3, stride=1, scope="Enc_conv_6") #[4,4,101]
            net6_h, state6  = cell6(net6, state6) #[4, 4, 101]

            net1_hList.append(net1_h)
            net2_hList.append(net2_h)
            net3_hList.append(net3_h)
            net4_hList.append(net4_h)
            net5_hList.append(net5_h)
            net6_hList.append(net6_h)

    return net1_hList, net2_hList, net3_hList, net4_hList, net5_hList, net6_hList

def encoder(inputList, batchSize, reuse_vars=True):
    '''Create encoder network.

    Args:
        input_tensor: a batch of images [batch_size, 128, 128, n]]
        inh1: hidden state [128, 128]

    Returns:
        neti_out(i=1~5) for skipped connection
        net6_out        for encoder output
    '''
    with tf.variable_scope("encoder", reuse=reuse_vars):
        net1_hList = []
        net2_hList = []
        net3_hList = []
        net4_hList = []
        net5_hList = []
        net6_hList = []

        for i, inp in enumerate(inputList):
            if i > 0:
                tf.get_variable_scope().reuse_variables()

            net1            = layers.conv2d(inp, 32, 3, stride=1, scope="Enc_conv_0") #[128,128,32]
            net1_h          = layers.conv2d(net1, 32, 3, stride=1, scope="Enc_conv_1") #[128,128,32]
            net2            = layers.max_pool2d(net1_h, kernel_size=2, stride=2, scope="Enc_maxpool_1") #[64, 64, 32]
            net2_h          = layers.conv2d(net2, 43, 3, stride=1, scope="Enc_conv_2") #[64,64,43]
            net3            = layers.max_pool2d(net2_h, kernel_size=2, stride=2, scope="Enc_maxpool_2") #[32, 32, 43]
            net3_h          = layers.conv2d(net3, 57, 3, stride=1, scope="Enc_conv_3") #[32,32,57]
            net4            = layers.max_pool2d(net3_h, kernel_size=2, stride=2, scope="Enc_maxpool_3") #[16, 16, 57]
            net4_h            = layers.conv2d(net4, 76, 3, stride=1, scope="Enc_conv_4") #[16,16,76]
            net5            = layers.max_pool2d(net4_h, kernel_size=2, stride=2, scope="Enc_maxpool_4") #[8, 8, 76]
            net5_h            = layers.conv2d(net5, 101, 3, stride=1, scope="Enc_conv_5") #[8,8,101]
            net6            = layers.max_pool2d(net5_h, kernel_size=2, stride=2, scope="Enc_maxpool_5") #[4, 4, 101]
            net6_h            = layers.conv2d(net6, 101, 3, stride=1, scope="Enc_conv_6") #[4,4,101]

            net1_hList.append(net1_h)
            net2_hList.append(net2_h)
            net3_hList.append(net3_h)
            net4_hList.append(net4_h)
            net5_hList.append(net5_h)
            net6_hList.append(net6_h)

    return net1_hList, net2_hList, net3_hList, net4_hList, net5_hList, net6_hList

def decoder(net1_hList, net2_hList, net3_hList, net4_hList, net5_hList, net6_hList, cropSize, reuse_vars=True):
    '''Create decoder network.
    Args:
        net1_out [128,128,32], 
        net2_out [64,64,43], 
        net3_out [32,32,57], 
        net4_out [16,16,76], 
        net5_out [8,8,101], 
        enc_out [4,4,101]

    Returns:
        A tensor that expresses the decoder network
    '''
    inpH, inpW = cropSize
    outputList = []
    with tf.variable_scope("decoder", reuse=reuse_vars):
        for i, inp in enumerate(net6_hList):
            if i > 0:
                tf.get_variable_scope().reuse_variables()
            net = tf.image.resize_nearest_neighbor(inp, [int(inpH/16), int(inpW/16)]) #[8,8,101]
            net = layers.conv2d(tf.concat([net, net5_hList[i]], 3), 76, 3, stride=1, scope="Dec_conv_1_1") #[8,8,76]
            net = layers.conv2d(net, 76, 3, stride=1, scope="Dec_conv_1_2") #[8,8,76]
            net = tf.image.resize_nearest_neighbor(net, [int(inpH/8), int(inpW/8)]) #[16,16,76]
            net = layers.conv2d(tf.concat([net, net4_hList[i]], 3), 57, 3, stride=1, scope="Dec_conv_2_1") #[16,16,57]
            net = layers.conv2d(net, 57, 3, stride=1, scope="Dec_conv_2_2") #[16,16,57]
            net = tf.image.resize_nearest_neighbor(net, [int(inpH/4), int(inpW/4)]) #[32, 32, 57]
            net = layers.conv2d(tf.concat([net, net3_hList[i]], 3), 43, 3, stride=1, scope="Dec_conv_3_1") #[32, 32, 43]
            net = layers.conv2d(net, 43, 3, stride=1, scope="Dec_conv_3_2") #[32, 32, 43]
            net = tf.image.resize_nearest_neighbor(net, [int(inpH/2), int(inpW/2)]) #[64, 64, 43]
            net = layers.conv2d(tf.concat([net, net2_hList[i]], 3), 32, 3, stride=1, scope="Dec_conv_4_1") #[64, 64, 32]
            net = layers.conv2d(net, 32, 3, stride=1, scope="Dec_conv_4_2") #[64, 64, 32]
            net = tf.image.resize_nearest_neighbor(net, [inpH, inpW]) #[128, 128, 32]
            net = layers.conv2d(tf.concat([net, net1_hList[i]], 3), 128, 3, stride=1, scope="Dec_conv_5_1") #[128, 128, 128]
            net = layers.conv2d(net, 64, 3, stride=1, scope="Dec_conv_5_2") #[128, 128, 64]

            output = layers.conv2d(net, 3, 3, stride=1, scope="Dec_conv_6", activation_fn=None) #[128, 128, 3]
            outputList.append(output)

    return outputList


