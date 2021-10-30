#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from __future__ import division, print_function
import tensorflow as tf
import numpy as np


# In[ ]:


def get_mean_stddev(input_tensor):
    with tf.name_scope('mean_stddev_cal'):
        mean, variance = tf.nn.moments(input_tensor, axes=range(len(input_tensor.get_shape())))
        stddev = tf.sqrt(variance, name="standard_deviation")
        return mean, stddev


# In[ ]:


# TODO: Allow shift parameters to be learnable
def get_shifted_stddev(stddev, no_filters):
    with tf.name_scope('shifted_stddev'):
        spreaded_deviation = -1. + (2./(no_filters - 1)) * tf.convert_to_tensor(range(no_filters),
                                                                                dtype=tf.float32)
        return spreaded_deviation * stddev


# In[ ]:


def get_alphas(convolution_filters, binary_filters, no_filters, name=None):
    with tf.name_scope(name, "get_alphas"):
        # Reshaping convolution filters to be one dimensional and binary filters to be of [no_filters, -1] dimension
        reshaped_convolution_filters = tf.reshape(convolution_filters, [-1], name="reshaped_convolution_filters")
        reshaped_binary_filters = tf.reshape(binary_filters, [no_filters, -1],
                                             name="reshaped_binary_filters")
        
        # Creating variable for alphas
        alphas = tf.Variable(tf.random_normal(shape=(no_filters, 1), mean=1.0, stddev=0.1), name="alphas")
        
        # Calculating W*alpha
        weighted_sum_filters = tf.reduce_sum(tf.multiply(alphas, reshaped_binary_filters),
                                             axis=0, name="weighted_sum_filters")
        
        # Defining loss
        error = tf.square(reshaped_convolution_filters - weighted_sum_filters, name="alphas_error")
        loss = tf.reduce_mean(error, axis=0, name="alphas_loss")
        
        # Defining optimizer
        training_op = tf.train.AdamOptimizer().minimize(loss, var_list=[alphas],
                                                        name="alphas_training_op")
        
        return alphas, training_op, loss


# In[ ]:


def get_binary_filters(convolution_filters, no_filters, name=None):
    with tf.name_scope(name, default_name="get_binary_filters"):
        mean, stddev = get_mean_stddev(convolution_filters)
        shifted_stddev = get_shifted_stddev(stddev, no_filters)
        
        # Normalize the filters by subtracting mean from them
        mean_adjusted_filters = convolution_filters - mean
        
        # Tiling filters to match the number of filters
        expanded_filters = tf.expand_dims(mean_adjusted_filters, axis=0, name="expanded_filters")
        tiled_filters = tf.tile(expanded_filters, [no_filters] + [1] * len(convolution_filters.get_shape()),
                                name="tiled_filters")
        
        # Similarly tiling spreaded stddev to match the shape of tiled_filters
        expanded_stddev = tf.reshape(shifted_stddev, [no_filters] + [1] * len(convolution_filters.get_shape()),
                                     name="expanded_stddev")
        
        binarized_filters = tf.sign(tiled_filters + expanded_stddev, name="binarized_filters")
        return binarized_filters


# In[ ]:


def get_alphas(convolution_filters, binary_filters, no_filters, name=None):
    with tf.name_scope(name, "get_alphas"):
        # Reshaping convolution filters to be one dimensional and binary filters to be of [no_filters, -1] dimension
        reshaped_convolution_filters = tf.reshape(convolution_filters, [-1], name="reshaped_convolution_filters")
        reshaped_binary_filters = tf.reshape(binary_filters, [no_filters, -1],
                                             name="reshaped_binary_filters")
        
        # Creating variable for alphas
        alphas = tf.Variable(tf.random_normal(shape=(no_filters, 1), mean=1.0, stddev=0.1), name="alphas")
        
        # Calculating W*alpha
        weighted_sum_filters = tf.reduce_sum(tf.multiply(alphas, reshaped_binary_filters),
                                             axis=0, name="weighted_sum_filters")
        
        # Defining loss
        error = tf.square(reshaped_convolution_filters - weighted_sum_filters, name="alphas_error")
        loss = tf.reduce_mean(error, axis=0, name="alphas_loss")
        
        # Defining optimizer
        training_op = tf.train.AdamOptimizer().minimize(loss, var_list=[alphas],
                                                        name="alphas_training_op")
        
        return alphas, training_op, loss


# In[ ]:


def ApproxConv3D(no_filters, convolution_filters, convolution_biases=None,
               strides=(1, 1, 1), padding="VALID", name=None):
    with tf.name_scope(name, "ApproxConv3D"):
        # Creating variables from input convolution filters and convolution biases
        filters = tf.Variable(convolution_filters, dtype=tf.float32, name="filters")
        if convolution_biases is None:
            biases = 0.
        else:
            biases = tf.Variable(convolution_biases, dtype=tf.float32, name="biases")
        
        # Creating binary filters
        binary_filters = get_binary_filters(filters, no_filters)
        
        # Getting alphas
        alphas, alphas_training_op, alphas_loss = get_alphas(filters, binary_filters,
                                                             no_filters)
        
        # Defining function for closure to accept multiple inputs with same filters
        def ApproxConvLayer(input_tensor, name=None):
            with tf.name_scope(name, "ApproxConv_Layer"):
                # Reshaping alphas to match the input tensor
                reshaped_alphas = tf.reshape(alphas,
                                             shape=[no_filters] + [1] * len(input_tensor.get_shape()),
                                             name="reshaped_alphas")
                
                # Calculating convolution for each binary filter
                approxConv_outputs = []
                for index in range(no_filters):
                    # Binary convolution
                    this_conv = tf.nn.conv3d(input_tensor, binary_filters[index],
                                             strides=(1,) + strides + (1,),
                                             padding=padding)
                    approxConv_outputs.append(this_conv + biases)
                conv_outputs = tf.convert_to_tensor(approxConv_outputs, dtype=tf.float32,
                                                    name="conv_outputs")
                
                # Summing up each of the binary convolution
                ApproxConv_output = tf.reduce_sum(tf.multiply(conv_outputs, reshaped_alphas), axis=0)
                
                return ApproxConv_output
        
        return alphas_training_op, ApproxConvLayer, alphas_loss


# In[ ]:


def ABC(convolution_filters, convolution_biases=None, no_binary_filters=5, no_ApproxConvLayers=5,
        strides=(1, 1, 1), padding="VALID", name=None):
    with tf.name_scope(name, "ABC"):
        # Creating variables shift parameters and weighted sum parameters (betas)
        shift_parameters = tf.Variable(tf.constant(0., shape=(no_ApproxConvLayers, 1)), dtype=tf.float32,
                                       name="shift_parameters")
        betas = tf.Variable(tf.constant(1., shape=(no_ApproxConvLayers, 1)), dtype=tf.float32,
                            name="betas")
        
        # Instantiating the ApproxConv Layer
        alphas_training_op, ApproxConvLayer, alphas_loss = ApproxConv(no_binary_filters,
                                                                      convolution_filters, convolution_biases,
                                                                      strides, padding)
        
        def ABCLayer(input_tensor, name=None):
            with tf.name_scope(name, "ABCLayer"):
                # Reshaping betas to match the input tensor
                reshaped_betas = tf.reshape(betas,
                                            shape=[no_ApproxConvLayers] + [1] * len(input_tensor.get_shape()),
                                            name="reshaped_betas")
                
                # Calculating ApproxConv for each shifted input
                ApproxConv_layers = []
                for index in range(no_ApproxConvLayers):
                    # Shifting and binarizing input
                    shifted_input = tf.clip_by_value(input_tensor + shift_parameters[index], 0., 1.,
                                                     name="shifted_input_" + str(index))
                    binarized_activation = tf.sign(shifted_input - 0.5)
                    
                    # Passing through the ApproxConv layer
                    ApproxConv_layers.append(ApproxConvLayer(binarized_activation))
                ApproxConv_output = tf.convert_to_tensor(ApproxConv_layers, dtype=tf.float32,
                                                         name="ApproxConv_output")
                
                # Taking the weighted sum using the betas
                ABC_output = tf.reduce_sum(tf.multiply(ApproxConv_output, reshaped_betas), axis=0)
                return ABC_output
        
        return alphas_training_op, ABCLayer, alphas_loss


# In[ ]:


def weight_variable(shape, name="weight"):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, name=name)

def bias_variable(shape, name="bias"):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name=name)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')


# In[ ]:


# Creating the graph
without_ABC_graph = tf.Graph()
with without_ABC_graph.as_default():
    # Defining inputs
    x = tf.placeholder(dtype=tf.float32)
    x_image = tf.reshape(x, [-1, 28, 28, 1])
    
     # Convolution Layer 1
    W_conv1 = weight_variable(shape=([5, 5, 1, 32]), name="W_conv1")
    b_conv1 = bias_variable(shape=[32], name="b_conv1")
    conv1 = (conv2d(x_image, W_conv1) + b_conv1)
    pool1 = max_pool_2x2(conv1)
    bn_conv1 = tf.layers.batch_normalization(pool1, axis=-1, name="batchNorm1")
    h_conv1 = tf.nn.relu(bn_conv1)

    # Convolution Layer 2
    W_conv2 = weight_variable(shape=([5, 5, 32, 64]), name="W_conv2")
    b_conv2 = bias_variable(shape=[64], name="b_conv2")
    conv2 = (conv2d(h_conv1, W_conv2) + b_conv2)
    pool2 = max_pool_2x2(conv2)
    bn_conv2 = tf.layers.batch_normalization(pool2, axis=-1, name="batchNorm2")
    h_conv2 = tf.nn.relu(bn_conv2)

    # Flat the conv2 output
    h_conv2_flat = tf.reshape(h_conv2, shape=(-1, 7*7*64))

    # Dense layer1
    W_fc1 = weight_variable([7 * 7 * 64, 1024])
    b_fc1 = bias_variable([1024])
    h_fc1 = tf.nn.relu(tf.matmul(h_conv2_flat, W_fc1) + b_fc1)

    # Dropout
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    # Output layer
    W_fc2 = weight_variable([1024, 10])
    b_fc2 = bias_variable([10])

    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
    
    # Labels
    y = tf.placeholder(tf.int32, [None])
    y_ = tf.one_hot(y, 10)
    
    # Defining optimizer and loss
    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
    # Initializer
    graph_init = tf.global_variables_initializer()


# In[ ]:


# Defining variables to save. These will be fed to our custom layer
variables_to_save = {"W_conv1": W_conv1,
                     "b_conv1": b_conv1,
                     "W_conv2": W_conv2,
                     "b_conv2": b_conv2,
                     "W_fc1": W_fc1,
                     "b_fc1": b_fc1,
                     "W_fc2": W_fc2,
                     "b_fc2": b_fc2}
values = {}


# In[ ]:


n_epochs = 5
batch_size = 32
        
with tf.Session(graph=without_ABC_graph) as sess:
    sess.run(graph_init)
    for epoch in range(n_epochs):
        for iteration in range(1, 200 + 1):
            batch = mnist.train.next_batch(50)
            
            # Run operation and calculate loss
            _, loss_train = sess.run([train_step, cross_entropy],
                                     feed_dict={x: batch[0], y: batch[1], keep_prob: 0.5})
            print("\rIteration: {}/{} ({:.1f}%)  Loss: {:.5f}".format(
                      iteration, 200,
                      iteration * 100 / 200,
                      loss_train),
                  end="")

        # At the end of each epoch,
        # measure the validation loss and accuracy:
        loss_vals = []
        acc_vals = []
        for iteration in range(1, 200 + 1):
            X_batch, y_batch = mnist.validation.next_batch(batch_size)
            acc_val, loss_val = sess.run([accuracy, cross_entropy],
                                     feed_dict={x: batch[0], y: batch[1], keep_prob: 1.0})
            loss_vals.append(loss_val)
            acc_vals.append(acc_val)
            print("\rEvaluating the model: {}/{} ({:.1f}%)".format(iteration, 200,
                iteration * 100 / 200),
                  end=" " * 10)
        loss_val = np.mean(loss_vals)
        acc_val = np.mean(acc_vals)
        print("\rEpoch: {}  Val accuracy: {:.4f}%  Loss: {:.6f}".format(
            epoch + 1, acc_val * 100, loss_val))
        
    # On completion of training, save the variables to be fed to custom model
    for var_name in variables_to_save:
        values[var_name] = sess.run(variables_to_save[var_name])


# In[ ]:


import cv2     # for capturing videos
import math   # for mathematical operations
import matplotlib.pyplot as plt    # for plotting the images
import pandas as pd
from keras.preprocessing import image   # for preprocessing the images
import numpy as np    # for mathematical operations
from keras.utils import np_utils
from skimage.transform import resize   # for resizing images
from sklearn.model_selection import train_test_split
from glob import glob
from tqdm import tqdm
import os


# In[ ]:


img_rows,img_cols,img_depth=100,100,15
path = 'ucf50-3-sample/'


# In[11]:


import tensorflow as tf
import math
from tensorflow.python.training import moving_averages
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.framework import ops


# In[12]:


def Sequential(moduleList):
    def model(x, is_training=True):
    # Create model
        output = x
        #with tf.variable_scope(name,None,[x]):
        for i,m in enumerate(moduleList):
            output = m(output, is_training=is_training)
            tf.add_to_collection(tf.GraphKeys.ACTIVATIONS, output)
        return output
    return model


# In[13]:


def Binarized3D(nOutputPlane, kW, kH, kD, dW=1, dH=1, dD=1,
                padding='VALID', bias=True, reuse=False, name='BinarizedSpatialTemporalConvolution'):
    def b_conv3d(x, is_training=True):
        nInputPlane = x.get_shape().as_list()[3]
        with tf.variable_scope(name, None,[x], reuse=reuse):
            w = tf.get_variable('weight', [kH, kW, nInputPlane, nOutputPlane],
                            initializer=tf.contrib.layers.xavier_initializer_conv2d())
            w = tf.clip_by_value(w,-1,1)
            bin_w = binarize(w)
            bin_x = binarize(x)
            out = tf.nn.conv3d(bin_x, bin_w, strides=[1, dH, dW, dD, 1], padding=padding)
            if bias:
                b = tf.get_variable('bias', [nOutputPlane],initializer=tf.zeros_initializer)
                out = tf.nn.bias_add(out, b)
            return out
    return b_conv3d


# In[14]:


def wrapNN(f,*args,**kwargs):
    def layer(x, scope='', is_training=True):
        return f(x,*args,**kwargs)
    return layer


# In[15]:


def HardTanh(name='HardTanh'):
    def layer(x, is_training=True):
        with tf.variable_scope(name,None,[x]):
            return tf.clip_by_value(x,-1,1)
    return layer


# In[19]:


def SpatialTemporalMaxPooling(kW, kH=None, kD=None, dW=None, dH=None, dD=None, padding='VALID',
            name='SpatialTemporalMaxPooling'):
    kH = kH or kW
    kD = kD or kW
    dW = dW or kW
    dH = dH or kH
    dD = dD or dW or dH
    def max_pool(x,is_training=True):
        with tf.variable_scope(name,None,[x]):
              return tf.nn.max_pool3D(x, ksize=[1, kW, kH, kD, 1], strides=[1, dW, dH, dD, 1], padding=padding)
    return max_pool


# In[31]:


def BatchNormalization(*kargs, **kwargs):
    return wrapNN(tf.compat.v1.layers.batch_normalization, *kargs, **kwargs)


# In[41]:


b3d_1 = Binarized3D(32, 3, 3, 3, 1, 1, 1, padding='VALID', bias=True, reuse=False, name='B3D_1')


# In[43]:


testModel=Sequential([
    Binarized3D(32, 3, 3, 3, 1, 1, 1, padding='VALID', bias=True, reuse=False, name='B3D_1'),
    BatchNormalization(),
    HardTanh(),
    Binarized3D(32, 3, 3, 3, 1, 1, 1, padding='VALID', bias=True, reuse=False, name='B3D_1'),
    SpatialTemporalMaxPooling(3, 3, 3, 1, 1, 1),
    BatchNormalization(),
    HardTanh(),
    BatchNormalization()
])


# In[38]:


testModel.summary()


# In[45]:


def train(model, data,
          batch_size=8,
          learning_rate=0.001,
          log_dir='./log',
          checkpoint_dir='./checkpoint',
          num_epochs=10):

    # tf Graph input
    with tf.device('/cpu:0'):
        with tf.name_scope('data'):
            x, yt = data.generate_batches(batch_size)

        global_step =  tf.get_variable('global_step', shape=[], dtype=tf.int64,
                             initializer=tf.constant_initializer(0),
                             trainable=False)
    if FLAGS.gpu:
        device_str='/gpu:' + str(FLAGS.device)
    else:
        device_str='/cpu:0'
    with tf.device(device_str):
        y = model(x, is_training=True)
        # Define loss and optimizer
        with tf.name_scope('objective'):
            loss = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(labels=yt, logits=y))
            accuracy = tf.reduce_mean(
                tf.cast(tf.nn.in_top_k(y, yt, 1), tf.float32))
        opt = tf.contrib.layers.optimize_loss(loss, global_step, learning_rate, 'Adam',
                                              gradient_noise_scale=None, gradient_multipliers=None,
                                              clip_gradients=None, #moving_average_decay=0.9,
                                              learning_rate_decay_fn=learning_rate_decay_fn, update_ops=None, variables=None, name=None)
        #grads = opt.compute_gradients(loss)
        #apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

    # loss_avg

    ema = tf.train.ExponentialMovingAverage(
        MOVING_AVERAGE_DECAY, global_step, name='average')
    ema_op = ema.apply([loss, accuracy] + tf.trainable_variables())
    tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, ema_op)

    loss_avg = ema.average(loss)
    tf.summary.scalar('loss/training', loss_avg)
    accuracy_avg = ema.average(accuracy)
    tf.summary.scalar('accuracy/training', accuracy_avg)

    check_loss = tf.check_numerics(loss, 'model diverged: loss->nan')
    tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, check_loss)
    updates_collection = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

    with tf.control_dependencies([opt]):
        train_op = tf.group(*updates_collection)

    if FLAGS.summary:
        add_summaries( scalar_list=[accuracy, accuracy_avg, loss, loss_avg],
            activation_list=tf.get_collection(tf.GraphKeys.ACTIVATIONS),
            var_list=tf.trainable_variables())
            # grad_list=grads)

    summary_op = tf.summary.merge_all()

    # Configure options for session
    gpu_options = tf.GPUOptions(allow_growth=True)
    sess = tf.InteractiveSession(
        config=tf.ConfigProto(
            log_device_placement=False,
            allow_soft_placement=True,
            gpu_options=gpu_options,
        )
    )
    saver = tf.train.Saver(max_to_keep=5)

    sess.run(tf.initialize_all_variables())

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    num_batches = data.size[0] / batch_size
    summary_writer = tf.summary.FileWriter(log_dir, graph=sess.graph)
    epoch = 0

    print('num of trainable paramaters: %d' %
          count_params(tf.trainable_variables()))
    while epoch != num_epochs:
        epoch += 1
        curr_step = 0
        # Initializing the variables

        #with tf.Session() as session:
        #    print(session.run(ww))

        print('Started epoch %d' % epoch)
        bar = Bar('Training', max=num_batches,
                  suffix='%(percent)d%% eta: %(eta)ds')
        while curr_step < data.size[0]:
            _, loss_val = sess.run([train_op, loss])
            curr_step += FLAGS.batch_size
            bar.next()

        step, acc_value, loss_value, summary = sess.run(
            [global_step, accuracy_avg, loss_avg, summary_op])
        saver.save(sess, save_path=checkpoint_dir +
                   '/model.ckpt', global_step=global_step)
        bar.finish()
        print('Finished epoch %d' % epoch)
        print('Training Accuracy: %.3f' % acc_value)
        print('Training Loss: %.3f' % loss_value)

        test_acc, test_loss = evaluate(model, FLAGS.dataset,
                                       batch_size=batch_size,
                                       checkpoint_dir=checkpoint_dir)  # ,
        # log_dir=log_dir)
        print('Test Accuracy: %.3f' % test_acc)
        print('Test Loss: %.3f' % test_loss)

        summary_out = tf.Summary()
        summary_out.ParseFromString(summary)
        summary_out.value.add(tag='accuracy/test', simple_value=test_acc)
        summary_out.value.add(tag='loss/test', simple_value=test_loss)
        summary_writer.add_summary(summary_out, step)
        summary_writer.flush()

    # When done, ask the threads to stop.
    coord.request_stop()
    coord.join(threads)
    coord.clear_stop()
    summary_writer.close()


# In[ ]:




