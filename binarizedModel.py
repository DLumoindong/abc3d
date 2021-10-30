import tensorflow as tf
import math
from tensorflow.python.training import moving_averages
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.framework import ops


def Sequential(moduleList):
    def model(x, is_training=True):
        # Create model
        output = x
        # with tf.variable_scope(name,None,[x]):
        for i, m in enumerate(moduleList):
            output = m(output, is_training=is_training)
            tf.add_to_collection(tf.GraphKeys.ACTIVATIONS, output)
        return output

    return model


def Binarized3D(nOutputPlane, kW, kH, kD, dW=1, dH=1, dD=1,
                padding='VALID', bias=True, reuse=False, name='BinarizedSpatialTemporalConvolution'):
    def b_conv3d(x, is_training=True):
        nInputPlane = x.get_shape().as_list()[3]
        with tf.variable_scope(name, None, [x], reuse=reuse):
            w = tf.get_variable('weight', [kH, kW, nInputPlane, nOutputPlane],
                                initializer=tf.contrib.layers.xavier_initializer_conv2d())
            w = tf.clip_by_value(w, -1, 1)
            bin_w = binarize(w)
            bin_x = binarize(x)
            out = tf.nn.conv3d(bin_x, bin_w, strides=[1, dH, dW, dD, 1], padding=padding)
            if bias:
                b = tf.get_variable('bias', [nOutputPlane], initializer=tf.zeros_initializer)
                out = tf.nn.bias_add(out, b)
            return out

    return b_conv3d


def wrapNN(f, *args, **kwargs):
    def layer(x, scope='', is_training=True):
        return f(x, *args, **kwargs)

    return layer


def HardTanh(name='HardTanh'):
    def layer(x, is_training=True):
        with tf.variable_scope(name, None, [x]):
            return tf.clip_by_value(x, -1, 1)

    return


def SpatialTemporalMaxPooling(kW, kH=None, kD=None, dW=None, dH=None, dD=None, padding='VALID',
                              name='SpatialTemporalMaxPooling'):
    kH = kH or kW
    kD = kD or kW
    dW = dW or kW
    dH = dH or kH
    dD = dD or dW or dH

    def max_pool(x, is_training=True):
        with tf.variable_scope(name, None, [x]):
            return tf.nn.max_pool3D(x, ksize=[1, kW, kH, kD, 1], strides=[1, dW, dH, dD, 1], padding=padding)

    return max_pool


def BatchNormalization(*kargs, **kwargs):
    return wrapNN(tf.nn.batch_normalization, *kargs, **kwargs)


def BinarizedAffine(nOutputPlane, bias=True, name=None, reuse=False):
    def b_affineLayer(x, is_training=True):
        with tf.variable_scope(name, 'Affine', [x], reuse=reuse):
            '''
            Note that we use binarized version of the input (bin_x) and the weights (bin_w). Since the binarized function uses STE
            we calculate the gradients using bin_x and bin_w but we update w (the full precition version).
            '''
            bin_x = binarize(x)
            reshaped = tf.reshape(bin_x, [x.get_shape().as_list()[0], -1])
            nInputPlane = reshaped.get_shape().as_list()[1]
            w = tf.get_variable('weight', [nInputPlane, nOutputPlane],
                                initializer=tf.contrib.layers.xavier_initializer())
            w = tf.clip_by_value(w, -1, 1)
            bin_w = binarize(w)
            output = tf.matmul(reshaped, bin_w)
            if bias:
                b = tf.get_variable('bias', [nOutputPlane], initializer=tf.zeros_initializer)
                output = tf.nn.bias_add(output, b)
        return output

    return b_affineLayer


def train(model, data,
          batch_size=8,
          learning_rate=0.001,
          log_dir='./log',
          checkpoint_dir='./checkpoint',
          num_epochs=10):
    # tf Graph input
    with tf.name_scope('data'):
        x, yt = data.generate_batches(batch_size)

    global_step = tf.get_variable('global_step', shape=[], dtype=tf.int64,
                                  initializer=tf.constant_initializer(0),
                                  trainable=False)
    y = model(x, is_training=True)
    loss = tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(labels=yt, logits=y))
    accuracy = tf.reduce_mean(
        tf.cast(tf.nn.in_top_k(y, yt, 1), tf.float32))
    opt = tf.contrib.layers.optimize_loss(loss, global_step, learning_rate, 'Adam',
                                          gradient_noise_scale=None, gradient_multipliers=None,
                                          clip_gradients=None,  # moving_average_decay=0.9,
                                          learning_rate_decay_fn=learning_rate_decay_fn, update_ops=None,
                                          variables=None, name=None)
    # grads = opt.compute_gradients(loss)
    # apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

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

        # with tf.Session() as session:
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
