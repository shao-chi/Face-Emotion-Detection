from __future__ import print_function
import tensorflow as tf
import math
import numpy as np
import time

time.clock()

x = np.load('../data/new_enhance2_7data.npy')
y = np.load('../data/new_enhance2_7labels.npy')
print('data shape : {}'.format(x.shape))
print('labels shape : {}'.format(y.shape))

x_train = x[:118500, :]/255.
y_train = y[:118500]/1.
x_valid = x[118500:120000, :]/255.
y_valid = y[118500:120000]/1.
x_test = x[120000:,:]/255.
y_test = y[120000:]/1.

print('data_train shape : {}'.format(x_train.shape))
print('labels_train shape : {}'.format(y_train.shape))
print('data_valid shape : {}'.format(x_valid.shape))
print('labels_valid shape : {}\n'.format(y_valid.shape))
print('data_test shape : {}'.format(x_test.shape))
print('labels_test shape : {}'.format(y_test.shape))

train_images = x_train
train_labels = y_train

def weight_variable(shape, name):
    initial = tf.truncated_normal(shape, stddev = 0.1)
    return tf.Variable(initial, name = name)

def bias_variable(shape, name):
    initial = tf.constant(0.1, shape = shape)
    return tf.Variable(initial, name = name)

def conv2d(x, w):
    return tf.nn.conv2d(x, w, strides = [1, 1, 1, 1], padding = 'SAME')

def max_pool_2x2(x):
    return tf.nn.avg_pool(x, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')

def batchnorm(Ylogits, is_test, iteration, offset, convolutional=False):
    exp_moving_avg = tf.train.ExponentialMovingAverage(0.999, iteration) # adding the iteration prevents from averaging across non-existing iterations
    bnepsilon = 1e-5
    if convolutional:
        mean, variance = tf.nn.moments(Ylogits, [0, 1, 2])
    else:
        mean, variance = tf.nn.moments(Ylogits, [0])
    update_moving_everages = exp_moving_avg.apply([mean, variance])
    m = tf.cond(is_test, lambda: exp_moving_avg.average(mean), lambda: mean)
    v = tf.cond(is_test, lambda: exp_moving_avg.average(variance), lambda: variance)
    Ybn = tf.nn.batch_normalization(Ylogits, m, v, offset, None, bnepsilon)
    return Ybn, update_moving_everages

with tf.name_scope('inputs'):
    xs = tf.placeholder(tf.float32, [None, 2304], name = 'xs') # 48*48
    ys = tf.placeholder(tf.float32, [None, 7], name = 'ys')
    lr = tf.placeholder(tf.float32) #For learning rate
    # test flag for batch normalization
    tst = tf.placeholder(tf.bool) 
    iter = tf.placeholder(tf.int32)
    keep_prob = tf.placeholder(tf.float32, name = 'keep_prob')

global_step = tf.Variable(0, name = 'global_step', trainable = False)

def model(xs):
    x_image = tf.reshape(xs, [-1, 48, 48, 1])
    # tf.summary.image('data input', x_image, 5)

    # conv1 layer
    with tf.name_scope('layer_1'):
        with tf.name_scope('weights'):
            w_conv1_1 = weight_variable([3, 3, 1, 64], 'w_conv1')
        tf.summary.histogram('/weights', w_conv1_1)
        with tf.name_scope('bias'):
            b_conv1_1 = bias_variable([64], 'b_conv1')
        tf.summary.histogram('/bias', b_conv1_1)
        with tf.name_scope('outputs'):
            c_conv1_1 = conv2d(x_image, w_conv1_1) + b_conv1_1
            bn_1, update_ema1 = batchnorm(c_conv1_1, tst, iter, b_conv1_1, convolutional=True)
            h_conv1_1 = tf.nn.relu(bn_1)
            h_pool1 = max_pool_2x2(h_conv1_1)
        tf.summary.histogram('/outputs', h_pool1)

    # conv2 layer
    with tf.name_scope('layer_2'):
        with tf.name_scope('weights'):
            w_conv2_1 = weight_variable([3, 3, 64, 128], 'w_conv2')
        tf.summary.histogram('/weights', w_conv2_1)
        with tf.name_scope('bias'):
            b_conv2_1 = bias_variable([128], 'b_conv2')
        tf.summary.histogram('/bias', b_conv2_1)
        with tf.name_scope('outputs'):
            c_conv2_1 = conv2d(h_pool1, w_conv2_1) + b_conv2_1
            bn_2, update_ema2 = batchnorm(c_conv2_1, tst, iter, b_conv2_1, convolutional=True)
            h_conv2_1 = tf.nn.relu(bn_2)
            h_pool2 = max_pool_2x2(h_conv2_1)
        tf.summary.histogram('/outputs', h_pool2)

    # conv3 layer
    with tf.name_scope('layer_3'):
        with tf.name_scope('weights_1'):
            w_conv3_1 = weight_variable([3, 3, 128, 128], 'w_conv3_1')   # ~new5 [1, 1, 64, 64]
        tf.summary.histogram('/weights_1', w_conv3_1)
        with tf.name_scope('bias_1'):
            b_conv3_1 = bias_variable([128], 'b_conv3_1')
        tf.summary.histogram('/bias_1', b_conv3_1)
        with tf.name_scope('outputs_1'):
            c_conv3_1 = conv2d(h_pool2, w_conv3_1) + b_conv3_1
            bn_3_1, update_ema3_1 = batchnorm(c_conv3_1, tst, iter, b_conv3_1, convolutional=True)
            h_conv3_1 = tf.nn.relu(bn_3_1)
            h_pool3 = max_pool_2x2(h_conv3_1)
        # tf.summary.histogram('/outputs_1', h_conv3_1)
        tf.summary.histogram('/outputs_1', h_pool3)

        # with tf.name_scope('weights_2'):
        #     w_conv3_2 = weight_variable([1, 1, 128, 128], 'w_conv3_2')
        # tf.summary.histogram('/weights_2', w_conv3_2)
        # with tf.name_scope('bias_2'):
        #     b_conv3_2 = bias_variable([128], 'b_conv3_2')
        # tf.summary.histogram('/bias_2', b_conv3_2)
        # with tf.name_scope('ioutputs_2'):
        #     c_conv3_2 = conv2d(c_conv3_1, w_conv3_2) + b_conv3_2
        #     bn_3_2, update_ema3_2 = batchnorm(c_conv3_2, tst, iter, b_conv3_2, convolutional=True)
        #     h_conv3_2 = tf.nn.relu(bn_3_2)
        # tf.summary.histogram('/outputs_2', h_conv3_2)

        # with tf.name_scope('weights_3'):
        #     w_conv3_3 = weight_variable([1, 1, 128, 128], 'w_conv3_3')   # ~new5 [3, 3, 64, 64]
        # tf.summary.histogram('/weights_3', w_conv3_3)
        # with tf.name_scope('bias_3'):
        #     b_conv3_3 = bias_variable([128], 'b_conv3_3')
        # tf.summary.histogram('/bias_3', b_conv3_3)
        # with tf.name_scope('outputs_3'):
        #     c_conv3_3 = conv2d(c_conv3_2, w_conv3_3) + b_conv3_3
        #     bn_3_3, update_ema3_3 = batchnorm(c_conv3_3, tst, iter, b_conv3_3, convolutional=True)
        #     h_conv3_3 = tf.nn.relu(bn_3_3)
        #     h_pool3 = max_pool_2x2(h_conv3_3)
        # tf.summary.histogram('/outputs_3', h_pool3)

    # conv4 layer
    with tf.name_scope('layer_4'):
        with tf.name_scope('weights_1'):
            w_conv4_1 = weight_variable([3, 3, 128, 256], 'w_conv4_1')
        tf.summary.histogram('/weights_1', w_conv4_1)
        with tf.name_scope('bias_1'):
            b_conv4_1 = bias_variable([256], 'b_conv4_1')
        tf.summary.histogram('/bias_1', b_conv4_1)
        with tf.name_scope('outputs_1'):
            c_conv4_1 = conv2d(h_pool3, w_conv4_1) + b_conv4_1
            bn_4_1, update_ema4_1 = batchnorm(c_conv4_1, tst, iter, b_conv4_1, convolutional=True)
            h_conv4_1 = tf.nn.relu(bn_4_1)
            h_pool4 = max_pool_2x2(h_conv4_1)
        # tf.summary.histogram('/outputs_1', h_conv4_1)
        tf.summary.histogram('/outputs_1', h_pool4)

        # with tf.name_scope('weights_2'):
        #     w_conv4_2 = weight_variable([3, 3, 256, 256], 'w_conv4_2')
        # tf.summary.histogram('/weights_2', w_conv4_2)
        # with tf.name_scope('bias_2'):
        #     b_conv4_2 = bias_variable([256], 'b_conv4_2')
        # tf.summary.histogram('/bias_2', b_conv4_2)
        # with tf.name_scope('outputs_2'):
        #    c_conv4_2 = conv2d(c_conv4_1, w_conv4_2) + b_conv4_2
        #    bn_4_2, update_ema4_2 = batchnorm(c_conv4_2, tst, iter, b_conv4_2, convolutional=True)
        #    h_conv4_2 = tf.nn.relu(bn_4_2)
        # tf.summary.histogram('/outputs_2', h_conv4_2)

        # with tf.name_scope('weights_3'):
        #    w_conv4_3 = weight_variable([1, 1, 256, 256], 'w_conv4_3')
        # tf.summary.histogram('/weights_3', w_conv4_3)
        # with tf.name_scope('bias_3'):
        #    b_conv4_3 = bias_variable([256], 'b_conv4_3')
        # tf.summary.histogram('/bias_3', b_conv4_3)
        # with tf.name_scope('outputs_3'):
        #    c_conv4_3 = conv2d(c_conv4_2, w_conv4_3) + b_conv4_3
        #    bn_4_3, update_ema4_3 = batchnorm(c_conv4_3, tst, iter, b_conv4_3, convolutional=True)
        #    h_conv4_3 = tf.nn.relu(bn_4_3)
        #    h_pool4 = max_pool_2x2(h_conv4_3)
        # tf.summary.histogram('/outputs_3', h_pool4)

    #func1 layer
    with tf.name_scope('layer_func1'):
        with tf.name_scope('weights'):
            w_f1 = weight_variable([3*3*256, 2048], 'w_f1')  #new5 [256]
        tf.summary.histogram('/weights', w_f1)
        with tf.name_scope('bias'):
            b_f1 = bias_variable([2048], 'b_f1')  #new5 [256]
        tf.summary.histogram('/bias', b_f1)
        with tf.name_scope('outputs'):
            h_pool4_flat = tf.reshape(h_pool4, [-1, 3*3*256])
            h_m1 = tf.matmul(h_pool4_flat, w_f1) + b_f1
            bn_fl, update_ema_f1 = batchnorm(h_m1, tst, iter, b_f1)
            h_f1 = tf.nn.relu(bn_fl)
            h_f1_drop = tf.nn.dropout(h_f1, keep_prob)
        tf.summary.histogram('/outputs', h_f1_drop)

    #func2 layer
    with tf.name_scope('layer_func2'):
        with tf.name_scope('weights'):
            w_f2 = weight_variable([2048, 256], 'w_f2')  #new5 [256]
        tf.summary.histogram('/weights', w_f2)
        with tf.name_scope('bias'):
            b_f2 = bias_variable([256], 'b_f2')
        tf.summary.histogram('/bias', b_f2)
        with tf.name_scope('outputs'):
            h_m2 = tf.matmul(h_f1_drop, w_f2) + b_f2
            bn_f2, update_ema_f2 = batchnorm(h_m2, tst, iter, b_f2)
            h_f2 = tf.nn.relu(bn_f2)
            h_f2_drop = tf.nn.dropout(h_f2, keep_prob)
        tf.summary.histogram('/outputs', h_f2_drop)

    #func2 layer
    with tf.name_scope('layer_func3'):
        with tf.name_scope('weights'):
            w_f3 = weight_variable([256, 7], 'w_f3')  #new5 [256]
        tf.summary.histogram('/weights', w_f3)
        with tf.name_scope('bias'):
            b_f3 = bias_variable([7], 'b_f3')
        tf.summary.histogram('/bias', b_f3)
        with tf.name_scope('outputs'):
            h_m3 = tf.matmul(bn_f2, w_f3) + b_f3
            bn_f3, update_ema_f3 = batchnorm(h_m3, tst, iter, b_f3)
            prediction = tf.nn.softmax(bn_f3, name = 'prediction')
        tf.summary.histogram('/outputs', prediction)


        
        # update_ema = tf.group(update_ema1, update_ema2, update_ema3_1, update_ema3_2, update_ema3_3, update_ema4_1, update_ema4_2, update_ema4_3)
    # return prediction, update_ema1, update_ema2, update_ema3_1, update_ema3_2, update_ema3_3, update_ema4_1, update_ema4_2, update_ema4_3, update_ema_f1, update_ema_f2, update_ema_f3
     return prediction, update_ema1, update_ema2, update_ema3_1, update_ema4_1, update_ema_f1, update_ema_f2, update_ema_f3

prediction, update_ema1, update_ema2, update_ema3_1, update_ema3_2, update_ema3_3, update_ema4_1, update_ema4_2, update_ema4_3, update_ema_f1, update_ema_f2, update_ema_f3 = model(xs)
tf.add_to_collection('outputs', prediction)
update_ema = tf.group(update_ema1, update_ema2, update_ema3_3, update_ema4_3, update_ema_f1, update_ema_f2, update_ema_f3)

# loss
with tf.name_scope('cross_entropy'):
    # cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction), reduction_indices = [1]))
    # cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(tf.clip_by_value(prediction, 1e-10, 1.0)), reduction_indices = [1]))
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=ys, logits=tf.log(tf.clip_by_value(prediction, 1e-10, 1.0))))
    tf.summary.scalar('cross_entropy', cross_entropy)
with tf.name_scope('optimizer'):
    train_step = tf.train.AdamOptimizer(lr).minimize(cross_entropy, global_step = global_step)
    # learning_rate = tf.train.exponential_decay(0.1, global_step, 150, 0.9) 
    # optimizer = tf.train.MomentumOptimizer(learning_rate, 0.9) 
    # train_step = optimizer.minimize(cross_entropy, global_step)

with tf.name_scope('correct_prediction'):
    correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(ys, 1))  
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32)) 
    tf.summary.scalar('accuracy', accuracy)

batch_size = 16
epochs_completed = 0
index_in_epoch = 0
num_examples = x_train.shape[0]

# serve data by batches
def next_batch(batch_size):

    global train_images
    global train_labels
    global index_in_epoch
    global epochs_completed

    start = index_in_epoch
    index_in_epoch += batch_size

    # when all trainig data have been already used, it is reorder randomly
    if index_in_epoch > num_examples:
        # finished epoch
        epochs_completed += 1
        # shuffle the data
        perm = np.arange(num_examples)
        np.random.shuffle(perm)
        train_images = train_images[perm]
        train_labels = train_labels[perm]
        # start next epoch
        start = 0
        index_in_epoch = batch_size
        assert batch_size <= num_examples
    end = index_in_epoch
    return train_images[start:end], train_labels[start:end]

with tf.device('/CPU:0'):
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())

    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter("logs_1/train", sess.graph)
    test_writer = tf.summary.FileWriter("logs_1/test")  
    saver = tf.train.Saver(max_to_keep=2)
    saver_max_acc = 0 

    for i in range(5000001):
        # learning rate decay
        max_learning_rate = 0.05
        min_learning_rate = 0.0001
        decay_speed = 160
        learning_rate = min_learning_rate + (max_learning_rate - min_learning_rate) * math.exp(-i/decay_speed)
        batch_xs, batch_ys = next_batch(batch_size)
        feed_dict = {xs: batch_xs, ys: batch_ys, keep_prob: 0.3, lr: learning_rate, tst: False}
        ab, _, c , summary= sess.run([accuracy, train_step, cross_entropy, merged], feed_dict = feed_dict)
        sess.run(update_ema, {xs: batch_xs, ys: batch_ys, tst: False, iter: i, keep_prob: 1.0})

        if i % 10 == 0:
            a, l , result= sess.run([accuracy, cross_entropy, merged], feed_dict = {xs: x_valid, ys: y_valid, tst: False, iter: i, keep_prob: 1})
            test_writer.add_summary(result, i)
            print('----------- batch accuracy: %.4f%%  loss: %.4f' % (ab*100, c)) 
            print('%d times valid accuracy: %.4f%%  loss: %.4f' % (i/10, a*100, l))

            if a > saver_max_acc:
                saver.save(sess, '../model/model_1/model_1.ckpt', global_step = global_step)
                print('saved\n')
                saver_max_acc = a
        elif i % 100 == 99:
            run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()
            summary, _ = sess.run([merged, train_step], feed_dict = feed_dict, options=run_options, run_metadata=run_metadata)
            train_writer.add_run_metadata(run_metadata, 'step%03d' % i)
            print('Adding run metadata for', i)
        else:  # Record a summary
            summary, _ = sess.run([merged, train_step], feed_dict= feed_dict)
        train_writer.add_summary(summary, i)

    try:
        print('\n test: %.1f%% \n' % sess.run(accuracy, feed_dict = {xs: x_test, ys: y_test, tst: False, iter: i, keep_prob: 1}))
        sess.close()
    except:
        sess.close()
train_writer.close()  
test_writer.close() 
print(time.clock())
