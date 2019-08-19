import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim

#######################
# 0. mnist 불러오기
def mnist_load():
    (train_x, train_y), (test_x, test_y) = tf.keras.datasets.mnist.load_data()

    # Train - Image
    train_x = train_x.astype('float32') / 255
    # Train - Label(OneHot)
    train_y = tf.keras.utils.to_categorical(train_y, num_classes=10)

    # Test - Image
    test_x = test_x.astype('float32') / 255
    # Test - Label(OneHot)
    test_y = tf.keras.utils.to_categorical(test_y, num_classes=10)
    
    return (train_x, train_y), (test_x, test_y)


(train_x, train_y), (test_x, test_y) = mnist_load()


#######################
# 1. placeholder 정의
x = tf.placeholder(tf.float32, shape=[None, 28, 28])
y_ = tf.placeholder(tf.float32, shape=[None, 10])
is_training = tf.placeholder(tf.bool)


########################
# 2. TF-Slim을 이용한 CNN 모델 구현
with slim.arg_scope([slim.conv2d],
                    padding='SAME',
                    activation_fn=tf.nn.elu,
                    weights_initializer=tf.truncated_normal_initializer(stddev=0.01)):
    inputs = tf.reshape(x, [-1, 28, 28, 1])
    
    net = slim.conv2d(inputs=inputs, num_outputs=32, kernel_size=[5, 5], scope='conv1')
    net = slim.max_pool2d(inputs=net, kernel_size=[2, 2], scope='pool1')
    net = slim.conv2d(net, 64, [5, 5], scope='conv2')
    net = slim.max_pool2d(net, [2, 2], scope='pool2')
    net = slim.flatten(net, scope='flatten3')
    
with slim.arg_scope([slim.fully_connected],
                    weights_initializer=tf.truncated_normal_initializer(stddev=0.01)):
    net = slim.fully_connected(net, 1024, activation_fn=tf.nn.relu, scope='fc3')
    net = slim.dropout(net, is_training=is_training, scope='dropout3')
    outputs = slim.fully_connected(net, 10, activation_fn=None)
    
########################
# 3. loss, optimizer, accuracy
# loss
cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits_v2(logits=outputs, labels=y_))
# optimizer
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
# accuracy
correct_prediction = tf.equal(tf.argmax(outputs, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

########################
# 4. Hyper-Paramter 설정 및 데이터 설정
# Hyper Parameters
STEPS = 5000
MINI_BATCH_SIZE = 50

# tf.data.Dataset을 이용한 배치 크기 만큼 데이터 불러오기
dataset = tf.data.Dataset.from_tensor_slices(({"image": train_x}, train_y))
dataset = dataset.shuffle(100000).repeat().batch(MINI_BATCH_SIZE)
iterator = dataset.make_one_shot_iterator()
next_batch = iterator.get_next()

########################
# Training & Testing
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    # 학습
    for step in range(STEPS):
        batch_xs, batch_ys = sess.run(next_batch)
        _, cost_val = sess.run([train_step, cross_entropy], feed_dict={x: batch_xs['image'], 
                                                                       y_: batch_ys, 
                                                                       is_training: True})
        
        if (step+1) % 500 == 0:
            train_accuracy = sess.run(accuracy, feed_dict={x: batch_xs['image'],
                                                           y_: batch_ys, 
                                                           is_training: False})
            print("Step : {}, cost : {:.5f}, training accuracy: {:.5f}".format(step+1, cost_val, 
                                                                               train_accuracy))
            
    X = test_x.reshape([10, 1000, 28, 28])
    Y = test_y.reshape([10, 1000, 10])

    test_accuracy = np.mean(
            [sess.run(accuracy, feed_dict={x: X[i], 
                                           y_: Y[i], 
                                           is_training: False}) for i in range(10)])

print("test accuracy: {:.5f}".format(test_accuracy))