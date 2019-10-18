import tensorflow as tf
import tensorflow.contrib.slim as slim

def Stem(input):
    with tf.variable_scope("Stem", "Stem", reuse=tf.AUTO_REUSE):
        net = slim.conv2d(input, 32, [3, 3], 2, padding="VALID", scope="Conv1")
        net = slim.conv2d(net, 32, [3, 3], padding="VALID", scope="Conv2")
        net = slim.conv2d(net, 64, [3, 3], scope="Conv3")
        net_left = slim.max_pool2d(net, [3, 3], 2, padding="VALID", scope="LeftMaxpool1")
        net_right = slim.conv2d(net, 96, [3, 3], 2, padding="VALID", scope="RightConv1")
        net = tf.concat([net_left, net_right], 3, name="Concat1")

        net_left = slim.conv2d(net, 64, [1, 1], scope="LeftConv1")
        net_left = slim.conv2d(net_left, 96, [3, 3], padding="VALID", scope="LeftConv2")
        net_right = slim.conv2d(net, 64, [1, 1], scope="RightConv2")
        net_right = slim.conv2d(net_right, 64, [7, 1], scope="RightConv3")
        net_right = slim.conv2d(net_right, 64, [1, 7], scope="RightConv4")
        net_right = slim.conv2d(net_right, 96, [3, 3], padding="VALID", scope="RightConv5")
        net = tf.concat([net_left, net_right], axis=3, name="Concat2")

        net_left = slim.conv2d(net, 192, [3, 3], 2, padding="VALID", scope="LeftConv3")
        net_right = slim.max_pool2d(net, [2, 2], 2, padding="VALID", scope="RightMaxpool1")
        net = tf.concat([net_left, net_right], axis=3, name="Concat3")
        net = tf.nn.relu(net)

    return net


def Inception_ResNet_A(input):
    with tf.variable_scope("Inception_ResNet_A", "Inception_ResNet_A", reuse=tf.AUTO_REUSE):
        res_net = tf.identity(input)

        net_left = slim.conv2d(input, 32, [1, 1], scope="LeftConv1")

        net_mid = slim.conv2d(input, 32, [1, 1], scope="MidConv1")
        net_mid = slim.conv2d(net_mid, 32, [3, 3], scope="MidConv2")

        net_right = slim.conv2d(input, 32, [1, 1], scope="RightConv1")
        net_right = slim.conv2d(net_right, 48, [3, 3], scope="RightConv2")
        net_right = slim.conv2d(net_right, 64, [3, 3], scope="RightConv3")

        net = tf.concat([net_left, net_mid, net_right], axis=3)
        net = slim.conv2d(net, input.get_shape()[3], [1, 1], scope="ConcatAndConv")
        net = tf.multiply(net, 0.1)
        net = tf.clip_by_value(net, -6.0, 6.0)
        net = tf.nn.relu(tf.add_n([net, res_net]))

    return net


def Reduction_A(input):
    with tf.variable_scope("Reduction_A", "Reduction_A", reuse=tf.AUTO_REUSE):
        net_left = slim.max_pool2d(input, [3, 3], 2, padding="VALID", scope="LeftMaxpool")

        net_mid = slim.conv2d(input, 384, [3, 3], 2, padding="VALID", scope="MidConv")

        net_right = slim.conv2d(input, 256, [1, 1], scope="RightConv1")
        net_right = slim.conv2d(net_right, 256, [3, 3], scope="RightConv2")
        net_right = slim.conv2d(net_right, 384, [3, 3], 2, padding="VALID", scope="RightConv3")

        net = tf.concat([net_left, net_mid, net_right], axis=3)
        net = tf.nn.relu(net)

    return net


def Inception_ResNet_B(input):
    with tf.variable_scope("Inception_ResNet_B","Inception_ResNet_B",reuse=tf.AUTO_REUSE):
        res_net = tf.identity(input)

        net_left = slim.conv2d(input, 192, [1, 1], scope="LeftConv")

        net_right = slim.conv2d(input, 128, [1, 1], scope="RightConv1")
        net_right = slim.conv2d(net_right, 160, [1, 7], scope="RightConv2")
        net_right = slim.conv2d(net_right, 192, [7, 1], scope="RightConv3")

        net = tf.concat([net_left, net_right], axis=3)
        net = slim.conv2d(net, input.get_shape()[3], [1, 1], scope="ConcatAndConv")
        net = tf.multiply(net, 0.1)
        net = tf.clip_by_value(net, -6.0, 6.0)
        net = tf.nn.relu(tf.add_n([net, res_net]))

    return net


def Reduction_B(input):
    with tf.variable_scope("Reduction_B", "Reduction_B", reuse=tf.AUTO_REUSE):
        net_left = slim.max_pool2d(input, [3, 3], 2, padding="VALID", scope="LeftMaxpool")

        net_mid_left = slim.conv2d(input, 256, [1, 1], scope="MidLeftConv1")
        net_mid_left = slim.conv2d(net_mid_left, 384, [3, 3], 2, padding="VALID", scope="MidLeftConv2")

        net_mid_right = slim.conv2d(input, 256, [1, 1], scope="MidRightConv1")
        net_mid_right = slim.conv2d(net_mid_right, 288, [3, 3], 2, padding="VALID", scope="MidRightConv2")

        net_right = slim.conv2d(input, 256, [1, 1], scope="RightConv1")
        net_right = slim.conv2d(net_right, 288, [3, 3], scope="RightConv2")
        net_right = slim.conv2d(net_right, 320, [3, 3], 2, padding="VALID", scope="RightConv3")

        net = tf.concat([net_left, net_mid_left, net_mid_right, net_right], axis=3)
        net = tf.nn.relu(net)

    return net


def Inception_ResNet_C(input):
    with tf.variable_scope("Inception_ResNet_C", "Inception_ResNet_C", reuse=tf.AUTO_REUSE):
        res_net = tf.identity(input)

        net_left = slim.conv2d(input, 192, [1, 1], scope="LeftConv1")

        net_right = slim.conv2d(input, 192, [1, 1], scope="RightConv1")
        net_right = slim.conv2d(net_right, 224, [1, 3], scope="RightConv2")
        net_right = slim.conv2d(net_right, 256, [3, 1], scope="RightConv3")

        net = tf.concat([net_left, net_right], axis=3)
        net = slim.conv2d(net, input.get_shape()[3], [1, 1], scope="ConcatAndConv")
        net = tf.multiply(net, 0.1)
        net = tf.clip_by_value(net, -6.0, 6.0)
        net = tf.nn.relu(tf.add_n([net, res_net]))

    return net

def Avg_pool(input):
    with tf.variable_scope("Avg_pool", "Avg_pool", reuse=tf.AUTO_REUSE):
        output = slim.avg_pool2d(input, [8, 8], scope="Average_Pooling")
    return output

def Dropout(input,keep=0.8):
    with tf.variable_scope("Dropout", "Dropout", reuse=tf.AUTO_REUSE):
        output = slim.dropout(input, keep_prob=keep)
    return output

def Fully_connected(input,bottleneck_layer_size):
    with tf.variable_scope("Fully_connected", "Fully_connected", reuse=tf.AUTO_REUSE):
        output = slim.fully_connected(input, bottleneck_layer_size, activation_fn=None, reuse=False)
    return output

