import Inception_ResNet_v2 as ir
import tensorflow as tf
import tensorflow.contrib.slim as slim

def inference(input,bottleneck_layer_size,keep=0.8):
    with tf.variable_scope("Inception_ResNet_V2", "Inception_ResNet_V2", reuse=tf.AUTO_REUSE):
        with slim.arg_scope([slim.conv2d], weights_initializer=slim.initializers.xavier_initializer(),
                            activation_fn=tf.nn.relu, normalizer_fn=slim.batch_norm):
            with tf.name_scope('Stem'):
                output = ir.Stem(input)

            with tf.name_scope('5_x_Inception_ResNet_A'):
                for i in range(5):
                    output = ir.Inception_ResNet_A(output)

            with tf.name_scope('Reduction_A'):
                output = ir.Reduction_A(output)

            with tf.name_scope('10_x_Inception_ResNet_B'):
                for i in range(10):
                    output = ir.Inception_ResNet_B(output)

            with tf.name_scope('Reduction_B'):
                output = ir.Reduction_B(output)

            with tf.name_scope('5_x_Inception_ResNet_C'):
                for i in range(5):
                    output = ir.Inception_ResNet_C(output)

            with tf.name_scope('AveragePooling'):
                output = ir.Avg_pool(output)
                output = slim.flatten((output))

            with tf.name_scope('Dropout'):
                output = ir.Dropout(output, keep)

            with tf.name_scope('FC'):
                output = ir.Fully_connected(output, bottleneck_layer_size)

            return output