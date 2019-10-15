import Inception_ResNet_v2 as ir
import tensorflow as tf
import tensorflow.contrib.slim as slim

def inference(input,bottleneck_layer_size,keep=0.8):
    with slim.arg_scope([slim.conv2d],weights_initialializer=slim.initializers.xavier_initializer,
                        activation_fn=tf.nn.relu,normalizer_fn=slim.batch_norm):
        with tf.name_scope('Stem'):
            output = ir.Stem(input)

        with tf.name_scope('5 x Inception-ResNet-A'):
            for i in range(5):
                output = ir.Inception_ResNet_A(output)

        with tf.name_scope('Reduction-A'):
            output = ir.Reduction_A(output)

        with tf.name_scope('10 x Inception-ResNet-B'):
            for i in range(10):
                output = ir.Inception_ResNet_B(output)

        with tf.name_scope('Reduction-B'):
            output = ir.Reduction_B(output)

        with tf.name_scope('5 x Inception-ResNet-C'):
            for i in range(5):
                output = ir.Inception_ResNet_C(output)

        with tf.name_scope('AveragePooling'):
            output = ir.avg_pool(output)
            output = slim.flatten((output))

        with tf.name_scope('Dropout'):
            output = ir.Dropout(output,keep)

        with tf.name_scope('FC'):
            output = ir.fully_connected(output,bottleneck_layer_size)

        return output