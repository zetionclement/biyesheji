import Inception_ResNet_v2 as ir
import tensorflow as tf
import tensorflow.contrib.slim as slim

def inference(input,bottleneck_layer_size,weight_decay,keep=0.8,is_training=True):
    with tf.variable_scope("Inception_ResNet_V2", "Inception_ResNet_V2", reuse=tf.AUTO_REUSE):

        with slim.arg_scope([slim.batch_norm, slim.dropout], is_training=is_training):

            batch_norm_params = {'decay':0.995, 'epsilon':0.001, 'updates_collections':None, 'variables_collections':[tf.GraphKeys.TRAINABLE_VARIABLES],}

            with slim.arg_scope([slim.conv2d,slim.fully_connected], weights_initializer=tf.truncated_normal_initializer(stddev=0.1),
                                weights_regularizer=slim.l2_regularizer(weight_decay), activation_fn=tf.nn.relu, normalizer_fn=slim.batch_norm,
                                normalizer_params=batch_norm_params):

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
                    output = ir.Dropout(output, is_training, keep)

                with tf.name_scope('FC'):
                    output = ir.Fully_connected(output, bottleneck_layer_size)

            return output