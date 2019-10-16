import os
import numpy as np
import tensorflow as tf
import preprocess
import inference
import tensorflow.contrib.slim as slim
from tensorflow.python.ops import array_ops
from tensorflow.python.framework import ops
from tensorflow.python.ops import data_flow_ops
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

image_path = "D:/Google Cloud/lfw_mtcnnpy_160/"
# image_path = "/content/drive/My Drive/lfw_mtcnnpy_160/"

epoch_size = 1000               # 每个epoch要跑多少个batch
epoch = 10                      # epoch
image_size = (299, 299)         # 图片的大小
batch_size = 32                 # 每个batch的大小
learning_rate = 0.8             # 初始学习率
decay_steps = 100               # 衰减步数
decay_rate = 0.9                # 衰减率
bottleneck_layer_size = 512     #最后一层的输出维度
keep_probability = 0.8          # Dropout参数
weight_decay = 5e-5             # L2权重正则化参数
center_loss_alfa = 0.95         # 中心损失的中心更新率
center_loss_factor = 0.5        # 中心损失权重
train_step = tf.Variable(0, trainable=False)    # 当前训练步数

dataset = preprocess.get_dataset(image_path=image_path)
image_path_list, label_list = preprocess.create_image_path_list_and_label_list(dataset=dataset)

labels = ops.convert_to_tensor(label_list, dtype=tf.int32)
size = array_ops.shape(labels)[0]
index_queue = tf.train.range_input_producer(limit=size, num_epochs=None, shuffle=True, capacity=batch_size)
index_dequeue_op = index_queue.dequeue_many(batch_size)

image_paths_placeholder = tf.placeholder(shape=(None, 1), dtype=tf.string, name="image_paths")
labels_placeholder = tf.placeholder(shape=(None, 1), dtype=tf.int32, name="labels")

nrof_preprocess_threads = 4
input_queue = data_flow_ops.FIFOQueue(capacity=200000, dtypes=[tf.string, tf.int32], shapes=[(1,), (1,)])
enqueue_op = input_queue.enqueue_many([image_paths_placeholder, labels_placeholder], name="enqueue_op")
# image_batch, label_batch, filenames = preprocess.create_input_pipeline(input_queue=input_queue, image_size=image_size, batch_size=batch_size,
#                                                             nrof_preprocess_threads=4, rotate=False, crop=True, flip=True, standardization=False)

images_and_labels_list = preprocess.create_input_pipeline(input_queue, nrof_preprocess_threads,
                                                          image_size=image_size, batch_size=batch_size, rotate=False,
                                                          crop=True, flip=True, standardization=False)

image_batch, label_batch = tf.train.batch_join(images_and_labels_list, batch_size=batch_size,
                                               shapes=[image_size + (3,), ()], enqueue_many=True,
                                               capacity=4 * nrof_preprocess_threads * batch_size,
                                               allow_smaller_final_batch=True)

prelogits = inference.inference(image_batch, bottleneck_layer_size=bottleneck_layer_size, keep=keep_probability)
logits = slim.fully_connected(prelogits, len(dataset), activation_fn=None,
                              weights_initializer=tf.truncated_normal_initializer(stddev=0.1),
                              weights_regularizer=slim.l2_regularizer(weight_decay), scope='Logits', reuse=False)

prelogits_center_loss, _ = preprocess.center_loss(prelogits, label_batch, center_loss_alfa, len(dataset))
tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, prelogits_center_loss * center_loss_factor)

learning_rate = tf.train.exponential_decay(learning_rate, global_step=train_step, decay_steps=decay_steps,
                                           decay_rate=decay_rate, staircase=True)
# tf.summary_scalar('learning_rate', learning_rate)

cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=label_batch, logits=logits,
                                                               name="cross_entropy_per_example")
cross_entropy = tf.reduce_mean(cross_entropy, name='cross_entropy')
tf.add_to_collection('losses', cross_entropy)

correct_prediction = tf.cast(tf.equal(tf.argmax(logits, 1), tf.cast(label_batch, tf.int64)), dtype=tf.float32)
accuracy = tf.reduce_mean(correct_prediction)

regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
total_loss = tf.add_n([cross_entropy] + regularization_losses, name='total_loss')

optimizer = tf.train.RMSPropOptimizer(learning_rate, decay=0.9, momentum=0.9, epsilon=0.1)
grads_and_vars = optimizer.compute_gradients(total_loss, tf.global_variables())
apply_gradient_op = optimizer.apply_gradients(grads_and_vars, global_step=train_step)
with tf.control_dependencies([apply_gradient_op]):
    train_op = tf.no_op(name='train')

config = tf.ConfigProto()
config.gpu_options.allocator_type = "BFC"
with tf.Session(config=config) as sess:
    coord = tf.train.Coordinator()
    tf.train.start_queue_runners(coord=coord)

    index_epoch = sess.run(index_dequeue_op)
    image_path_epoch = np.array(image_path_list)[index_epoch]
    label_epoch = np.array(label_list)[index_epoch]

    image_path_array = np.expand_dims(image_path_epoch, 1)
    label_array = np.expand_dims(label_epoch, 1)
    sess.run(enqueue_op, feed_dict={image_paths_placeholder: image_path_array, labels_placeholder: label_array})
    # for _ in range(nrof_preprocess_threads):
    # 	filenames_tensor, labels_tensor = input_queue.dequeue_many(batch_size)
    # 	filenames, labels = sess.run([filenames_tensor, labels_tensor])
    # 	filenames = np.squeeze(filenames)

    # shapes=[(160, 160, 3), (1)],
    for step in (0, epoch):
        batch_number = 0
        while batch_number < epoch_size:
            print(batch_number)
            image_batch, label_batch = sess.run([image_batch, label_batch])
            print(image_batch.shape)
            print(label_batch.shape)
            accuracy, _ = sess.run([accuracy, train_op])
            print(accuracy)
            batch_number += 1
