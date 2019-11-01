import os
import numpy as np
import time
import sys
import argparse
import tensorflow as tf
import preprocess
import inference
import tensorflow.contrib.slim as slim
from datetime import datetime
from tensorflow.python.ops import array_ops
from tensorflow.python.framework import ops
from tensorflow.python.ops import data_flow_ops
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

tf.reset_default_graph()


# image_path = "/home/dc2-user/biyesheji/lfw_mtcnnpy_160/"
image_path = "/home/dc2-user/biyesheji/casia/casia_maxpy_mtcnnpy_299/"

model_path = "/home/dc2-user/biyesheji/models/"                         # 模型保存的路径
summary_base_path = "/home/dc2-user/biyesheji/summary/"                 # summary保存路径
learning_rate_path = '/home/dc2-user/biyesheji/learning_rate.txt'       # 学习率文件路径
dataset_type = "casia"                                                  # 人脸数据集类型，可改为casia或者lfw
log_histogram = True                                                    # 是否对weights/bias采用直方图来记录变化 
epochs = 350                                                            # epoch
epoch_size = 1000                                                       # 每个epoch要跑多少个batch
save_batch = 500                                                        # 每个epoch中要跑多少个batch才保存一次模型
image_size = (299, 299)                                                 # 图片的大小
batch_size = 15                                                         # 每个batch的大小
learning_rate = 0.001                                                   # 初始学习率
learning_rate_dcay_epochs = 10                                          # 经过n个epoch后对学习率进行一次衰减
decay_steps = learning_rate_dcay_epochs * epoch_size                    # 训练decay_steps步后对学习率进行一次衰减
decay_rate = 0.99                                                       # 学习率的衰减速度
moving_average_decay_rate = 0.99                                        # 滑动平均衰减率
bottleneck_layer_size = 512                                             # 最后一层的输出维度
keep_probability = 0.8                                                  # Dropout参数
weight_decay = 5e-5                                                     # L2权重正则化参数
center_loss_alfa = 0.95                                                 # 中心损失的中心更新率
center_loss_factor = 0.5                                                # 中心损失权重
train_step = tf.Variable(0, trainable=False)                            # 当前训练步数
pretrained_model_path = "/home/dc2-user/biyesheji/models/"              # 之前训练的模型的路径
pretrained_model = False                                                # 是否有已训练过的模型


if len(os.listdir(pretrained_model_path)) > 0:
	pretrained_model = True
	print("Using pretrained model")

dataset = preprocess.get_dataset(image_path, dataset_type)
image_path_list, label_list = preprocess.create_image_path_list_and_label_list(dataset=dataset)

labels = ops.convert_to_tensor(label_list, dtype=tf.int32)
size = array_ops.shape(labels)[0]
index_queue = tf.train.range_input_producer(limit=size, num_epochs=None, shuffle=True, capacity=32)
index_dequeue_op = index_queue.dequeue_many(batch_size * epoch_size)

image_paths_placeholder = tf.placeholder(shape=(None, 1), dtype=tf.string, name="image_paths")
labels_placeholder = tf.placeholder(dtype=tf.int32, name="labels")
learning_rate_placeholder = tf.placeholder(dtype=tf.float64, name="learning_rate")
is_training_placeholder = tf.placeholder(dtype=tf.bool, name="is_training")

nrof_preprocess_threads = 4
input_queue = data_flow_ops.FIFOQueue(capacity=2000000, dtypes=[tf.string, tf.int32], shapes=[(1,), (1,)])
enqueue_op = input_queue.enqueue_many([image_paths_placeholder, labels_placeholder], name="enqueue_op")

images_and_labels_list = preprocess.create_input_pipeline(input_queue, nrof_preprocess_threads,
                                                          image_size=image_size, batch_size=batch_size, rotate=False,
                                                          crop=True, flip=True, standardization=False)

image_batch, label_batch = tf.train.batch_join(images_and_labels_list, batch_size=batch_size,
                                               shapes=[image_size + (3,), ()], enqueue_many=True,
                                               capacity=4 * nrof_preprocess_threads * batch_size,
                                               allow_smaller_final_batch=True)

image_batch = tf.identity(image_batch, name='image_batch')
image_batch = tf.identity(image_batch, name='input')
label_batch = tf.identity(label_batch, name='label_batch')

prelogits = inference.inference(image_batch, bottleneck_layer_size=bottleneck_layer_size, weight_decay=weight_decay, is_training=is_training_placeholder, keep=keep_probability)
logits = slim.fully_connected(prelogits, len(dataset), activation_fn=None,
                              weights_initializer=tf.truncated_normal_initializer(stddev=0.1),
                              weights_regularizer=slim.l2_regularizer(weight_decay), scope='Logits', reuse=False)

embeddings = tf.nn.l2_normalize(prelogits, 1, 1e-10, name='embeddings')

prelogits_center_loss, _ = preprocess.center_loss(prelogits, label_batch, center_loss_alfa, len(dataset))
tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, prelogits_center_loss * center_loss_factor)

learning_rate = tf.train.exponential_decay(learning_rate_placeholder, global_step=train_step, decay_steps=decay_steps,
                                           decay_rate=decay_rate, staircase=True)
tf.summary.scalar('learning_rate', learning_rate)

cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=label_batch, logits=logits, name="cross_entropy_per_example")
cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
tf.add_to_collection('losses', cross_entropy_mean)

regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)  # regularization_losses包括中心损失和L2权重正则化损失
total_loss = tf.add_n([cross_entropy_mean] + regularization_losses, name='total_loss') # total_loss包括交叉熵和regularization_losses

total_loss_average_op = preprocess.moving_average_total_loss(total_loss)

optimizer = tf.train.RMSPropOptimizer(learning_rate, decay=0.9, momentum=0.9, epsilon=0.1)
grads_and_vars = optimizer.compute_gradients(total_loss, tf.global_variables())
apply_gradient_op = optimizer.apply_gradients(grads_and_vars, global_step=train_step)

tf.summary.scalar('loss', total_loss)

variable_average_op = preprocess.summary_all_variables_and_gradient(grads_and_vars, tf.trainable_variables(), moving_average_decay_rate, 
                      log_histogram, train_step)

with tf.control_dependencies([total_loss_average_op, variable_average_op, apply_gradient_op]):
  # train_op = tf.train.RMSPropOptimizer(learning_rate, decay=0.9, momentum=0.9, epsilon=0.1).minimize(total_loss,global_step=train_step)
  train_op = tf.no_op(name='train_op')

saver = tf.train.Saver(tf.trainable_variables(), max_to_keep=3)

summary_op = tf.summary.merge_all()


config = tf.ConfigProto()
config.gpu_options.allocator_type = "BFC"

with tf.Session(config=config) as sess:

    epoch_start = 0
    batch_number_start = 0

    if pretrained_model:
        print("Restoring pretrained model")
        model_file = tf.train.get_checkpoint_state(pretrained_model_path).model_checkpoint_path
        current_step = model_file.split('/')[-1].split('.')[1].split('-')[1]
        epoch_start = int(current_step) // 1000
        batch_number_start = int(current_step) % 1000    
        saver.restore(sess, model_file)

    tf.global_variables_initializer().run()
    coord = tf.train.Coordinator()
    tf.train.start_queue_runners(coord=coord)
    date_time = time.localtime(time.time())
    summary_save_time = str(date_time.tm_year) + str(date_time.tm_mon) + str(date_time.tm_mday)
    summary_path = os.path.join(summary_base_path, summary_save_time)
    if not os.path.exists(summary_path):
      os.makedirs(summary_path)
    summary_writer = tf.summary.FileWriter(summary_path, sess.graph)
    
    begin_time = time.localtime(time.time())

    for epoch in range(epoch_start,epochs):
        index_epoch = sess.run(index_dequeue_op)
        image_path_epoch = np.array(image_path_list)[index_epoch]
        label_epoch = np.array(label_list)[index_epoch]

        image_path_array = np.expand_dims(image_path_epoch, 1)
        label_array = np.expand_dims(label_epoch, 1)

        lr = preprocess.get_learning_rate_from_file(learning_rate_path, epoch)
        sess.run(enqueue_op, feed_dict={image_paths_placeholder: image_path_array, labels_placeholder: label_array})

        for batch_number in range(batch_number_start, epoch_size):
            start_time = time.time()
            _, _toal_loss, _regular_loss, summary_str, step  = sess.run([train_op, total_loss, regularization_losses, summary_op, train_step], 
                                                                          feed_dict={learning_rate_placeholder:lr, is_training_placeholder:True})
            duration = time.time() - start_time
            print("epoch[%d][%d], time:%.3f, total_loss:%.3f, regularization_loss:%.3f"%(epoch, batch_number, duration, _toal_loss, np.sum(_regular_loss)))
            summary_writer.add_summary(summary_str, global_step=step)
            if batch_number % save_batch == 0 and batch_number > 0:
              start_time = time.time()
              current_time = datetime.strftime(datetime.now(), '%Y-%m-%d_%H_%M_%S')
              model_name = os.path.join(model_path,'model-%s.ckpt'%(current_time))
              saver.save(sess, model_name, global_step=step, write_meta_graph=True)
              duration = time.time() - start_time
              print("Model saved in %.3f seconds"%(duration))

    end_time = time.localtime(time.time())

    print("Begin time : %d-%d-%d %d:%d:%d"%(begin_time.tm_year, begin_time.tm_mon, begin_time.tm_mday, begin_time.tm_hour, begin_time.tm_min, begin_time.tm_sec))
    print("End time : %d-%d-%d %d:%d:%d"%(end_time.tm_year, end_time.tm_mon, end_time.tm_mday, end_time.tm_hour, end_time.tm_min, end_time.tm_sec))

