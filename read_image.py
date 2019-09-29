import os
import numpy as np
import tensorflow as tf
from tensorflow.python.ops import array_ops
from tensorflow.python.framework import ops
from tensorflow.python.ops import data_flow_ops
import preprocess

image_path = "D:/21TensorFlow/Deep-Learning-21-Examples-master/chapter_6/datasets/casia/casia_maxpy_mtcnnpy_182/"
image_size = (160,160)

dataset = preprocess.get_dataset(image_path=image_path)
image_path_list,label_list = preprocess.create_image_path_list_and_label_list(dataset=dataset)
print(image_path_list[0:12])

labels = ops.convert_to_tensor(label_list, dtype=tf.int32)
size = array_ops.shape(labels)[0]
index_queue = tf.train.range_input_producer(limit=size, num_epochs=None, shuffle=True, capacity=32)
index_dequeue_op = index_queue.dequeue_many(32)

image_paths_placeholder = tf.placeholder(shape=(None, 1), dtype=tf.string, name="image_paths")
labels_placeholder = tf.placeholder(shape=(None, 1), dtype=tf.int32, name="labels")

preprocess_threads = 4
input_queue = data_flow_ops.FIFOQueue(capacity=200000, dtypes=[tf.string, tf.int32], shapes=[(1,),(1,)])
image_batch, label_batch = preprocess.create_input_pipeline(input_queue=input_queue, image_size=image_size, batch_size=32,
                                                            nrof_preprocess_threads=4, rotate=False, crop=True, flip=True, standardization=False)
