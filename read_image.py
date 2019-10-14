import os
import numpy as np
import tensorflow as tf
from tensorflow.python.ops import array_ops
from tensorflow.python.framework import ops
from tensorflow.python.ops import data_flow_ops
import preprocess

image_path = "D:/Google Cloud/lfw_mtcnnpy_160/"
# image_path = "/content/drive/My Drive/lfw_mtcnnpy_160/"
image_size = (160,160)
batch_size = 32

dataset = preprocess.get_dataset(image_path=image_path)
image_path_list,label_list = preprocess.create_image_path_list_and_label_list(dataset=dataset)

labels = ops.convert_to_tensor(label_list, dtype=tf.int32)
size = array_ops.shape(labels)[0]
index_queue = tf.train.range_input_producer(limit=size, num_epochs=None, shuffle=True, capacity=batch_size)
index_dequeue_op = index_queue.dequeue_many(batch_size)

image_paths_placeholder = tf.placeholder(shape=(None, 1), dtype=tf.string, name="image_paths")
labels_placeholder = tf.placeholder(shape=(None, 1), dtype=tf.int32, name="labels")

nrof_preprocess_threads = 4
input_queue = data_flow_ops.FIFOQueue(capacity=200000, dtypes=[tf.string, tf.int32], shapes=[(1,),(1,)])
enqueue_op = input_queue.enqueue_many([image_paths_placeholder, labels_placeholder],name="enqueue_op")
# image_batch, label_batch, filenames = preprocess.create_input_pipeline(input_queue=input_queue, image_size=image_size, batch_size=batch_size,
#                                                             nrof_preprocess_threads=4, rotate=False, crop=True, flip=True, standardization=False)

with tf.Session() as sess:
	coord = tf.train.Coordinator()
	tf.train.start_queue_runners(coord=coord)
	
	index_epoch = sess.run(index_dequeue_op)
	image_path_epoch = np.array(image_path_list)[index_epoch]
	label_epoch = np.array(label_list)[index_epoch]
	

	image_path_array = np.expand_dims(image_path_epoch,1)
	label_array = np.expand_dims(label_epoch,1)
	sess.run(enqueue_op,feed_dict={image_paths_placeholder:image_path_array,labels_placeholder:label_array})
	# for _ in range(nrof_preprocess_threads):
	# 	filenames_tensor, labels_tensor = input_queue.dequeue_many(batch_size)
	# 	filenames, labels = sess.run([filenames_tensor, labels_tensor])
	# 	filenames = np.squeeze(filenames)
	images_and_labels_list = preprocess.create_input_pipeline(input_queue, nrof_preprocess_threads, image_size=image_size, 
																	batch_size=batch_size, rotate=False, crop=True, flip=True, standardization=False)
	image_batch, label_batch = tf.train.batch_join(
        images_and_labels_list, batch_size=batch_size,
        capacity=4 * nrof_preprocess_threads * batch_size,
        allow_smaller_final_batch=True)
	# shapes=[(160, 160, 3), (1)],
	print("123")
	image_batch, label_batch = sess.run([image_batch, label_batch])
	
	