import os
import tensorflow as tf
import numpy as np
import random
from scipy import misc
from scipy import interpolate
from tensorflow.python.framework import ops


def center_loss(features, label, alfa, nrof_classes):
    nrof_features = features.get_shape()[1]
    centers = tf.get_variable('centers', [nrof_classes, nrof_features], dtype=tf.float32,
                              initializer=tf.constant_initializer(0), trainable=False)
    label = tf.reshape(label, [-1])
    centers_batch = tf.gather(centers, label)
    diff = (1 - alfa) * (centers - features)
    centers = tf.scatter_sub(centers, label, diff)
    with tf.control_dependencies([centers]):
        loss = tf.reduce_mean(tf.square(features - centers_batch))
    return loss, centers

class ImageClass():
    def __init__(self, name, image_paths):
        self.name = name
        self.image_paths = image_paths

def get_dataset(image_path):
    dataset = []
    images_path_exp = os.path.expanduser(image_path)
    people = os.listdir(images_path_exp)
    for person in people:
        if os.path.isdir(os.path.join(images_path_exp, person)):
            images = os.listdir(os.path.join(images_path_exp, person))
            if len(images) >= 3:
                images_path = []
                index_list = generate_random_index(len(images)+1)
                for index in index_list:
                    images_path.append(os.path.join(images_path_exp, person, person + '_' + index + '.png'))
                Image = ImageClass(person, images_path)
                dataset.append(Image)
    return dataset

def generate_random_index(size):
    random_list = list(random.sample(range(1, size), 3))
    index_list = [str(i) for i in random_list]
    for i in range(len(index_list)):
        if len(index_list[i]) == 1:
            index_list[i] = '000' + index_list[i]
        elif len(index_list[i]) == 2:
            index_list[i] = '00' + index_list[i]
        elif len(index_list[i]) == 3:
            index_list[i] = '0' + index_list[i]
    return index_list

def create_image_path_list_and_label_list(dataset):
    image_path_list = []
    label_list = []
    for i in range(len(dataset)):
        image_path_list += dataset[i].image_paths
        label_list += [i] * len(dataset[i].image_paths)
    return image_path_list, label_list

def random_rotate(image):
    angle = np.random.uniform(low=-10.0, high=10.0)
    return misc.imrotate(image, angle, 'bicubic')

def create_input_pipeline(input_queue, nrof_preprocess_threads, image_size, batch_size=32,  rotate=False, crop=False, flip=False, standardization=False):
    images_and_labels_list = []
    for _ in range(nrof_preprocess_threads):
        images = []
        filenames, label = input_queue.dequeue()
        for filename in tf.unstack(filenames):
            file_contenet = tf.read_file(filename)
            image = tf.image.decode_png(file_contenet, channels=3)
            image = tf.image.resize_images(image, [image_size[0], image_size[1]], method=2)
            image = tf.cond(tf.cast((rotate == True), tf.bool), lambda: tf.py_func(random_rotate, [image], tf.float32),
                            lambda: tf.identity(image))
            image = tf.cond(tf.cast((crop == True), tf.bool), lambda: tf.random_crop(image, image_size + (3,)),
                            lambda: tf.image.resize_image_with_crop_or_pad(image, image_size[0], image_size[1]))
            image = tf.cond(tf.cast((flip == True), tf.bool), lambda: tf.image.random_flip_left_right(image),
                            lambda: tf.identity(image))
            image = tf.cond(tf.cast((standardization == True), tf.bool), lambda: (tf.cast(image, tf.float32) - 127.5)/128.0,
                            lambda: tf.cast(tf.image.per_image_standardization(image), tf.float32))
            # image = tf.image.resize_images(image,[160, 160])
            image.set_shape((299, 299, 3))
            images.append(image)
        images_and_labels_list.append([images, label])

    # image_batch, label_batch = tf.train.batch_join(images_and_labels_list, batch_size=batch_size,
    #                                                capacity=32, enqueue_many=True,
    #                                                shapes=[image_size + (3,), []], allow_smaller_final_batch=True)
    return images_and_labels_list
