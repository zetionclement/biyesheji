import os
import tensorflow as tf
import numpy as np
import random
from scipy import misc
from scipy import interpolate

class ImageClass():
    def __init__(self, name, image_paths):
        self.name = name
        self.image_paths = image_paths

def get_dataset(image_path):
    dataset = []
    images_path_exp = os.path.expanduser(image_path)
    people = os.listdir(images_path_exp)
    for person in people:
        if os.path.isdir(os.path.join(images_path_exp,person)):
            images = os.listdir(os.path.join(images_path_exp,person))
            if len(images) >= 10:
                images_path = []
                index_list = generate_random_index(len(images)+1)
                for index in index_list:
                    images_path.append(os.path.join(images_path_exp,person,index + '.png'))
                Image = ImageClass(person, images_path)
                dataset.append(Image)
    return dataset

def generate_random_index(size):
    random_list = list(random.sample(range(1, size), 10))
    index_list = [str(i) for i in random_list]
    for i in range(len(index_list)):
        if len(index_list[i]) == 1:
            index_list[i] = '00' + index_list[i]
        elif len(index_list[i]) == 2:
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
    angle = np.random.uniform(low=-10.0,high=10.0)
    return misc.imrotate(image, angle, 'bicubic')

def create_input_pipeline(input_queue, image_size, batch_size=32, nrof_preprocess_threads=4, rotate=False, crop=False, flip=False, standardization=False):
    images_and_labels_list = []
    for _ in range(nrof_preprocess_threads):
        filenames, labels = input_queue.dequeue()
        images = []
        for filename in tf.unstack(filenames, 0):
            file_contenet = tf.read_file(filename)
            image = tf.image.decode_image(file_contenet, channels=3)
            image = tf.cond(rotate, lambda: tf.py_func(random_rotate, [image], tf.uint8),
                            lambda: tf.identity(image))
            image = tf.cond(crop, lambda: tf.random_crop(image_size + (3,)),
                            lambda: tf.image.resize_image_with_crop_or_pad(image, image_size[0], image_size[1]))
            image = tf.cond(flip, lambda: tf.image.random_flip_left_right(image),
                            lambda: tf.identity(image))
            image = tf.cond(standardization, lambda: (tf.cast(image, tf.float32) - 127.5)/128.0,
                            lambda: tf.image.per_image_standardization(image))
            image.set_shape(image_size + (3,))
            images.append(image)
        images_and_labels_list.append([images, labels])

    image_batch, label_batch = tf.train.batch_join(images_and_labels_list, batch_size=batch_size,
                                                   capacity=32, enqueue_many=True,
                                                   shapes=[image_size + (3,), []], allow_smaller_final_batch=True)
    return image_batch, label_batch


