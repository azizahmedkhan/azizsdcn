from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets(".", one_hot=True, reshapr=False)

import tensorflow as tf

#PArameters
learning_rate = 0.00001
epochs = 10
batch_size = 128

