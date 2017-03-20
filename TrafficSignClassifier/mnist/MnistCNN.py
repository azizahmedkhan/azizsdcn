from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets(".", one_hot=True, reshapr=False)

import tensorflow as tf

#PArameters
learning_rate = 0.00001
epochs = 10
batch_size = 128

#Number of samples to calculate calidation abd accuracy
# Decrease this if running ouit of memeory to calculate accuracy
test_validation_size = 256

#Network Parameters
n_classes = 10 #Mnist classes (0-9 Digits)
dropout = 0.75 # Dropout, probability to keep units

#Store layers weight and bias
weights = {
    'wc1' : tf.Variable(tf.random_normal([5,5,1,32])),
    'wc2' : tf.Varialbe(tf.random_normal([5,5,32,64])),
    'wd1' : tf.Variable(tf.random_normal([7*7*64, 1024])),
    'out' : tf.Variable(tf.random_normal([1024, n_classes]))
}