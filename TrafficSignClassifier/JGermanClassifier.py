import pandas as pd

import numpy as np
import tensorflow as tf
import time
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os

from multiprocessing import Queue

import pickle

training_file = "./traffic-signs-data/train.p"
validation_file="./traffic-signs-data/valid.p"
testing_file = "./traffic-signs-data/test.p"

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)

X_train, y_train = train['features'], train['labels']
X_test, y_test = test['features'], test['labels']

n_train = len(X_train)
n_test = len(X_test)
image_shape = X_train[0].shape
n_classes = len(set(y_train))

print("Number of training examples =", n_train)
print("Number of testing examples =", n_test)
print("Image data shape =", image_shape)
print("Number of classes =", n_classes)

print('Sample images')
#for i in range(4):
#    plt.subplot(2,2,i+1)
#    plt.imshow(X_train[i*1500+1])

#plt.hist(y_train, bins=n_classes)
#plt.title('Number of examples of each sign in the training set')
#plt.xlabel('Sign')
#plt.ylabel('Count')
#plt.plot()

#plt.imshow(X_train[17031])
# Shuffle training examples
from sklearn.utils import shuffle
X_train, y_train = shuffle(X_train, y_train, random_state=42)

X_train_orig = X_train
X_test_orig = X_test

# Normalise input (images still in colour)
X_train = (X_train - X_train.mean()) / (np.max(X_train) - np.min(X_train))
X_test = (X_test - X_test.mean()) / (np.max(X_test) - np.min(X_test))

def plot_norm_image(image_index):
    """Plots original image on the left and normalised image on the right."""
    plt.subplot(2,2,1)
    plt.imshow(X_train_orig[image_index])
    plt.subplot(2,2,2)
    plt.imshow(X_train[image_index])

#plot_norm_image(20)


from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train,
                                                  test_size=0.2,
                                                  random_state=42)

# Network parameters
n_input = 32 * 32 * 3
nb_filters = 32
kernel_size = (3, 3)
input_shape = (32, 32, 3)
n_fc1 = 512
n_fc2 = 128
in_channels = 3
pool_size = 2 # i.e. (2,2)

dropout_conv = 0.9
dropout_fc = 0.9

weights_stddev = 0.1
weights_mean = 0.0
biases_mean = 0.0

padding = 'VALID'
if padding == 'SAME':
    conv_output_length = 6
elif padding == 'VALID':
    conv_output_length = 5
#else:
#    raiseException("Unknown padding.")

# tf Graph input
x_unflattened = tf.placeholder("float", [None, 32, 32, 3])
x = x_unflattened

y_rawlabels = tf.placeholder("int32", [None])
y = tf.one_hot(y_rawlabels, depth=43, on_value=1., off_value=0., axis=-1)


## Create model

def conv2d(x, W, b, strides=3):
    print("v>>",x, " Weights >>", W," Biases >>", b)

    """Conv2D wrapper, with bias and relu activation"""
    # strides = [batch, in_height, in_width, channels]
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)


def maxpool2d(x, k=2, padding_setting='SAME'):
    """MaxPool2D wrapper."""
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                          padding=padding_setting)

def conv_net(model_x, model_weights, model_biases, model_pool_size,
             model_dropout_conv, model_dropout_fc, padding='SAME'):
    """Convolutional neural network model."""
    # Convolution Layer 1
    print("model_x>>", model_x)
    print ("model_weights['conv1']",model_weights['conv1'])
    conv1 = conv2d(model_x, model_weights['conv1'], model_biases['conv1'])
    print("conv1 after conv>>", conv1)
    # Max Pooling (down-sampling)
    conv1 = maxpool2d(conv1, k=model_pool_size, padding_setting=padding)
    print("conv1 after max pool>>", conv1)
    conv1 = tf.nn.dropout(conv1, model_dropout_conv)
    print("conv1 after dropout>>",conv1)
    # Fully connected layer 1
    # Reshape conv1 output to fit fully connected layer input
    conv1_shape = conv1.get_shape().as_list()
    fc1 = tf.reshape(conv1, [-1, conv1_shape[1]*conv1_shape[2]*conv1_shape[3]])
    print("fc1 after reshape>>", fc1)
    fc1 = tf.add(tf.matmul(fc1, model_weights['fc1']), model_biases['fc1'])
    print("fc1 after matmul>>", fc1)
    fc1 = tf.nn.relu(fc1)
    print("fc1 after relu>>", fc1)
    fc1 = tf.nn.dropout(fc1, model_dropout_fc)
    print("fc1 after dropout>>", fc1)
    # Fully connected layer 2
    fc2 = tf.add(tf.matmul(fc1, model_weights['fc2']), model_biases['fc2'])
    print("fc2 after matmul>>", fc2)
    fc2 = tf.nn.relu(fc2)
    print("fc2  after relu>>", fc2)
    fc2 = tf.nn.dropout(fc2, model_dropout_fc)
    print("fc2 after dropout>>", fc2)
    # Output layer
    output = tf.add(tf.matmul(fc2, model_weights['out']), model_biases['out'])
    print("output>>", output)
    # Note: Softmax is outside the model
    return output


## Store layers weight & bias

# NEW: initialise neurons with slightly positive initial bias
# to avoid dead neurons.
def weight_variable(shape, weight_mean, weight_stddev):
    initial = tf.truncated_normal(shape, stddev=weight_stddev, mean=weight_mean)
    # alt: tf.random_normal(shape)
    return tf.Variable(initial)


def bias_variable(shape, bias_mean):
    initial = tf.constant(bias_mean, shape=shape)
    return tf.Variable(initial)


weights = {
    'conv1': weight_variable([kernel_size[0], kernel_size[1], in_channels, nb_filters], weights_mean, weights_stddev),
    'fc1': weight_variable([nb_filters * conv_output_length**2, n_fc1], weights_mean, weights_stddev),
    'fc2': weight_variable([n_fc1, n_fc2], weights_mean, weights_stddev),
    'out': weight_variable([n_fc2, n_classes], weights_mean, weights_stddev)
}

biases = {
    'conv1': bias_variable([nb_filters], biases_mean),
    'fc1': bias_variable([n_fc1], biases_mean),
    'fc2': bias_variable([n_fc2], biases_mean),
    'out': bias_variable([n_classes], biases_mean)
}

# Training parameters
learning_rate = 0.001
initial_learning_rate = learning_rate
training_epochs = 150
batch_size = 100
display_step = 1
n_train = len(X_train)

anneal_mod_frequency = 15
# Annealing rate of 1: learning rate remains constant.
annealing_rate = 1

print_accuracy_mod_frequency = 1

# Construct model
pred = conv_net(x, weights, biases, pool_size, dropout_conv, dropout_fc, padding=padding)
pred_probs = tf.nn.softmax(pred)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = pred, labels = y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Function to initialise the variables
init = tf.initialize_all_variables()

### RUN MODEL ###
# Launch the graph
sess = tf.Session()

# Initialise variables
sess.run(init)

# Initialise time logs
init_time = time.time()
epoch_time = init_time

five_epoch_moving_average = 0.
epoch_accuracies = []

# Training cycle
for epoch in range(training_epochs):
    if five_epoch_moving_average > 0.96:
        break

    avg_cost = 0.

    total_batch = int(n_train / batch_size)
    # Loop over all batches
    for i in range(total_batch):
        batch_x, batch_y = np.array(X_train[i * batch_size:(i + 1) * batch_size]), \
                           np.array(y_train[i * batch_size:(i + 1) * batch_size])
        # tf.train.batch([X_train, y_train], batch_size=100, enqueue_many=True)
        # Run optimization op (backprop) and cost op (to get loss value)
        _, c = sess.run([optimizer, cost], feed_dict={x_unflattened: batch_x, y_rawlabels: batch_y})
        # Compute average loss
        avg_cost += c / total_batch
        # print(avg_cost)
    # Display logs per epoch step
    if epoch % display_step == 0:
        print("Epoch:", '%04d' % (epoch + 1), "cost=",
              "{:.9f}".format(avg_cost))
        last_epoch_time = epoch_time
        epoch_time = time.time()
        # print("Time since last epoch: ", epoch_time - last_epoch_time)
    # Anneal learning rate
    if (epoch + 1) % anneal_mod_frequency == 0:
        learning_rate *= annealing_rate
        print("New learning rate: ", learning_rate)

    if (epoch + 1) % print_accuracy_mod_frequency == 0:
        correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        # Line below needed only when not using `with tf.Session() as sess`
        with sess.as_default():
            epoch_accuracy = accuracy.eval({x_unflattened: X_val, y_rawlabels: y_val})
            # TODO: optimise five_epoch_moving_average, e.g. using a queue
            epoch_accuracies.append(epoch_accuracy)
            if epoch >= 4:
                five_epoch_moving_average = np.sum(epoch_accuracies[epoch - 5:epoch]) / 5
                print("Five epoch moving average: ", five_epoch_moving_average)
            print("Accuracy (validation):", epoch_accuracy)

print("Optimization Finished!")

# Test model
correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
# Calculate accuracy
# accuracy_train = tf.reduce_mean(tf.cast(correct_prediction, "float"))
# print("Accuracy (train):", accuracy_train.eval({x_unflattened: X_train, y_rawlabels: y_train}))
train_predict_time = time.time()
# print("Time to calculate accuracy on training set: ", train_predict_time - epoch_time)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
# Line below needed only when not using `with tf.Session() as sess`
with sess.as_default():
    print("Accuracy (test):", accuracy.eval({x_unflattened: X_test, y_rawlabels: y_test}))
test_predict_time = time.time()
print("Time to calculate accuracy on test set: ", test_predict_time - train_predict_time)

# Print parameters for reference
print("\nParameters:")
print("Learning rate (initial): ", initial_learning_rate)
print("Anneal learning rate every ", anneal_mod_frequency, " epochs by ", 1 - annealing_rate)
print("Learning rate (final): ", learning_rate)
print("Training epochs: ", training_epochs)
print("Batch size: ", batch_size)
print("Dropout (conv): ", dropout_conv)
print("Dropout (fc): ", dropout_fc)
print("Padding: ", padding)
print("weights_mean: ", weights_mean)
print("weights_stddev: ", weights_stddev)
print("biases_mean: ", biases_mean)


epochs_results = pd.read_csv('logs/tsc-p2-submission-2-model-perf.csv')
epochs_results.plot(x='epoch', y='validation_accuracy',
                    title='Validation accuracy')
epochs_results.plot(x='epoch', y='five_epoch_moving_average',
                    title='Validation accuracy (five-epoch moving average)')

