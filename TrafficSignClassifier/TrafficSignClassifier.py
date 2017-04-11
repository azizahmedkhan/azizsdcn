# Load pickled data
import pickle
import numpy as np
import matplotlib.pyplot as plt
import random

#%matplotlib inline
# TODO: Fill this in based on where you saved the training and testing data

training_file = "../traffic-signs-data/train.p"
validation_file="../traffic-signs-data/valid.p"
testing_file = "../traffic-signs-data/test.p"

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(validation_file, mode='rb') as f:
    valid = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)
x_train, y_train = train['features'], train['labels']
x_valid, y_valid = valid['features'], valid['labels']
x_test, y_test = test['features'], test['labels']

### Replace each question mark with the appropriate value.
### Use python, pandas or numpy methods rather than hard coding the results

# TODO: Number of training examples
n_train = len(x_train)

# TODO: Number of testing examples.
n_test = len(x_test)

# TODO: What's the shape of an traffic sign image?
image_shape = x_test[0].shape

# TODO: How many unique classes/labels there are in the dataset.
n_classes = np.unique(y_train)

print("Number of training examples =", n_train)
print("Number of testing examples =", n_test)
print("Image data shape =", image_shape)
print("Number of classes =", len(n_classes))
print("Unique clases = ", n_classes)

### Data exploration visualization code goes here.
### Feel free to use as many code cells as needed.
import matplotlib.pyplot as plt
# Visualizations will be shown in the notebook.
#%matplotlib inline
# I can visualize how much a lable appear in a dataset.
#Also I can show how much label diversity in testing and validation sets are.
#from numpy.random import beta
import matplotlib.pyplot as plt
plt.style.use('bmh')
plt.xlabel("Image Label")
plt.ylabel("number of times")
plt.hist(y_train, label= "Train", alpha=0.1, bins = n_classes)
plt.hist(y_test, label= "Test", alpha=0.5, bins = n_classes)
plt.hist(y_valid, label= "Valid", alpha=0.9, bins = n_classes)
#plt.rcParams["figure.figsize"] = [1.0, 2.0]
#plt.figure(figsize=(20,10))
plt.xticks(n_classes)
plt.legend()
plt.show()

# to view how the image look like
plt.figure(figsize=(2,2))
plt.imshow(x_train[0])
print(y_train[0])
plt.imshow(x_train[21])
plt.show()
print(y_train)

### Preprocess the data here. Preprocessing steps could include normalization, converting to grayscale, etc.
### Feel free to use as many code cells as needed.
from sklearn.utils import shuffle
x_train,y_train = shuffle(x_train,y_train)

import tensorflow as tf
from tensorflow.contrib.layers import flatten
EPOCHS = 10
BATCH_SIZE = 128
from tensorflow.contrib.layers import flatten

# Arguments used for tf.truncated_normal, randomly defines variables for the weights and biases for each layer
mu = 0
sigma = 0.1
def convolution(input, filterSize, outPutDepth):
    return
def reluLayer(input, shapeForNormalWeight, sizeOfBiasArray ):
    conv_W = tf.Variable(tf.truncated_normal(shape=shapeForNormalWeight, mean = mu, stddev = sigma))
    conv_b = tf.Variable(tf.zeros(sizeOfBiasArray))
    conv   = tf.nn.conv2d(input, conv_W, strides=[1, 1, 1, 1], padding='SAME') + conv_b
    return tf.nn.relu(conv)

def xwplusb(inpputLayer, inputSize, outputSize):
    fc_W = tf.Variable(tf.truncated_normal(shape=(inputSize, outputSize), mean = mu, stddev = sigma))
    fc_b = tf.Variable(tf.zeros(outputSize))
    return  tf.matmul(inpputLayer, fc_W) + fc_b

### Define your architecture here.
### Feel free to use as many code cells as needed.
from tensorflow.contrib.layers import flatten
def flatten_layer(layer):
    # Get the shape of the input layer.
    layer_shape = layer.get_shape()
    num_features = layer_shape[1:4].num_elements()
    print (num_features)
    layer_flat = tf.reshape(layer, [-1, num_features])
    layer_flat.get_shape()
    return layer_flat, num_features

def trafficSignClassifier1(inputData):
    # Arguments used for tf.truncated_normal, randomly defines variables for the weights and biases for each layer
    mu = 0
    sigma = 0.1
    keepProb = tf.placeholder(tf.float32)
    layer1OutPut = reluLayer (inputData, (1,1,3,3), 3 )
    layer2OutPut = reluLayer (layer1OutPut, (5,5,3,32), 32 )
    layer3OutPut = reluLayer (layer2OutPut, (5,5,32,32), 32 )
    layer3OutPut = tf.nn.dropout(layer3OutPut, keepProb)
    #layer3OutPut = tf.nn.max_pool(layer3OutPut, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
    layer4OutPut = reluLayer (layer3OutPut, (5,5,32,64), 64 )
    layer5OutPut = reluLayer (layer4OutPut, (5,5,64,64), 64 )
    #layer5OutPut = tf.nn.max_pool(layer5OutPut, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
    layer5OutPut = tf.nn.dropout(layer5OutPut, keepProb)
    layer6OutPut = reluLayer (layer5OutPut, (5,5,64,128), 128 )
    layer7OutPut = reluLayer (layer6OutPut, (5,5,128,128), 128 )
    layer5OutPut = tf.nn.dropout(layer5OutPut, keepProb)
    #layer7OutPut = tf.nn.max_pool(layer7OutPut, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # TODO: Layer 1: Convolutional. Input = 32x32x1. Output = 28x28x6.
    #layer1OutPut = reluLayer (x, (5,5,3,6), 6 )
    # TODO: Pooling. Input = 28x28x6. Output = 14x14x6.
    #layer1OutPut = tf.nn.max_pool(layer1OutPut, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
    # TODO: Layer 2: Convolutional. Output = 10x10x16.
    #layer2OutPut = reluLayer (layer1OutPut, (5,5,6,16), 16)
    # TODO: Pooling. Input = 10x10x16. Output = 5x5x16.
    #layer2OutPut = tf.nn.max_pool(layer2OutPut, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
    # TODO: Flatten. Input = 5x5x16. Output = 400.
    layer_flat3, num_fc_layers3 = flatten_layer(layer3OutPut)
    layer_flat5, num_fc_layers5 = flatten_layer(layer5OutPut)
    layer_flat7, num_fc_layers7 = flatten_layer(layer7OutPut)
    fc0   = flatten(layer7OutPut)
    # TODO: Layer 3: Fully Connected. Input = 400. Output = 120.
    # TODO: Activation.
    #fc1 =tf.nn.relu(xwplusb (fc0, 400, 120))
    # TODO: Layer 4: Fully Connected. Input = 120. Output = 84.
     # TODO: Activation.
    #fc2 =tf.nn.relu(xwplusb (fc1, 120, 84))
    # TODO: Layer 5: Fully Connected. Input = 84. Output = 10.
    #logits =xwplusb (fc2, 84, 10 )
    #return logits
    return fc0

def trafficSignClassifier(inputData):
    # Arguments used for tf.truncated_normal, randomly defines variables for the weights and biases for each layer
    mu = 0
    sigma = 0.1
    # TODO: Layer 1: Convolutional. Input = 32x32x1. Output = 28x28x6.
    layer1OutPut = reluLayer(inputData, (5, 5, 3, 6), 6)
    # TODO: Pooling. Input = 28x28x6. Output = 14x14x6.
    layer1OutPut = tf.nn.max_pool(layer1OutPut, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
    # TODO: Layer 2: Convolutional. Output = 10x10x16.
    # TODO: Activation.
    layer2OutPut = reluLayer(layer1OutPut, (5, 5, 6, 16), 16)
    # TODO: Pooling. Input = 10x10x16. Output = 5x5x16.
    layer2OutPut = tf.nn.max_pool(layer2OutPut, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
    # TODO: Flatten. Input = 5x5x16. Output = 400.
    fc0 = flatten(layer2OutPut)
    # TODO: Layer 3: Fully Connected. Input = 400. Output = 120.
    # TODO: Activation.
    fc1 = tf.nn.relu(xwplusb(fc0, 400, 120))
    # TODO: Layer 4: Fully Connected. Input = 120. Output = 84.
    # TODO: Activation.
    fc2 = tf.nn.relu(xwplusb(fc1, 120, 84))
    # TODO: Layer 5: Fully Connected. Input = 84. Output = 10.
    logits = xwplusb(fc2, 84, 10)
    return logits

x = tf.placeholder(tf.float32, (None, 32, 32, 3))
y = tf.placeholder(tf.int32, (None))
one_hot_y = tf.one_hot(y, 10)
### Train your model here.
rate = 0.001
logits = trafficSignClassifier(x)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=one_hot_y)
loss_operation = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate=rate)
training_operation = optimizer.minimize(loss_operation)
### Calculate and report the accuracy on the training and validation set.
correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
saver = tf.train.Saver()


def evaluate(X_data, y_data):
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset + BATCH_SIZE], y_data[offset:offset + BATCH_SIZE]
        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples


### Once a final model architecture is selected,
### the accuracy on the test set should be calculated and reported as well.
### Feel free to use as many code cells as needed.
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    num_examples = len(x_train)

    print("Training...")
    print()
    for i in range(EPOCHS):
        x_train, y_train = shuffle(x_train, y_train)
        for offset in range(0, num_examples, BATCH_SIZE):
            end = offset + BATCH_SIZE
            batch_x, batch_y = x_train[offset:end], y_train[offset:end]
            sess.run(training_operation, feed_dict={x: batch_x, y: batch_y})

        validation_accuracy = evaluate(x_valid, y_valid)
        print("EPOCH {} ...".format(i + 1))
        print("Validation Accuracy = {:.3f}".format(validation_accuracy))
        print()

    saver.save(sess, './trafficsigns')
    print("Model saved")