Write a function def l2_reg_create_layer(prev, n, activation, lambtha): that creates a neural network layer in tensorFlow that includes L2 regularization:

prev is a tensor containing the output of the previous layer
n is the number of nodes the new layer should contain
activation is the activation function that should be used on the layer
lambtha is the L2 regularization parameter
Returns: the output of the new layer
ubuntu@alexa-ml:~/regularization$ cat 3-main.py 
#!/usr/bin/env python3

import numpy as np
import tensorflow as tf
import os
import random

SEED = 0

os.environ['PYTHONHASHSEED'] = str(SEED)
os.environ['TF_ENABLE_ONEDNN_OPTS']= '0'
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

l2_reg_cost = __import__('2-l2_reg_cost').l2_reg_cost
l2_reg_create_layer = __import__('3-l2_reg_create_layer').l2_reg_create_layer

def one_hot(Y, classes):
    """convert an array to a one-hot matrix"""
    m = Y.shape[0]
    one_hot = np.zeros((m, classes))
    one_hot[np.arange(m), Y] = 1
    return one_hot

lib= np.load('MNIST.npz')
X_train_3D = lib['X_train']
Y_train = lib['Y_train']
X_train = X_train_3D.reshape((X_train_3D.shape[0], -1))
Y_train_oh = one_hot(Y_train, 10)

input_shape = X_train.shape[1]

x = tf.keras.Input(shape=input_shape)
h1 = l2_reg_create_layer(x, 256, tf.nn.tanh, 0.05)  
y_pred = l2_reg_create_layer(h1, 10, tf.nn.softmax, 0.)   
model = tf.keras.Model(inputs=x, outputs=y_pred)

Predictions = model(X_train)
cost = tf.keras.losses.CategoricalCrossentropy()(Y_train_oh, Predictions)

l2_cost = l2_reg_cost(cost,model)
print(l2_cost)

ubuntu@alexa-ml:~/regularization$ ./3-main.py 
tf.Tensor([41.20865   2.642724], shape=(2,), dtype=float32)
ubuntu@alexa-ml:~/regularization$
Repo:

GitHub repository: holbertonschool-machine_learning
Directory: supervised_learning/regularization
File: 3-l2_reg_create_layer.py
