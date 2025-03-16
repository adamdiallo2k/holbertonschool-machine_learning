Write the function def l2_reg_cost(cost, model): that calculates the cost of a neural network with L2 regularization:

cost is a tensor containing the cost of the network without L2 regularization
model is a Keras model that includes layers with L2 regularization
Returns: a tensor containing the total cost for each layer of the network, accounting for L2 regularization
Note: To accompany the following main file, you are provided with a Keras model saved in the file model_reg.h5. The architecture of this model includes:

an input layer
two hidden layers with tanh and sigmoid activations, respectively
an output layer with softmax activation
L2 regularization is applied to all layers
ubuntu@alexa-ml:~/regularization$ cat 2-main.py
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

def one_hot(Y, classes):
    """convert an array to a one-hot matrix"""
    m = Y.shape[0]
    oh = np.zeros((m, classes))
    oh[np.arange(m), Y] = 1
    return oh

m = np.random.randint(1000, 2000)
c = 10
lib= np.load('MNIST.npz')

X = lib['X_train'][:m].reshape((m, -1))
Y = one_hot(lib['Y_train'][:m], c)

model_reg = tf.keras.models.load_model('model_reg.h5', compile=False)

Predictions = model_reg(X)
cost = tf.keras.losses.CategoricalCrossentropy()(Y, Predictions)

l2_cost = l2_reg_cost(cost,model_reg)
print(l2_cost)

ubuntu@alexa-ml:~/regularization$ ./2-main.py
tf.Tensor([121.24274   110.74535     6.1250796], shape=(3,), dtype=float32)
ubuntu@alexa-ml:~/regularization$
Repo:

GitHub repository: holbertonschool-machine_learning
Directory: supervised_learning/regularization
File: 2-l2_reg_cost.py
