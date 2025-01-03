import numpy as np
from datasets import load_breast_cancer, load_iris_data, load_pen_digits_data, load_magic_gamma_data, load_glass_data, load_wine_quality_data
from activations import SIGMOID, TANH, RELU, LEAKY_RELU, ELU, SOFTMAX, SWISH, SOFTPLUS, GELU
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

"""
This file contains common settings for the classical neural network (Classic-NN.py) 
and the PSO NN (PSO-NN.py). These settings include the dataset and its split, the structure 
of the NN in the sence of the number of hidden layers, the activation functions, 
the number of iteration, the batch size, and the learning rate.

The goal of having a seperte file for these settings is to use the same s
etup for both algorithms to enable an objective comparison.

The PSO part needs to be tuned additionally in the main function in PSO-NN.py. 
There, you find the PSO specific tuning parameters w, c1, c2, etc. which affect only the 
PSO-NN. The goal is to tune PSO on top of the same NN structure used in the classic NN to
enable objective comparison.  
"""
############ Select the Dataset ###############
# Select a data set here. See datasets.py

#data = load_breast_cancer()
#data = load_glass_data()
#data = load_magic_gamma_data()
#data = load_pen_digits_data()
data = load_wine_quality_data()
#data = load_iris_data()

####### Tune the NN parameters here to maximize the accuracy for your seleced DS #######

# Define the number of the hidden layer
# This should be suitable for the dataset you selected
n_hidden = 8

# Define the activation function
# you can choose any from activations.py. The options are imported above
# (Reference the tuple to ensure that also the right derivative is used)
activation = TANH

# Set the total number of iterations
n_iteration = 1000 

# Tune the learning rate (this is used only by classic-NN and will not affect PSO-NN)
learning_rate = 0.01 

############# Perform data split and setup ###########################

# Do the split
# Please don't change the test_train_ration
TEST_SPLIT_RATIO = 0.2
X = data['data']
y = data['target']
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SPLIT_RATIO, random_state=42)

# The following two parameters are defined by the data (should not be changed)
n_inputs = X.shape[1]
n_classes = len(np.unique(y))  





