import numpy as np
from datasets import load_breast_cancer, load_iris_data, load_pen_digits_data, load_magic_gamma_data, load_glass_data, load_wine_quality_data
from activations import SIGMOID, TANH, RELU, LEAKY_RELU, ELU, SOFTMAX, SWISH, SOFTPLUS, GELU
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.neural_network import MLPClassifier

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

def HyperparameterTuning(X,y):
    param_grid = {
        'hidden_layer_sizes': [(4,), (8,), (16,), (32,)],
        'activation': ['relu', 'tanh', 'logistic'],
        'learning_rate_init': [0.001, 0.01, 0.1],
        'max_iter':[100,1000,10000]
    }
    
    grid_search = GridSearchCV(
        estimator=MLPClassifier(learning_rate='constant'),
        param_grid=param_grid,
        scoring='accuracy',
        cv=5,
        verbose=1,
        n_jobs=-1
    )
    
    grid_search.fit(X, y)
    
    return grid_search.best_params_


def PreprocessData(dataset,split=0.2,random_state=1):
    load={"digits":load_pen_digits_data,
            "iris":load_iris_data,
            "wine":load_pen_digits_data}
    
    data=load[dataset]()
    X = data['data']
    y = data['target']

    scaler=StandardScaler()
    scaler.fit_transform(X)
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)

    n_inputs = X.shape[1]
    n_classes = len(np.unique(y))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split, random_state=random_state)

    params=HyperparameterTuning(X_train,y_train)
    n_hidden=params['hidden_layer_sizes'][0]
    n_iteration=params['max_iter']
    activations={'relu':RELU, 'tanh':TANH, 'logistic':SIGMOID}
    activation=activations[params['activation']]
    learning_rate=params['learning_rate_init']

    return X_train, X_test, y_train, y_test, n_inputs, n_classes, n_hidden,n_iteration,activation,learning_rate