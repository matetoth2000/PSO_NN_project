import numpy as np
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)
SIGMOID = (sigmoid, sigmoid_derivative)

def tanh(x):
    return np.tanh(x)
def tanh_derivative(x):
    return 1 - np.tanh(x) ** 2
TANH = (tanh, tanh_derivative)

def relu(x):
    return np.maximum(0, x)
def relu_derivative(x):
    return np.where(x > 0, 1, 0)
RELU = (relu, relu_derivative)

def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)
def leaky_relu_derivative(x, alpha=0.01):
    return np.where(x > 0, 1, alpha)
LEAKY_RELU = (leaky_relu, leaky_relu_derivative)

def elu(x, alpha=1.0):
    return np.where(x > 0, x, alpha * (np.exp(x) - 1))
def elu_derivative(x, alpha=1.0):
    return np.where(x > 0, 1, alpha * np.exp(x))
ELU = (elu, elu_derivative)

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0, keepdims=True)
def softmax_derivative(x):
    s = softmax(x)
    return s * (1 - s)
SOFTMAX = (softmax, softmax_derivative)

def swish(x):
    return x * sigmoid(x)
def swish_derivative(x):
    s = sigmoid(x)
    return s + x * s * (1 - s)
SWISH = (swish, swish_derivative)

def softplus(x):
    return np.log1p(np.exp(x))
def softplus_derivative(x):
    return sigmoid(x)
SOFTPLUS = (softplus, softplus_derivative)

def gelu(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))
def gelu_derivative(x):
    c = np.sqrt(2 / np.pi)
    tanh_term = np.tanh(c * (x + 0.044715 * x**3))
    return 0.5 * (1 + tanh_term) 
GELU = (gelu, gelu_derivative)

