import numpy as np
import pyswarms as ps
from commonsetup import n_hidden, X_train, X_test, y_train, y_test, n_inputs, n_classes, activation, n_iteration

class NeuralNetwork:
    def __init__(self, n_inputs, n_hidden, n_classes, activation):
        self.n_inputs = n_inputs
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.activation = activation

    def softmax(self, z):
        """Compute softmax values for each set of scores in z."""
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))  # Numerical stability
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)
    
    def count_param(self):
        """Calculate the total number of parameters to optimize."""
        return (self.n_inputs * self.n_hidden) + (self.n_hidden * self.n_classes) + self.n_hidden + self.n_classes

    def generate_logits(self, x, data):
        """ 
        Parameters:
        x: one PSO solution (a list of variables, i.e. coordinates of a particle) 
        data: The train or test data to be predited

        At first, the function builds the network by cutting the values in x into the weights and 
        biases of the NN. Then it passes the data and performs activation to get the logits. 
        """
        ind1 = self.n_inputs * self.n_hidden
        W1 = x[0:ind1].reshape((self.n_inputs, self.n_hidden))
        ind2 = ind1 + self.n_hidden
        b1 = x[ind1:ind2].reshape((self.n_hidden,))
        ind3 = ind2 + self.n_hidden * self.n_classes
        W2 = x[ind2:ind3].reshape((self.n_hidden, self.n_classes))
        b2 = x[ind3:ind3 + self.n_classes].reshape((self.n_classes,))

        z1 = data.dot(W1) + b1  # Pre-activation in Layer 1
        a1 = self.activation(z1)  # Activation in Layer 1
        logits = a1.dot(W2) + b2  # Pre-activation in Layer 2

        return logits

    def forward_prop(self, params, X_train, y_train):
        """Calculate the loss using forward propagation."""
        logits = self.generate_logits(params, X_train)
        exp_scores = np.exp(logits)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        correct_logprobs = -np.log(probs[range(X_train.shape[0]), y_train])
        loss = np.sum(correct_logprobs) / X_train.shape[0]
        return loss    
    
    def predict(self, weights, data):
        """Predict the classes based on trained weights."""
        logits = self.generate_logits(weights, data)
        return np.argmax(logits, axis=1)

class PSOOptimizer:
    def __init__(self, nn, c1, c2, w, swarm_size, n_iterations, batchsize):
        self.nn = nn
        self.c1 = c1  # self confidence
        self.c2 = c2  # swarm confidence
        self.w = w # inertia (omega)
        self.swarm_size = swarm_size
        self.n_iterations = n_iterations
        self.batchsize = batchsize

    def fitness_function(self, X, X_train, y_train):
        """
        Parameters:
        X: 2-D array holding the PSO solutions to be evaluated by the fitness function
        X_train: Train set
        Y_train: target of the train set

        This is the fitness function used by the PSO, which is to be implemented (completed ) 
        by the strudents. The objective is understanding the concept of how PSO is applied in 
        this use case, namely optimizing a NN.

        Note that in each iteration of the PSO algorithm, a set of solutions are generated, 
        namely one solution by each particle. These are passed to this fittness function in the 
        parameter X. Since each solution is a list of numbers (the coordinates of a particle 
        position), X is a two-dimensional array.

        Note that each solution is used to setup the weights and biases of the network. 
        Therefor, what you shuld do here is performing the forward propagation each time
        using the solution and random batches of training data to return the resulting accuracies 
        in a one-dimensional list.

        To do this successfully, refer to and understand the functions forward_prop() and 
        generate_logits() that is called inside it.

        Note that the current implementation of the function is random, which will run
        but it will result in low accuracy ~ 1/n_classes.
        """
        # TODO: complete the implementation of this function
        fitness_values = []
        # Get a random batch using the random_batch function
        X_batch, y_batch = self.random_batch(X_train, y_train)

        for solution in X:
            # Compute the loss using forward propagation
            loss = self.nn.forward_prop(solution, X_batch, y_batch)
            
            # Convert loss to a fitness value (lower loss = higher fitness)
            fitness = loss  # Negative loss since lower loss is better
            
            # Append fitness value
            fitness_values.append(fitness)

        return fitness_values

    def random_batch(self, X_train, y_train):
        indices = np.random.choice(len(X_train), size=self.batchsize, replace=False)
        X_batch = X_train[indices]
        y_batch = y_train[indices]
        return (X_batch, y_batch)
    
    def optimize(self, X_train, y_train):
        """Perform the PSO optimization."""
        dimensions = self.nn.count_param()
        optimizer = ps.single.GlobalBestPSO(n_particles=self.swarm_size, dimensions=dimensions,
                                            options={'c1': self.c1, 'c2': self.c2, 'w': self.w})
        cost, weights = optimizer.optimize(self.fitness_function, iters=self.n_iterations, verbose=False,
                                       X_train=X_train, y_train=y_train)
        # Log the best weights
        print(f"Best Weights Found: {weights[:10]}")  # Log a sample of the weights

        return weights



def main():
    ####### PSO  Tuning ################
    # Tune the PSO parameters here trying to outperform the classic NN 
    # For more about these parameters, see the lecture resources
    par_C1 = 0.8
    par_C2 = 1.6
    par_W = 0.5
    par_SwarmSize = 30
    batchsize = 200 # The number of data instances used by the fitness function

    print ("############ you are using the following settings:")
    print ("Number hidden layers: ", n_hidden)
    print ("activation: ", activation[0])
    print ("Number of variables to optimize: ", (n_inputs * n_hidden) + (n_hidden * n_classes) + n_hidden + n_classes)
    print ("PSO parameters C1: ", par_C1, "C2: ", par_C2, "W: ", par_W, "Swarmsize: ", par_SwarmSize,  "Iteration: ", n_iteration)
    print ("\n")


    # Initialize Neural Network and PSO optimizer
    nn = NeuralNetwork(n_inputs, n_hidden, n_classes, activation[0])
    pso = PSOOptimizer(nn, par_C1, par_C2, par_W, par_SwarmSize, n_iteration, batchsize)

    # Perform optimization
    weights = pso.optimize(X_train, y_train)

    # Evaluate accuracy on the test set
    y_pred = nn.predict(weights, X_test)
    accuracy = (y_pred == y_test).mean()
    print(f"Accuracy PSO-NN: {accuracy:.2f}")

    from sklearn.metrics import confusion_matrix, classification_report
    import matplotlib.pyplot as plt
    import seaborn as sns

    def plot_confusion_matrix(y_true, y_pred, classes):
        """Plot a confusion matrix."""
        cm = confusion_matrix(y_true, y_pred, labels=classes)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.show()
        return cm

    # After accuracy computation in `main()`
    print(f"Classification Report:\n{classification_report(y_test, y_pred)}")
    classes = sorted(list(set(y_test)))  # Adjust class labels based on your dataset
    confusion_mat = plot_confusion_matrix(y_test, y_pred, classes)


if __name__ == "__main__":
    main()