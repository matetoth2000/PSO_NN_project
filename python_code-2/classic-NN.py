import numpy as np
from commonsetup import n_hidden, X_train, X_test, y_train, y_test, n_inputs, n_classes, activation, n_iteration, learning_rate

class NeuralNetwork:
    def __init__(self, n_inputs, n_hidden, n_classes, activation, activation_derivative):
        self.n_inputs = n_inputs
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.activation = activation
        self.activation_derivative = activation_derivative
        
        # Initialize weights and biases
        self.W1 = np.random.randn(self.n_inputs, self.n_hidden) * 0.01
        self.b1 = np.zeros((1, self.n_hidden))
        self.W2 = np.random.randn(self.n_hidden, self.n_classes) * 0.01
        self.b2 = np.zeros((1, self.n_classes))

    def forward(self, X):
        """Forward pass through the network."""
        self.z1 = np.dot(X, self.W1) + self.b1  # Pre-activation in Layer 1
        self.a1 = self.activation(self.z1)      # Activation in Layer 1
        self.z2 = np.dot(self.a1, self.W2) + self.b2  # Pre-activation in Layer 2
        self.a2 = self.softmax(self.z2)  # Output layer using softmax
        return self.a2

    def backward(self, X, y, learning_rate):
        """Backward pass to update weights and biases using gradient descent."""
        m = y.shape[0]
        delta2 = self.a2.copy()
        delta2[range(m), y] -= 1 
        delta2 /= m

        dW2 = np.dot(self.a1.T, delta2) 
        db2 = np.sum(delta2, axis=0, keepdims=True) 

        delta1 = np.dot(delta2, self.W2.T) * self.activation_derivative(self.z1) 
        dW1 = np.dot(X.T, delta1) 
        db1 = np.sum(delta1, axis=0, keepdims=True)

        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1

    def train(self, X_train, y_train, epochs, learning_rate):
        """Train the neural network."""
        for epoch in range(epochs):
            self.forward(X_train)  # Forward pass
            self.backward(X_train, y_train, learning_rate)  # Backward pass
            if epoch % 100 == 0:
                loss = self.compute_loss(y_train)
                print(f'Epoch {epoch}, Loss: {loss:.4f}')

    def predict(self, X):
        """Predict the classes based on trained weights."""
        a2 = self.forward(X)
        return np.argmax(a2, axis=1)

    def compute_loss(self, y):
        """Compute the categorical cross-entropy loss."""
        m = y.shape[0]
        log_likelihood = -np.log(self.a2[range(m), y])
        return np.sum(log_likelihood) / m

    def softmax(self, z):
        """Compute softmax values for each set of scores in z."""
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))  # Numerical stability
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

def main():
    nn = NeuralNetwork(n_inputs, n_hidden, n_classes, activation[0], activation[1])
    nn.train(X_train, y_train, n_iteration, learning_rate)
    y_pred = nn.predict(X_test)
    accuracy = (y_pred == y_test).mean()
    print(f"Accuracy Classic-NN: {accuracy:.2f}")
    
if __name__ == "__main__":
    main()