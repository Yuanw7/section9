import numpy as np
import gzip
import os
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split

# ==================================================
# Data Loading and Preprocessing
# ==================================================

def load_mnist(path, kind='train'):
    labels_path = os.path.join(path, f'{kind}-labels-idx1-ubyte.gz')
    images_path = os.path.join(path, f'{kind}-images-idx3-ubyte.gz')
    
    with gzip.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8, offset=8)

    with gzip.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8, offset=16).reshape(len(labels), 784)
    
    return images, labels

# Load and prepare data
X_train, y_train = load_mnist('data/', 'train')
X_test, y_test = load_mnist('data/', 't10k')

# Normalize and split
X_train = X_train / 255.0
X_test = X_test / 255.0
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# One-hot encode labels
def one_hot(y, num_classes=10):
    return np.eye(num_classes)[y]

y_train_oh = one_hot(y_train)
y_val_oh = one_hot(y_val)
y_test_oh = one_hot(y_test)

# Reshape for CNN
X_train_cnn = X_train.reshape(-1, 28, 28, 1)
X_val_cnn = X_val.reshape(-1, 28, 28, 1)
X_test_cnn = X_test.reshape(-1, 28, 28, 1)

# ==================================================
# MLP Implementation
# ==================================================

class MLP:
    def __init__(self, layer_sizes, activations):
        self.layer_sizes = layer_sizes
        self.activations = activations
        self.weights = []
        self.biases = []
        
        # He initialization for ReLU layers
        for i in range(len(layer_sizes)-1):
            fan_in = layer_sizes[i]
            he_std = np.sqrt(2 / fan_in)
            self.weights.append(he_std * np.random.randn(fan_in, layer_sizes[i+1]))
            self.biases.append(np.zeros(layer_sizes[i+1]))
            
    def relu(self, x):
        return np.maximum(0, x)
    
    def softmax(self, x):
        exps = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exps / np.sum(exps, axis=1, keepdims=True)
    
    def relu_derivative(self, x):
        return (x > 0).astype(float)
    
    def forward(self, X):
        self.layer_inputs = []
        self.layer_outputs = [X]
        
        for i in range(len(self.weights)):
            z = self.layer_outputs[-1] @ self.weights[i] + self.biases[i]
            self.layer_inputs.append(z)
            
            if self.activations[i] == 'relu':
                a = self.relu(z)
            elif self.activations[i] == 'softmax':
                a = self.softmax(z)
                
            self.layer_outputs.append(a)
            
        return self.layer_outputs[-1]
    
    def backward(self, X, y, lr=0.01):
        m = X.shape[0]
        gradients = []
        delta = (self.layer_outputs[-1] - y) / m  # Cross-entropy derivative
        
        for i in reversed(range(len(self.weights))):
            a_prev = self.layer_outputs[i]
            
            if self.activations[i] == 'relu':
                delta *= self.relu_derivative(self.layer_inputs[i])
            
            dW = a_prev.T @ delta
            db = np.sum(delta, axis=0)
            gradients.append((dW, db))
            
            if i > 0:
                delta = delta @ self.weights[i].T
        
        # Update weights and biases
        for i in range(len(self.weights)):
            self.weights[i] -= lr * gradients[-(i+1)][0]
            self.biases[i] -= lr * gradients[-(i+1)][1]

# ==================================================
# CNN Implementation
# ==================================================

class CNN:
    def __init__(self):
        # Conv layer: 5x5 filters, 32 channels
        self.conv_weights = np.random.randn(5, 5, 1, 32) * np.sqrt(2 / (5*5*1))
        self.conv_bias = np.zeros(32)
        
        # FC layer
        self.fc_weights = np.random.randn(12*12*32, 10) * np.sqrt(2 / (12*12*32))
        self.fc_bias = np.zeros(10)
        
    def relu(self, x):
        return np.maximum(0, x)
    
    def softmax(self, x):
        exps = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exps / np.sum(exps, axis=1, keepdims=True)
    
    def max_pool(self, x):
        n, h, w, c = x.shape
        return x.reshape(n, h//2, 2, w//2, 2, c).max(axis=(2,4))
    
    def forward(self, X):
        # Conv layer
        self.conv_out = np.zeros((X.shape[0], 24, 24, 32))
        for i in range(24):
            for j in range(24):
                patch = X[:, i:i+5, j:j+5, :]
                self.conv_out[:, i, j, :] = np.tensordot(patch, self.conv_weights, axes=([1,2,3], [0,1,2])) + self.conv_bias
        self.conv_out = self.relu(self.conv_out)
        
        # Max pooling
        self.pool_out = self.max_pool(self.conv_out)
        
        # FC layer
        self.flatten = self.pool_out.reshape(X.shape[0], -1)
        self.fc_out = self.flatten @ self.fc_weights + self.fc_bias
        return self.softmax(self.fc_out)
    
    def backward(self, X, y, lr=0.01):
        # Simplified backward pass for demonstration
        m = X.shape[0]
        delta = (self.fc_out - y) / m
        
        # FC gradients
        d_fc_w = self.flatten.T @ delta
        d_fc_b = np.sum(delta, axis=0)
        
        # Update parameters
        self.fc_weights -= lr * d_fc_w
        self.fc_bias -= lr * d_fc_b

# ==================================================
# Training and Evaluation
# ==================================================

def train_model(model, X_train, y_train, X_val, y_val, epochs=10, batch_size=64, cnn=False):
    train_loss = []
    val_acc = []
    
    for epoch in range(epochs):
        # Mini-batch training
        for i in range(0, X_train.shape[0], batch_size):
            X_batch = X_train[i:i+batch_size]
            y_batch = y_train[i:i+batch_size]
            
            if cnn:
                X_batch = X_batch.reshape(-1, 28, 28, 1)
                pred = model.forward(X_batch)
                model.backward(X_batch, y_batch, lr=0.01)
            else:
                pred = model.forward(X_batch)
                model.backward(X_batch, y_batch, lr=0.01)
        
        # Validation
        if cnn:
            val_pred = model.forward(X_val.reshape(-1, 28, 28, 1))
        else:
            val_pred = model.forward(X_val)
            
        acc = accuracy_score(np.argmax(val_pred, axis=1), np.argmax(y_val, axis=1))
        val_acc.append(acc)
        print(f"Epoch {epoch+1}/{epochs} - Val Acc: {acc:.4f}")
    
    return val_acc

# ==================================================
# Main Execution
# ==================================================

if __name__ == "__main__":
    # MLP Training
    print("Training MLP...")
    mlp = MLP([784, 512, 256, 10], ['relu', 'relu', 'softmax'])
    mlp_acc = train_model(mlp, X_train, y_train_oh, X_val, y_val_oh, epochs=10)
    
    # CNN Training
    print("\nTraining CNN...")
    cnn = CNN()
    cnn_acc = train_model(cnn, X_train_cnn, y_train_oh, X_val_cnn, y_val_oh, epochs=10, cnn=True)
    
    # Evaluation
    def evaluate_model(model, X_test, y_test, cnn=False):
        if cnn:
            X_test = X_test.reshape(-1, 28, 28, 1)
        preds = model.forward(X_test)
        y_pred = np.argmax(preds, axis=1)
        y_true = np.argmax(y_test, axis=1)
        return confusion_matrix(y_true, y_pred), accuracy_score(y_true, y_pred)
    
    # MLP Evaluation
    mlp_cm, mlp_acc = evaluate_model(mlp, X_test, y_test_oh)
    print("\nMLP Test Accuracy:", mlp_acc)
    
    # CNN Evaluation
    cnn_cm, cnn_acc = evaluate_model(cnn, X_test_cnn, y_test_oh, cnn=True)
    print("CNN Test Accuracy:", cnn_acc)
    
    # Plot Results
    plt.figure(figsize=(12, 5))
    
    plt.subplot(121)
    plt.imshow(mlp_cm, cmap='Blues')
    plt.title("MLP Confusion Matrix")
    
    plt.subplot(122)
    plt.imshow(cnn_cm, cmap='Oranges')
    plt.title("CNN Confusion Matrix")
    
    plt.figure()
    plt.plot(mlp_acc, label='MLP')
    plt.plot(cnn_acc, label='CNN')
    plt.title("Validation Accuracy During Training")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    
    plt.show()
