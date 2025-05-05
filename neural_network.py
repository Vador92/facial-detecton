import os
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

# load Files
def load_data(folder_path, img_width, img_height):
    def parse_img(lines):
        if len(lines) != img_height:
            raise ValueError(f"Expected {img_height} lines per image, got {len(lines)}")
        return [1 if ch != ' ' else 0 for line in lines for ch in line.strip('\n')]
    
    if "digitdata" in folder_path:
        train_img_file = "trainingimages"
        train_lbl_file = "traininglabels"
        test_img_file = "testimages"
        test_lbl_file = "testlabels"
        
    else:
        train_img_file = "facedatatrain"
        train_lbl_file = "facedatatrainlabels"
        test_img_file = "facedatatest"
        test_lbl_file = "facedatatestlabels"
    
    with open(os.path.join(folder_path, train_img_file), 'r') as f:
        raw_imgs = f.readlines()
    with open(os.path.join(folder_path, train_lbl_file), 'r') as f:
        train_labels = [int(l.strip()) for l in f.readlines()]
    with open(os.path.join(folder_path, test_img_file), 'r') as f:
        raw_test_imgs = f.readlines()
    with open(os.path.join(folder_path, test_lbl_file), 'r') as f:
        test_labels = [int(l.strip()) for l in f.readlines()]
    
    train_imgs = [parse_img(raw_imgs[i:i+img_height]) for i in range(0, len(raw_imgs), img_height)]
    test_imgs = [parse_img(raw_test_imgs[i:i+img_height]) for i in range(0, len(raw_test_imgs), img_height)]
    
    return np.array(train_imgs), np.array(train_labels), np.array(test_imgs), np.array(test_labels)

class ImprovedNeuralNetwork:
    def __init__(self, input_size, hidden1_size, hidden2_size, output_size, learning_rate=0.001):
        # Xavier/Glorot initialization for better convergence
        self.weights1 = np.random.randn(input_size, hidden1_size) * np.sqrt(2 / (input_size + hidden1_size))
        self.bias1 = np.zeros((1, hidden1_size))
        
        self.weights2 = np.random.randn(hidden1_size, hidden2_size) * np.sqrt(2 / (hidden1_size + hidden2_size))
        self.bias2 = np.zeros((1, hidden2_size))
        
        self.weights3 = np.random.randn(hidden2_size, output_size) * np.sqrt(2 / (hidden2_size + output_size))
        self.bias3 = np.zeros((1, output_size))
        
        self.learning_rate = learning_rate
        
        self.velocity_w1 = np.zeros_like(self.weights1)
        self.velocity_b1 = np.zeros_like(self.bias1)
        self.velocity_w2 = np.zeros_like(self.weights2)
        self.velocity_b2 = np.zeros_like(self.bias2)
        self.velocity_w3 = np.zeros_like(self.weights3)
        self.velocity_b3 = np.zeros_like(self.bias3)
        self.momentum = 0.9
    
    def relu(self, x):
        return np.maximum(0, x)
    
    def relu_derivative(self, x):
        return np.where(x > 0, 1, 0)
    
    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True)) 
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    def forward(self, X):
        self.z1 = np.dot(X, self.weights1) + self.bias1
        self.a1 = self.relu(self.z1)
        
        self.z2 = np.dot(self.a1, self.weights2) + self.bias2
        self.a2 = self.relu(self.z2)
        
        self.z3 = np.dot(self.a2, self.weights3) + self.bias3
        self.a3 = self.softmax(self.z3)
        
        return self.a3
    
    def backward(self, X, y, output):
        batch_size = X.shape[0]
        
        y_one_hot = np.zeros((batch_size, output.shape[1]))
        y_one_hot[np.arange(batch_size), y] = 1
        
        delta3 = output - y_one_hot
        delta2 = np.dot(delta3, self.weights3.T) * self.relu_derivative(self.z2)
        delta1 = np.dot(delta2, self.weights2.T) * self.relu_derivative(self.z1)
        
        dW3 = np.dot(self.a2.T, delta3) / batch_size
        db3 = np.sum(delta3, axis=0, keepdims=True) / batch_size
        dW2 = np.dot(self.a1.T, delta2) / batch_size
        db2 = np.sum(delta2, axis=0, keepdims=True) / batch_size
        dW1 = np.dot(X.T, delta1) / batch_size
        db1 = np.sum(delta1, axis=0, keepdims=True) / batch_size
        
        self.velocity_w3 = self.momentum * self.velocity_w3 - self.learning_rate * dW3
        self.velocity_b3 = self.momentum * self.velocity_b3 - self.learning_rate * db3
        self.velocity_w2 = self.momentum * self.velocity_w2 - self.learning_rate * dW2
        self.velocity_b2 = self.momentum * self.velocity_b2 - self.learning_rate * db2
        self.velocity_w1 = self.momentum * self.velocity_w1 - self.learning_rate * dW1
        self.velocity_b1 = self.momentum * self.velocity_b1 - self.learning_rate * db1
        
        self.weights3 += self.velocity_w3
        self.bias3 += self.velocity_b3
        self.weights2 += self.velocity_w2
        self.bias2 += self.velocity_b2
        self.weights1 += self.velocity_w1
        self.bias1 += self.velocity_b1
    
    def train(self, X, y, epochs=100, batch_size=32, verbose=True):
        num_samples, losses = X.shape[0], []
        
        for epoch in range(epochs):
            indices = np.random.permutation(num_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            
            for i in range(0, num_samples, batch_size):
                end = min(i + batch_size, num_samples)
                X_batch = X_shuffled[i:end]
                y_batch = y_shuffled[i:end]
                
                output = self.forward(X_batch)
                
                self.backward(X_batch, y_batch, output)
            
            if verbose and epoch % 5 == 0:
                output = self.forward(X)
                y_one_hot = np.zeros((num_samples, output.shape[1]))
                y_one_hot[np.arange(num_samples), y] = 1
                loss = -np.sum(y_one_hot * np.log(np.clip(output, 1e-15, 1.0))) / num_samples
                losses.append(loss)
                
                predictions = np.argmax(output, axis=1)
                accuracy = np.mean(predictions == y)
                
                print(f"Epoch {epoch}, Loss: {loss:.4f}, Training Accuracy: {accuracy:.4f}")
        
        return losses
    
    def predict(self, X):
        return np.argmax(self.forward(X), axis=1)
    
    def evaluate(self, X, y):
        return accuracy_score(y, self.predict(X))

def test_neural_network(name, folder, img_size, num_classes, training_percentages=None):
    if training_percentages is None: training_percentages = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    
    print(f"\n--- Improved Neural Network Test: {name} ---")
    
    X_train, y_train, X_test, y_test = load_data(folder, *img_size)
    accuracies, training_times, std_devs = [], [], []
    
    for percentage in training_percentages:
        print(f"\nTraining with {percentage}% of data")
        
        n_samples = int(len(X_train) * percentage / 100)
        
        run_accuracies = []
        run_times = []
        
        for run in range(3):
            indices = np.random.choice(len(X_train), n_samples, replace=False)
            X_train_subset = X_train[indices]
            y_train_subset = y_train[indices]
            
            if "digit" in name.lower():
                hidden1_size = 128  # Larger network
                hidden2_size = 64
                epochs = 50        # More epochs
                batch_size = 32
                learning_rate = 0.001  # Lower learning rate
            else: 
                hidden1_size = 64
                hidden2_size = 32
                epochs = 30
                batch_size = 16
                learning_rate = 0.001
            
            input_size = X_train_subset.shape[1]
            model = ImprovedNeuralNetwork(input_size, hidden1_size, hidden2_size, num_classes, learning_rate)
            
            start_time = time.time()
            model.train(X_train_subset, y_train_subset, epochs=epochs, batch_size=batch_size, verbose=(run==0))
            train_time = time.time() - start_time
            
            accuracy = model.evaluate(X_test, y_test)
            
            run_accuracies.append(accuracy)
            run_times.append(train_time)
            
            print(f"  Run {run+1}: Accuracy = {accuracy:.4f}, Training time = {train_time:.2f}s")
        
        mean_accuracy = np.mean(run_accuracies)
        std_accuracy = np.std(run_accuracies)
        mean_time = np.mean(run_times)
        
        accuracies.append(mean_accuracy)
        std_devs.append(std_accuracy)
        training_times.append(mean_time)
        
        print(f"Average accuracy: {mean_accuracy:.4f} ± {std_accuracy:.4f}")
        print(f"Average training time: {mean_time:.2f}s")
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 2)
    plt.errorbar(training_percentages, np.array(accuracies) * 100, yerr=np.array(std_devs) * 100, capsize=5)
    plt.xlabel('Training Data Percentage (%)')
    plt.ylabel('Test Accuracy (%)')
    plt.title(f'Neural Network Accuracy on {name}')
    plt.grid(True)
    
    # Training time plot
    plt.subplot(1, 2, 1)
    plt.plot(training_percentages, training_times)
    plt.xlabel('Training Data Percentage (%)')
    plt.ylabel('Training Time (s)')
    plt.title('Training Time')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(f'extras/results/{name}_reLu_results.png')
    plt.show()
    
    return accuracies, std_devs, training_times

if __name__ == "__main__":
    # digits dataset
    test_neural_network("Digits", "data/digitdata", (28, 28), 10)
    
    # faces dataset
    test_neural_network("Faces", "data/facedata", (60, 70), 2)