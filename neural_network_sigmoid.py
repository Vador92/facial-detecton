import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import matplotlib.pyplot as plt
import random
import os
from torch.utils.data import Dataset, DataLoader, random_split, Subset

# from sklearn.metrics import accuracy_score
# This file is for the sigmoid function and triple layer neural network

class ImageDataset(Dataset):
    """
    Custom PyTorch Dataset for loading digit/face images
    """
    def __init__(self, data, labels):
        # Convert numpy arrays to PyTorch tensors
        # FloatTensor for input features, LongTensor for class labels
        self.data = torch.FloatTensor(data)
        self.labels = torch.LongTensor(labels)
        
    def __len__(self):
        """Return the total number of samples in the dataset"""
        return len(self.data)
    
    def __getitem__(self, idx):
        """Return a specific sample and its label for the given index"""
        return self.data[idx], self.labels[idx]


class ThreeLayerSigmoidNN(nn.Module):
    """
    Three-layer Neural Network with Sigmoid activation
    
    Architecture:
    - Input layer: Takes flattened pixel values (input_size nodes)
    - Hidden layer 1: hidden1_size nodes with Sigmoid activation
    - Hidden layer 2: hidden2_size nodes with Sigmoid activation
    - Output layer: output_size nodes (10 for digits, 2 for faces)
    """
    def __init__(self, input_size, hidden1_size, hidden2_size, output_size):
        """Initialize the network architecture"""
        super(ThreeLayerSigmoidNN, self).__init__()
        
        # First hidden layer
        self.layer1 = nn.Linear(input_size, hidden1_size)
        self.activation1 = nn.Sigmoid()
        
        # Second hidden layer
        self.layer2 = nn.Linear(hidden1_size, hidden2_size)
        self.activation2 = nn.Sigmoid()
        
        # Output layer (no activation - will be handled by loss function)
        self.output_layer = nn.Linear(hidden2_size, output_size)
        
        # Initialize weights using Xavier/Glorot initialization
        nn.init.xavier_uniform_(self.layer1.weight)
        nn.init.xavier_uniform_(self.layer2.weight)
        nn.init.xavier_uniform_(self.output_layer.weight)
        
    def forward(self, x):
        """
        Forward pass through the network
        
        Args:
            x: Input tensor with shape [batch_size, input_size]
            
        Returns:
            Output tensor with shape [batch_size, output_size]
        """
        # Apply first hidden layer and activation
        x = self.activation1(self.layer1(x))
        
        # Apply second hidden layer and activation
        x = self.activation2(self.layer2(x))
        
        # Apply output layer (no activation applied here)
        x = self.output_layer(x)
        
        return x


def load_data(folder_path, img_width, img_height):
    """
    Load and parse data

    Args:
        folder_path: Path to the data directory (digitdata or facedata)
        img_width: Width of the images in pixels
        img_height: Height of the images in pixels
        
    Returns:
        train_imgs: Training images as numpy array
        train_labels: Training labels as numpy array
        test_imgs: Test images as numpy array
        test_labels: Test labels as numpy array
    """
    def parse_img(lines):
        """Parse image lines into binary pixel values"""
        if len(lines) != img_height:
            raise ValueError(f"Expected {img_height} lines per image, got {len(lines)}")
        return [1 if ch != ' ' else 0 for line in lines for ch in line.strip('\n')]
    
    # Set file names based on dataset type
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
    
    # Load raw data from files
    with open(os.path.join(folder_path, train_img_file), 'r') as f:
        raw_imgs = f.readlines()
    with open(os.path.join(folder_path, train_lbl_file), 'r') as f:
        train_labels = [int(l.strip()) for l in f.readlines()]
    with open(os.path.join(folder_path, test_img_file), 'r') as f:
        raw_test_imgs = f.readlines()
    with open(os.path.join(folder_path, test_lbl_file), 'r') as f:
        test_labels = [int(l.strip()) for l in f.readlines()]
    
    # Parse raw images into binary pixel values
    train_imgs = [parse_img(raw_imgs[i:i+img_height]) for i in range(0, len(raw_imgs), img_height)]
    test_imgs = [parse_img(raw_test_imgs[i:i+img_height]) for i in range(0, len(raw_test_imgs), img_height)]
    
    return np.array(train_imgs), np.array(train_labels), np.array(test_imgs), np.array(test_labels)


def train_model(model, train_loader, criterion, optimizer, device, epochs=10):
    """
    Train the model
    
    Args:
        model: The neural network model to train
        train_loader: DataLoader for training data
        criterion: Loss function
        optimizer: Optimization algorithm
        device: Device to train on (CPU or GPU)
        epochs: Number of training epochs
        
    Returns:
        model: Trained model
        losses: List of loss values per epoch
        training_time: Total training time in seconds
    """
    model.train()  # Set model to training mode
    losses = []
    start_time = time.time()
    
    for epoch in range(epochs):
        running_loss = 0.0
        
        # Iterate over mini-batches
        for inputs, targets in train_loader:
            # Move data to device (CPU/GPU)
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Zero the gradients from previous batch
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Backward pass and optimization
            loss.backward()      # Compute gradients
            optimizer.step()     # Update weights
            
            # Accumulate batch loss
            running_loss += loss.item()
            
        # Calculate average loss for the epoch
        epoch_loss = running_loss / len(train_loader)
        losses.append(epoch_loss)
        
        # Print progress every 5 epochs
        if (epoch + 1) % 5 == 0:
            print(f'Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}')
    
    # Calculate total training time
    training_time = time.time() - start_time
    return model, losses, training_time


def evaluate_model(model, test_loader, device):
    """
    Evaluate the model
    
    Args:
        model: The neural network model to evaluate
        test_loader: DataLoader for test data
        device: Device to evaluate on (CPU or GPU)
        
    Returns:
        accuracy: Classification accuracy on test data
    """
    model.eval()  # Set model to evaluation mode
    correct = 0
    total = 0
    
    # Disable gradient calculation for efficiency
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Forward pass
            outputs = model(inputs)
            
            # Get predicted class (highest probability)
            _, predicted = torch.max(outputs.data, 1)
            
            # Count correct predictions
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
    
    # Calculate accuracy
    accuracy = correct / total
    return accuracy


def run_experiment(data_type='digits', iterations=5):
    """
    Run experiment with different percentages of training data
    
    Args:
        data_type: Type of data to use ('digits' or 'faces')
        iterations: Number of iterations for each training percentage
        
    Returns:
        Dictionary containing experiment results
    """
    train_percentages = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    # Set device (GPU if available, otherwise CPU)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load data based on data type
    if data_type == 'digits':
        folder_path = 'data/digitdata'
        img_width, img_height = 28, 28
        train_data, train_labels, test_data, test_labels = load_data(folder_path, img_width, img_height)
        input_size = img_width * img_height  # 28x28 = 784 input features
        output_size = 10  # 10 digits (0-9)
    else:  # faces
        folder_path = 'data/facedata'
        img_width, img_height = 60, 70
        train_data, train_labels, test_data, test_labels = load_data(folder_path, img_width, img_height)
        input_size = img_width * img_height  # 60x70 = 4200 input features
        output_size = 2  # Binary classification (face or not face)
    
    # Create datasets
    train_dataset = ImageDataset(train_data, train_labels)
    test_dataset = ImageDataset(test_data, test_labels)
    
    # Model hyperparameters
    hidden1_size = 128
    hidden2_size = 64
    batch_size = 32
    learning_rate = 0.01  # Learning rate for sigmoid
    epochs = 20
    
    # Results storage
    training_times = []
    accuracies = []
    accuracy_stds = []
    
    # Main experiment loop
    for percentage in train_percentages:
        print(f"\nTraining with {percentage*100}% of the training data")
        
        # Calculate number of training examples to use
        n_train = int(len(train_dataset) * percentage)
        
        # Results for current percentage
        cur_times = []
        cur_accuracies = []
        
        # Run multiple iterations with different random samples
        for iteration in range(iterations):
            print(f"Iteration {iteration+1}/{iterations}")
            
            # Randomly sample training data
            indices = torch.randperm(len(train_dataset))[:n_train]
            sampled_train_dataset = Subset(train_dataset, indices)
            
            # Create data loaders
            train_loader = DataLoader(sampled_train_dataset, batch_size=batch_size, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=batch_size)
            
            # Initialize model
            model = ThreeLayerSigmoidNN(input_size, hidden1_size, hidden2_size, output_size).to(device)
            
            # Loss function and optimizer
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)
            
            # Train model
            _, _, training_time = train_model(model, train_loader, criterion, optimizer, device, epochs)
            cur_times.append(training_time)
            
            # Evaluate model
            accuracy = evaluate_model(model, test_loader, device)
            cur_accuracies.append(accuracy)
            
            print(f"Accuracy: {accuracy:.4f}, Training time: {training_time:.2f}s")
        
        # Calculate and store average results
        training_times.append(np.mean(cur_times))
        accuracies.append(np.mean(cur_accuracies))
        accuracy_stds.append(np.std(cur_accuracies))
        
        print(f"Average accuracy: {np.mean(cur_accuracies):.4f} ± {np.std(cur_accuracies):.4f}")
        print(f"Average training time: {np.mean(cur_times):.2f}s")
    
    # Plot results
    plot_results(train_percentages, training_times, accuracies, accuracy_stds, data_type)
    
    return {
        'train_percentages': train_percentages,
        'training_times': training_times,
        'accuracies': accuracies,
        'accuracy_stds': accuracy_stds
    }


def plot_results(train_percentages, training_times, accuracies, accuracy_stds, data_type):
    """
    Plot training time and accuracy results
    
    Args:
        train_percentages: List of training data percentages
        training_times: List of average training times
        accuracies: List of average accuracies
        accuracy_stds: List of accuracy standard deviations
        data_type: Type of data ('digits' or 'faces')
    """
    percentages = [p * 100 for p in train_percentages]
    
    plt.figure(figsize=(12, 5))
    
    # Plot training time
    plt.subplot(1, 2, 1)
    plt.plot(percentages, training_times, 'o-', color='blue')
    plt.xlabel('Percentage of Training Data (%)')
    plt.ylabel('Training Time (s)')
    plt.title(f'Training Time vs. Training Data Size ({data_type})')
    plt.grid(True)
    
    # Plot accuracy with error bars (± standard deviation)
    plt.subplot(1, 2, 2)
    plt.errorbar(percentages, accuracies, yerr=accuracy_stds, fmt='o-', color='green', capsize=5)
    plt.xlabel('Percentage of Training Data (%)')
    plt.ylabel('Accuracy')
    plt.title(f'Accuracy vs. Training Data Size ({data_type})')
    plt.grid(True)
    
    # Save and display the figure
    plt.tight_layout()
    plt.savefig(f'{data_type}_sigmoid_results.png')
    plt.show()


def main():
    """Main function to run the experiment"""
    print("PyTorch Neural Network with Sigmoid Activation for Image Classification")
    
    # Check if data directory exists
    if not os.path.exists('data'):
        print("Error: 'data' directory not found.")
        return
    
    # Run experiment for digits
    print("\n=== Digit Classification ===")
    digit_results = run_experiment(data_type='digits')
    
    # Run experiment for faces
    print("\n=== Face Classification ===")
    face_results = run_experiment(data_type='faces')
    
    print("\nExperiments completed!")


if __name__ == "__main__":
    main()

    
