"""
Group: Varun Doreswamy, Seth Yeh, Nicholas Kushnir
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import matplotlib.pyplot as plt
import random
import os
from torch.utils.data import Dataset, DataLoader, Subset

class ImageDataset(Dataset):
    def __init__(self, data, labels):
        self.data = torch.FloatTensor(data)
        self.labels = torch.LongTensor(labels)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

class ThreeLayerNN(nn.Module):
    def __init__(self, input_size, hidden1_size, hidden2_size, output_size):
        super(ThreeLayerNN, self).__init__()
        
        self.layer1 = nn.Linear(input_size, hidden1_size)
        self.activation1 = nn.ReLU()
        
        self.layer2 = nn.Linear(hidden1_size, hidden2_size)
        self.activation2 = nn.ReLU()
        
        self.output_layer = nn.Linear(hidden2_size, output_size)
        
    def forward(self, x):
        x = self.activation1(self.layer1(x))
        x = self.activation2(self.layer2(x))
        x = self.output_layer(x)
        return x

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

def train_model(model, train_loader, criterion, optimizer, device, epochs=10):
    model.train() 
    losses = []
    start_time = time.time()
    
    for epoch in range(epochs):
        running_loss = 0.0
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)
        
            loss.backward()      
            optimizer.step()     
            
            running_loss += loss.item()
            
        epoch_loss = running_loss / len(train_loader)
        losses.append(epoch_loss)
        
        if (epoch + 1) % 5 == 0:
            print(f'Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}')
    
    training_time = time.time() - start_time
    return model, losses, training_time

def evaluate_model(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            outputs = model(inputs)
            
            _, predicted = torch.max(outputs.data, 1)
            
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
    
    accuracy = correct / total
    return accuracy

def run_experiment(data_type='digits', iterations=5):
    train_percentages = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    if data_type == 'digits':
        folder_path = 'data/digitdata'
        img_width, img_height = 28, 28
        train_data, train_labels, test_data, test_labels = load_data(folder_path, img_width, img_height)
        input_size = img_width * img_height  # 28x28 = 784 input features
        output_size = 10  
    else: 
        folder_path = 'data/facedata'
        img_width, img_height = 60, 70
        train_data, train_labels, test_data, test_labels = load_data(folder_path, img_width, img_height)
        input_size = img_width * img_height
        output_size = 2 
    
    train_dataset = ImageDataset(train_data, train_labels)
    test_dataset = ImageDataset(test_data, test_labels)
    
    hidden1_size = 128
    hidden2_size = 64
    batch_size = 32
    learning_rate = 0.001
    epochs = 20
    
    training_times = []
    accuracies = []
    accuracy_stds = []
    
    for percentage in train_percentages:
        print(f"\nTraining with {percentage*100}% of the training data")
        
        n_train = int(len(train_dataset) * percentage)
        
        cur_times = []
        cur_accuracies = []
        
        for iteration in range(iterations):
            print(f"Iteration {iteration+1}/{iterations}")
            
            indices = torch.randperm(len(train_dataset))[:n_train]
            sampled_train_dataset = Subset(train_dataset, indices)
            
            train_loader = DataLoader(sampled_train_dataset, batch_size=batch_size, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=batch_size)
            
            model = ThreeLayerNN(input_size, hidden1_size, hidden2_size, output_size).to(device)
            
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)
            
            _, _, training_time = train_model(model, train_loader, criterion, optimizer, device, epochs)
            cur_times.append(training_time)
            
            accuracy = evaluate_model(model, test_loader, device)
            cur_accuracies.append(accuracy)
            
            print(f"Accuracy: {accuracy:.4f}, Training time: {training_time:.2f}s")
        
        training_times.append(np.mean(cur_times))
        accuracies.append(np.mean(cur_accuracies))
        accuracy_stds.append(np.std(cur_accuracies))
        
        print(f"Average accuracy: {np.mean(cur_accuracies):.4f} ± {np.std(cur_accuracies):.4f}")
        print(f"Average training time: {np.mean(cur_times):.2f}s")
    
    plot_results(train_percentages, training_times, accuracies, accuracy_stds, data_type)
    
    return {
        'train_percentages': train_percentages,
        'training_times': training_times,
        'accuracies': accuracies,
        'accuracy_stds': accuracy_stds
    }

def plot_results(train_percentages, training_times, accuracies, accuracy_stds, data_type):
    percentages = [p * 100 for p in train_percentages]
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(percentages, training_times, 'o-', color='blue')
    plt.xlabel('Percentage of Training Data (%)')
    plt.ylabel('Training Time (s)')
    plt.title(f'Training Time vs. Training Data Size ({data_type})')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.errorbar(percentages, accuracies, yerr=accuracy_stds, fmt='o-', color='green', capsize=5)
    plt.xlabel('Percentage of Training Data (%)')
    plt.ylabel('Accuracy')
    plt.title(f'Accuracy vs. Training Data Size ({data_type})')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(f'extras/results/{data_type}_pytorch_results.png')
    plt.show()

def main():
    print("PyTorch Neural Network for Image Classification")
    
    if not os.path.exists('data'):
        print("Error: 'data' directory not found.")   
        return
    print("\n=== Digit Classification ===")
    run_experiment(data_type='digits')
    print("\n=== Face Classification ===")
    run_experiment(data_type='faces')
    
    print("\nExperiments completed!")

if __name__ == "__main__":
    main()