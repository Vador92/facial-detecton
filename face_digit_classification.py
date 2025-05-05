"""
Group: Varun Doreswamy, Seth Yeh, Nicholas Kushnir
"""

import os
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

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


def test_dataset(name, folder, img_size):
    print(f"\n--- Data Loading Test: {name} ---")
    X_train, y_train, X_test, y_test = load_data(folder, *img_size)
    print("Train X shape:", X_train.shape)
    print("Train y shape:", y_train.shape)
    print("Test X shape:", X_test.shape)
    print("Test y shape:", y_test.shape)

    plt.imshow(X_train[0].reshape(img_size[1], img_size[0]), cmap='gray')
    plt.title(f"{name} Sample Label: {y_train[0]}")
    plt.show()


class Perceptron:
    def __init__(self, num_features: int, num_classes: int):
        self.weights = np.zeros((num_classes, num_features))

    def train(self, X, y, iters: int = 10):
        for _ in range(iters):
            for i in range(X.shape[0]):
                x_i = X[i]
                true_label = y[i]
                pred = np.argmax(self.weights @ x_i)
                if pred != true_label:
                    self.weights[true_label] += x_i
                    self.weights[pred] -= x_i

    def predict(self, X):
        return np.argmax(X @ self.weights.T, axis=1)

def run_perceptron_experiment(name, folder, img_size, num_classes):
    X_train, y_train, X_test, y_test = load_data(folder, *img_size)

    portions = [0.1 * i for i in range(1, 11)]
    mean_acc, std_acc, times = [], [], []

    for p in portions:
        accs = []
        total_time = 0
        for _ in range(5):  # 5 trials per portion
            idx = np.random.choice(len(X_train), int(p * len(X_train)), replace=False)
            X_sub, y_sub = X_train[idx], y_train[idx]

            model = Perceptron(X_sub.shape[1], num_classes)
            start = time.time()
            model.train(X_sub, y_sub)
            total_time += time.time() - start

            y_pred = model.predict(X_test)
            accs.append(accuracy_score(y_test, y_pred))

        mean_acc.append(np.mean(accs))
        std_acc.append(np.std(accs))
        times.append(total_time / 5)

    x_vals = [int(p * 100) for p in portions]

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(x_vals, times, 'bo-', label='Training Time')
    plt.xlabel("Percentage of Training Data (%)")
    plt.ylabel("Training Time (s)")
    plt.title(f"Training Time vs. Training Data Size ({name.lower()})")

    plt.subplot(1, 2, 2)
    plt.errorbar(x_vals, mean_acc, yerr=std_acc, fmt='g-', ecolor='black', capsize=4)
    plt.xlabel("Percentage of Training Data (%)")
    plt.ylabel("Accuracy")
    plt.title(f"Accuracy vs. Training Data Size ({name.lower()})")

    plt.tight_layout()
    plt.savefig(f"extras/results/perceptron_{name.lower()}_results.png")
    plt.show()

if __name__ == "__main__":
    test_dataset("Digits", "data/digitdata", (28, 28))
    test_dataset("Faces", "data/facedata", (60, 70))
    run_perceptron_experiment("Digits", "data/digitdata", (28, 28), num_classes=10)
    run_perceptron_experiment("Faces", "data/facedata", (60, 70), num_classes=2)
