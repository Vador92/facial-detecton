import os
import time
import random
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

## testing for loading data


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



# perceptron
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

def test_perceptron(name, folder, img_size, num_classes, n_train=1000):
    print(f"\n--- Perceptron Test: {name} ---")
    X_train, y_train, X_test, y_test = load_data(folder, *img_size)

    if len(X_train) > n_train:
        X_train, y_train = X_train[:n_train], y_train[:n_train]

    model = Perceptron(X_train.shape[1], num_classes)
    model.train(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy on {name}: {acc * 100:.2f}%")

if __name__ == "__main__":
    test_dataset("Digits", "data/digitdata", (28, 28))
    test_dataset("Faces", "data/facedata", (60, 70))
    test_perceptron("Digits", "data/digitdata", (28, 28), num_classes=10)
    test_perceptron("Faces", "data/facedata", (60, 70), num_classes=2)
