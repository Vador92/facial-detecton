import numpy as np
import time
from neural_network_sigmoid import SigmoidNeuralNetwork
from face_digit_classification import load_data  # or from neural_network import load_data

# Set these for digits or faces
folder = "data/digitdata"
img_size = (28, 28)
output_size = 10

# Load data (this function returns: X_train, y_train, X_test, y_test)
X_train, y_train, X_test, y_test = load_data(folder, *img_size)

percentages = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
num_repeats = 5
input_size = X_train.shape[1]
hidden1_size = 128
hidden2_size = 64
learning_rate = 0.01
num_epochs = 20
batch_size = 32

results = []

for pct in percentages:
    times = []
    errors = []
    for repeat in range(num_repeats):
        idx = np.random.choice(len(X_train), int(pct * len(X_train)), replace=False)
        X_sub = X_train[idx]
        y_sub = y_train[idx]

        model = SigmoidNeuralNetwork(input_size, hidden1_size, hidden2_size, output_size, learning_rate)
        start = time.time()
        model.train(X_sub, y_sub, epochs=num_epochs, batch_size=batch_size, verbose=False)
        end = time.time()
        times.append(end - start)

        error = 1 - model.evaluate(X_test, y_test)
        errors.append(error)

    results.append({
        'percentage': pct,
        'avg_time': np.mean(times),
        'std_time': np.std(times),
        'avg_error': np.mean(errors),
        'std_error': np.std(errors)
    })

print(f"{'% Train':>8} | {'Avg Time (s)':>12} | {'Std Time':>8} | {'Avg Error':>10} | {'Std Error':>10}")
print('-'*60)
for res in results:
    print(f"{res['percentage']*100:8.0f} | {res['avg_time']:12.4f} | {res['std_time']:8.4f} | {res['avg_error']:10.4f} | {res['std_error']:10.4f}") 