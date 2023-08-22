import random
import math
import numpy as np
import matplotlib.pyplot as plt


# Sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))
# Load the diabetes dataset
data = []
with open("diabetes.txt", "r") as file:
    for line in file:
        values = list(map(float, line.strip().split(",")))
        data.append(values)

# Split the dataset into Training (70%), Validation (15%), and Test (15%) sets
random.seed(42)  # For reproducibility
random.shuffle(data)
total_samples = len(data)
train_size = int(0.7 * total_samples)
val_size = int(0.15 * total_samples)
train_data = data[:train_size]
val_data = data[train_size : train_size + val_size]
test_data = data[train_size + val_size :]

# Training parameters
max_iter = 500

# Get user input for the learning rate
user_lr = float(input("Enter a learning rate (e.g., 0.1, 0.01, 0.001, 0.0001): "))

# Initialize history to store loss
history = []

# Initialize validation accuracy dictionary for the user-selected learning rate
val_acc_dict = {}

# Initialize weights and bias
n_features = len(train_data[0]) - 1
Theta = np.random.rand(n_features + 1)  # Include bias term

# Training loop
for itr in range(1, max_iter + 1):
    total_cost = 0
    dv = np.zeros(n_features + 1)
    for X in train_data:
        X_prime = np.concatenate((X[:-1], [1]))  # Add bias term
        z = np.dot(X_prime, Theta)
        h = sigmoid(z)

        # Update weights using gradient descent
        dv += X_prime * (h - X[-1])
        J = -(X[-1] * math.log(h + 1e-10) + (1 - X[-1]) * math.log(1 - h + 1e-10))
        total_cost += J
    Theta -= user_lr * dv / train_size
    history.append(total_cost / train_size)

    # Calculate validation accuracy for the user-selected learning rate
    correct = 0
    for X in val_data:
        X_prime = np.concatenate((X[:-1], [1]))  # Add bias term
        z = np.dot(X_prime, Theta)
        h = sigmoid(z)
        predicted_label = 1 if h >= 0.5 else 0
        if predicted_label == X[-1]:
            correct += 1
    val_acc = correct * 100 / len(val_data)
    val_acc_dict[user_lr] = val_acc

# Calculate test accuracy using the user-selected learning rate
correct = 0
for X in test_data:
    X_prime = np.concatenate((X[:-1], [1]))  # Add bias term
    z = np.dot(X_prime, Theta)
    h = sigmoid(z)
    predicted_label = 1 if h >= 0.5 else 0
    if predicted_label == X[-1]:
        correct += 1
test_acc = correct * 100 / len(test_data)

# Plot train_loss (history) vs epoch (iteration) graph
plt.plot(history)
plt.xlabel("Iteration")
plt.ylabel("Training Loss")
plt.title("Train Loss vs Iteration")
plt.show()

# Display the user-selected learning rate, its corresponding validation accuracy, and test accuracy
print("User-selected Learning Rate:", user_lr)
print("Validation Accuracy with User-selected LR:", val_acc_dict[user_lr])
print("Test Accuracy with User-selected LR:", test_acc)