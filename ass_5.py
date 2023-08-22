import random
import math
import matplotlib.pyplot as plt

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

# Sigmoid function
def sigmoid(z):
    return 1 / (1 + math.exp(-z))

# Training parameters
max_iter = 500
lr_values = [0.1, 0.01, 0.001, 0.0001]

# Initialize weights and bias
Theta = [random.uniform(0, 1) for _ in range(len(train_data[0]) - 1)]  # Exclude label

# Initialize history to store loss
history = []

# Hyperparameter tuning using validation set
best_lr = None
best_val_acc = 0

for lr in lr_values:
    # Training loop
    for itr in range(1, max_iter + 1):
        total_cost = 0
        dv = [0] * len(Theta)
        for X in train_data:
            X_prime = X[:-1] + [1]  # Add bias term
            z = sum(x * theta for x, theta in zip(X_prime, Theta))
            h = sigmoid(z)

            # Update weights using gradient descent
            for i in range(len(dv)):
                dv[i] += X_prime[i] * (h - X_prime[-1])
            J = -(
                X_prime[-1] * math.log(h + 1e-10)
                + (1 - X_prime[-1]) * math.log(1 - h + 1e-10)
            )
            total_cost += J
        for i in range(len(Theta)):
            Theta[i] -= lr * dv[i] / train_size
        history.append(total_cost / train_size)

    # Calculate validation accuracy for the current learning rate
    correct = 0
    for X in val_data:
        X_prime = X[:-1] + [1]  # Add bias term
        z = sum(x * theta for x, theta in zip(X_prime, Theta))
        h = sigmoid(z)
        predicted_label = 1 if h >= 0.5 else 0
        if predicted_label == X[-1]:
            correct += 1
    val_acc = correct * 100 / len(val_data)

    # Update best_lr and best_val_acc if current validation accuracy is higher
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_lr = lr

# Plot train_loss (history) vs epoch (iteration) graph
plt.plot(history)
plt.xlabel("Iteration")
plt.ylabel("Training Loss")
plt.title("Train Loss vs Iteration")
plt.show()

# Display the best learning rate and its corresponding validation accuracy
print("Best Learning Rate:", best_lr)
print("Best Validation Accuracy:", best_val_acc)

# Calculate test accuracy using the best learning rate
correct = 0
for X in test_data:
    X_prime = X[:-1] + [1]  # Add bias term
    z = sum(x * theta for x, theta in zip(X_prime, Theta))
    h = sigmoid(z)
    predicted_label = 1 if h >= 0.5 else 0
    if predicted_label == X[-1]:
        correct += 1
test_acc = correct * 100 / len(test_data)

# Display the test accuracy
print("Test Accuracy with Best LR:", test_acc)
