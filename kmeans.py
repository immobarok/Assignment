import random
import math
import matplotlib.pyplot as plt

# Set the random seed based on your student ID
random.seed(201300)

# Read data points from a file
Data = []
with open("jain_feats.txt", "r") as file:
    for line in file:
        x, y = map(float, line.strip().split())
        Data.append([x, y])


# Function to calculate Euclidean distance between two points
def distance(point1, point2):
    return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)


# Function to find the closest center for a data point
def closest_center(data_point, centers):
    distances = [distance(data_point, center) for center in centers]
    return distances.index(min(distances))


# Step 1: Select K random data points as the initial centers of K clusters
K_values = list(map(int, input("Enter multiple K values (separated by spaces): ").split()))
Centers_list = []

for K in K_values:
    Centers = random.sample(Data, K)
    Centers_list.append(Centers)

# List to store cluster assignments
ClusterAssignments = [0] * len(Data)

# Continuously iterate until convergence
while True:
    # Step 2: Assign each data point to the closest cluster
    for i, S in enumerate(Data):
        closest_idx = closest_center(S, Centers)
        ClusterAssignments[i] = closest_idx

    # Store previous centers for convergence check
    prev_centers = Centers.copy()

    # Step 4: Recalculate the center of each cluster
    for i in range(K):
        cluster_points = [
            Data[j] for j in range(len(Data)) if ClusterAssignments[j] == i
        ]
        if cluster_points:
            center_x = sum(point[0] for point in cluster_points) / len(cluster_points)
            center_y = sum(point[1] for point in cluster_points) / len(cluster_points)
            Centers[i] = [center_x, center_y]

    # Step 5: Reassign each data point to the closest cluster
    for i, S in enumerate(Data):
        closest_idx = closest_center(S, Centers)
        ClusterAssignments[i] = closest_idx

    # Check for convergence
    if Centers == prev_centers:
        break

# Print final clusters and centers
for i, center in enumerate(Centers):
    cluster_points = [Data[j] for j in range(len(Data)) if ClusterAssignments[j] == i]
    print(f"Cluster {i+1} Center: {center}")
    print(f"Cluster {i+1} Points: {cluster_points}")
    print()

# Plot clusters
colors = ["r", "g", "b", "y"]
for i, center in enumerate(Centers):
    cluster_points = [Data[j] for j in range(len(Data)) if ClusterAssignments[j] == i]
    x_values = [point[0] for point in cluster_points]
    y_values = [point[1] for point in cluster_points]
    plt.scatter(x_values, y_values, c=colors[i], label=f"Cluster {i+1}")
    plt.scatter(
        center[0], center[1], c="black", marker="x", s=100, label=f"Center {i+1}"
    )
    sum_squared_distances = []

# Loop over different K values to find the optimal number of clusters
for k in range(1, 11):
    Centers = random.sample(Data, k)
    ClusterAssignments = [0] * len(Data)

    # ... (K-means clustering steps)

    # Calculate the sum of squared distances for the current K value
    sum_squared = 0
    for i, center in enumerate(Centers):
        cluster_points = [Data[j] for j in range(len(Data)) if ClusterAssignments[j] == i]
        for point in cluster_points:
            sum_squared += distance(point, center) ** 2
    sum_squared_distances.append(sum_squared)
plt.legend()
plt.xlabel("X")
plt.ylabel("Y")
plt.title("K-means Clustering")
plt.show()
plt.plot(range(1, 11), sum_squared_distances, marker='o')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Sum of Squared Distances')
plt.title('Elbow Method for Optimal K')
plt.show()
