import random
import matplotlib.pyplot as plt

def distance(point1, point2):
    return sum((x - y) ** 2 for x, y in zip(point1, point2))

def find_closest_center(data_point, centers):
    min_distance = float('inf')
    closest_center = None

    for center in centers:
        dist = distance(data_point, center)
        if dist < min_distance:
            min_distance = dist
            closest_center = center

    return closest_center

def k_means_clustering(K, data):
    Centers = random.sample(data, K)  # Randomly select K different data points as initial centers

    Clusters = [[] for _ in range(K)]  # Initialize K empty lists for the K centers

    itr = 1
    Shift = 0

    while True:
        for data_point in data:
            closest_center = find_closest_center(data_point, Centers)
            index = Centers.index(closest_center)
            Clusters[index].append(data_point)

        new_centers = []
        for i in range(K):
            new_center = [sum(col) / len(col) for col in zip(*Clusters[i])]
            new_centers.append(new_center)

        # Calculate the shift in centers
        Shift = sum(distance(old_center, new_center) for old_center, new_center in zip(Centers, new_centers))

        Centers = new_centers

        # Clear the Clusters for the next iteration
        Clusters = [[] for _ in range(K)]

        if itr > 1 and Shift < 50:
            break

        itr += 1

    # Calculate the inertia
    inertia = 0
    for i in range(K):
        inertia += sum(distance(data_point, Centers[i]) ** 2 for data_point in Clusters[i])

    return Centers, Clusters, inertia

def plot_clusters(data, centers, clusters):
    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']

    for i in range(len(clusters)):
        cluster = clusters[i]
        if len(cluster) > 0:
            color = colors[i % len(colors)]
            x, y = zip(*cluster)
            plt.scatter(x, y, c=color, label=f'Cluster {i+1}')

    x_centers, y_centers = zip(*centers)
    plt.scatter(x_centers, y_centers, c='black', marker='X', s=100, label='Centers')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('K-means Clustering')
    plt.legend()
    plt.show()

# Load the dataset from the text file
file_path = '/content/jain_feats.txt'  # Replace 'path_to_your_text_file.txt' with the actual file path
with open(file_path, 'r') as file:
    g_data = [[float(num) for num in line.strip().split()] for line in file]

# Example usage:

K_values = [2, 4, 6, 7]

inertias = []

for K in K_values:
    Centers, Clusters, inertia = k_means_clustering(K, g_data)
    inertias.append(inertia)
    plot_clusters(g_data, Centers, Clusters)

print("K values:", K_values)
print("Inertia values:", inertias)