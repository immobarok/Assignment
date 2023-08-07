import random
import matplotlib.pyplot as plt

# Load the dataset into a 2D list named "Data"
# Assuming you have loaded the dataset into the variable "g_data"

# Set your student ID as seed
student_id = 123456
random.seed(student_id)

def calculate_distance(point1, point2):
    return sum((a - b)**2 for a, b in zip(point1, point2))

def assign_to_clusters(Data, Centers):
    Clusters = [[] for _ in Centers]
    for S in Data:
        min_dist = float('inf')
        min_idx = -1
        for i, C in enumerate(Centers):
            dist = calculate_distance(S, C)
            if dist < min_dist:
                min_dist = dist
                min_idx = i
        Clusters[min_idx].append(S)
    return Clusters

def calculate_new_centers(Clusters):
    return [[sum(col) / len(cluster) for col in zip(*cluster)] for cluster in Clusters]

def k_means_clustering(Data, K):
    Centers = random.sample(Data, K)
    
    while True:
        Clusters = assign_to_clusters(Data, Centers)
        
        new_centers = calculate_new_centers(Clusters)
        
        if new_centers == Centers:
            break
        
        Centers = new_centers
    
    inertia = 0
    for i in range(K):
        for S in Clusters[i]:
            inertia += sum((a - b)**2 for a, b in zip(S, Centers[i]))
    
    return Clusters, Centers, inertia

K_values = [2, 4, 6, 7]

for K in K_values:
    Clusters, Centers, inertia = k_means_clustering(Data, K)
    
    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
    
    for i, cluster in enumerate(Clusters):
        x_vals = [point[0] for point in cluster]
        y_vals = [point[1] for point in cluster]
        plt.scatter(x_vals, y_vals, color=colors[i], label=f'Cluster {i+1}')
    
    center_x = [center[0] for center in Centers]
    center_y = [center[1] for center in Centers]
    plt.scatter(center_x, center_y, color='black', marker='x', label='Centers')
    
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title(f'K-means Clustering (K={K})')
    plt.legend()
    plt.show()
    
    print(f'K = {K}, Inertia: {inertia}')
