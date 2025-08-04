import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances
import random
from tqdm import tqdm

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

def generate_fake_data():
    # Fog nodes in 4 layers (L1, L2, L3, L4)
    layers = {
        "L1": {"count": 200, "cpu_range": (1, 5), "mem_range": (1, 5), "bw_range": (1, 5)},
        "L2": {"count": 100, "cpu_range": (5, 10), "mem_range": (5, 10), "bw_range": (5, 10)},
        "L3": {"count": 50, "cpu_range": (10, 15), "mem_range": (10, 15), "bw_range": (10, 15)},
        "L4": {"count": 25, "cpu_range": (15, 20), "mem_range": (15, 20), "bw_range": (15, 20)}
    }
    fog_nodes = []
    for layer, specs in layers.items():
        for _ in range(specs["count"]):
            node = {
                "layer": layer,
                "cpu": np.random.randint(*specs["cpu_range"]),
                "mem": np.random.randint(*specs["mem_range"]),
                "bw": np.random.randint(*specs["bw_range"]),
                "x": np.random.uniform(0, 99),
                "y": np.random.uniform(0, 99)
            }
            fog_nodes.append(node)
    # IoT devices with resource requests
    iot_devices = []
    for _ in range(20):  # 20 devices as in paper
        device = {
            "cpu_req": np.random.randint(1, 15),
            "mem_req": np.random.randint(1, 15),
            "bw_req": np.random.randint(1, 15),
            "x": np.random.uniform(0, 99),
            "y": np.random.uniform(0, 99),
            "max_delay": np.random.uniform(5, 20)  # Tolerable delay
        }
        iot_devices.append(device)
    return fog_nodes, iot_devices

def cluster_fog_nodes(fog_nodes, n_clusters=5):
    # Extract locations for clustering
    locations = np.array([[node["x"], node["y"]] for node in fog_nodes])
    # Apply k-means
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(locations)
    cluster_centers = kmeans.cluster_centers_
    # Assign clusters to fog nodes
    for i, node in enumerate(fog_nodes):
        node["cluster"] = cluster_labels[i]
    # Calculate cluster resources
    clusters = []
    for i in range(n_clusters):
        cluster_nodes = [node for node in fog_nodes if node["cluster"] == i]
        total_cpu = sum(node["cpu"] for node in cluster_nodes)
        total_mem = sum(node["mem"] for node in cluster_nodes)
        total_bw = sum(node["bw"] for node in cluster_nodes)
        clusters.append({
            "id": i,
            "center": cluster_centers[i],
            "nodes": cluster_nodes,
            "total_cpu": total_cpu,
            "total_mem": total_mem,
            "total_bw": total_bw
        })
    return clusters

class GeneticAlgorithm:
    def __init__(self, devices, clusters, pop_size=50, generations=100, mutation_rate=0.1):
        self.devices = devices
        self.clusters = clusters
        self.pop_size = pop_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.n_devices = len(devices)
        self.n_clusters = len(clusters)
        # Initialize population
        self.population = self.initialize_population()
        self.best_solution = None
        self.best_fitness = float("inf")
        self.fitness_history = []
    def initialize_population(self):
        # Each chromosome is a list of cluster assignments for each device
        population = []
        for _ in range(self.pop_size):
            chromosome = []
            for device in self.devices:
                # Find clusters that can satisfy device requirements
                eligible_clusters = [c["id"] for c in self.clusters 
                                   if c["total_cpu"] >= device["cpu_req"] and 
                                      c["total_mem"] >= device["mem_req"] and 
                                      c["total_bw"] >= device["bw_req"]]
                if eligible_clusters:
                    chromosome.append(random.choice(eligible_clusters))
                else:
                    # If no cluster can satisfy, assign randomly (penalized in fitness)
                    chromosome.append(random.randint(0, self.n_clusters - 1))
            population.append(chromosome)
        return population
    def calculate_fitness(self, chromosome):
        # Calculate total distance (WCD)
        total_distance = 0
        penalty = 0
        for i, cluster_id in enumerate(chromosome):
            device = self.devices[i]
            cluster = self.clusters[cluster_id]
            # Calculate distance between device and cluster center
            dx = device["x"] - cluster["center"][0]
            dy = device["y"] - cluster["center"][1]
            distance = np.sqrt(dx**2 + dy**2)
            # Check if cluster can satisfy device requirements
            if (cluster["total_cpu"] < device["cpu_req"] or 
                cluster["total_mem"] < device["mem_req"] or 
                cluster["total_bw"] < device["bw_req"]):
                # Add penalty for unsatisfied requirements
                penalty += 1000
            # Check if distance exceeds device's max delay
            if distance > device["max_delay"]:
                # Add penalty for exceeding delay
                penalty += 500
            total_distance += distance
        return total_distance + penalty
    def selection(self):
        # Tournament selection
        selected = []
        for _ in range(self.pop_size):
            candidates = random.sample(self.population, 3)
            fitnesses = [self.calculate_fitness(chromo) for chromo in candidates]
            winner = candidates[np.argmin(fitnesses)]
            selected.append(winner)
        return selected
    def crossover(self, parent1, parent2):
        # Uniform crossover
        child1, child2 = [], []
        for i in range(self.n_devices):
            if random.random() < 0.5:
                child1.append(parent1[i])
                child2.append(parent2[i])
            else:
                child1.append(parent2[i])
                child2.append(parent1[i])
        return child1, child2
    def mutate(self, chromosome):
        # Random mutation
        for i in range(self.n_devices):
            if random.random() < self.mutation_rate:
                # Find eligible clusters for this device
                device = self.devices[i]
                eligible_clusters = [c["id"] for c in self.clusters 
                                   if c["total_cpu"] >= device["cpu_req"] and 
                                      c["total_mem"] >= device["mem_req"] and 
                                      c["total_bw"] >= device["bw_req"]]
                if eligible_clusters:
                    chromosome[i] = random.choice(eligible_clusters)
                else:
                    chromosome[i] = random.randint(0, self.n_clusters - 1)
        return chromosome
    def run(self):
        for generation in tqdm(range(self.generations), desc="GA Evolution"):
            # Calculate fitness for all chromosomes
            fitnesses = [self.calculate_fitness(chromo) for chromo in self.population]
            # Track best solution
            min_fitness = min(fitnesses)
            if min_fitness < self.best_fitness:
                self.best_fitness = min_fitness
                self.best_solution = self.population[np.argmin(fitnesses)]
            self.fitness_history.append(min_fitness)
            # Selection
            selected = self.selection()
            # Crossover and mutation
            new_population = []
            for i in range(0, self.pop_size, 2):
                parent1, parent2 = selected[i], selected[i+1]
                child1, child2 = self.crossover(parent1, parent2)
                child1 = self.mutate(child1)
                child2 = self.mutate(child2)
                new_population.extend([child1, child2])
            self.population = new_population
        return self.best_solution, self.best_fitness, self.fitness_history

def evaluate_solution(devices, clusters, solution):
    # Calculate total distance and other metrics
    total_distance = 0
    satisfied_devices = 0
    delay_satisfied = 0
    for i, cluster_id in enumerate(solution):
        device = devices[i]
        cluster = clusters[cluster_id]
        # Calculate distance
        dx = device["x"] - cluster["center"][0]
        dy = device["y"] - cluster["center"][1]
        distance = np.sqrt(dx**2 + dy**2)
        total_distance += distance
        # Check resource satisfaction
        if (cluster["total_cpu"] >= device["cpu_req"] and 
            cluster["total_mem"] >= device["mem_req"] and 
            cluster["total_bw"] >= device["bw_req"]):
            satisfied_devices += 1
        # Check delay satisfaction
        if distance <= device["max_delay"]:
            delay_satisfied += 1
    return {
        "total_distance": total_distance,
        "avg_distance": total_distance / len(devices),
        "resource_satisfaction": satisfied_devices / len(devices),
        "delay_satisfaction": delay_satisfied / len(devices)
    }

# Main execution
if __name__ == "__main__":
    # Generate data
    fog_nodes, iot_devices = generate_fake_data()
    print(f"Generated {len(fog_nodes)} fog nodes and {len(iot_devices)} IoT devices")
    # Cluster fog nodes
    clusters = cluster_fog_nodes(fog_nodes, n_clusters=5)
    print(f"Created {len(clusters)} fog clusters")
    # Run genetic algorithm
    ga = GeneticAlgorithm(iot_devices, clusters, pop_size=50, generations=100)
    best_solution, best_fitness, fitness_history = ga.run()
    # Evaluate solution
    results = evaluate_solution(iot_devices, clusters, best_solution)
    print("\nOptimization Results:")
    print(f"Best Fitness (Total Distance + Penalty): {best_fitness:.2f}")
    print(f"Average Distance: {results['avg_distance']:.2f}")
    print(f"Resource Satisfaction: {results['resource_satisfaction']*100:.1f}%")
    print(f"Delay Satisfaction: {results['delay_satisfaction']*100:.1f}%")
    # Plot fitness history
    plt.figure(figsize=(10, 6))
    plt.plot(fitness_history)
    plt.title("Genetic Algorithm Fitness History")
    plt.xlabel("Generation")
    plt.ylabel("Fitness (Lower is Better)")
    plt.grid(True)
    plt.show()
    # Visualize clusters and assignments
    plt.figure(figsize=(12, 10))
    # Plot fog nodes
    for layer, color in zip(["L1", "L2", "L3", "L4"], ["blue", "green", "orange", "red"]):
        layer_nodes = [node for node in fog_nodes if node["layer"] == layer]
        x = [node["x"] for node in layer_nodes]
        y = [node["y"] for node in layer_nodes]
        plt.scatter(x, y, c=color, alpha=0.3, label=f"Fog {layer}")
    # Plot cluster centers
    for cluster in clusters:
        plt.scatter(cluster["center"][0], cluster["center"][1], c='black', marker='X', s=100)
    # Plot devices and their assignments
    for i, device in enumerate(iot_devices):
        cluster_id = best_solution[i]
        cluster = clusters[cluster_id]
        # Draw line from device to cluster center
        plt.plot([device["x"], cluster["center"][0]], 
                 [device["y"], cluster["center"][1]], 
                 "k--", alpha=0.2)
        # Plot device
        plt.scatter(device["x"], device["y"], c="purple", marker="o", s=50)
    plt.title("Fog Clusters and Device Assignments")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.legend()
    plt.grid(True)
    plt.show()