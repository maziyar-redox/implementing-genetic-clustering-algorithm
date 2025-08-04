# Genetic-Based Clustering for IoT Resource Allocation in Fog Computing

## Description
This project implements a genetic-based clustering algorithm for efficient resource allocation of IoT applications in layered fog heterogeneous platforms, as described in the research paper "A genetic-based clustering algorithm for efficient resource allocating of IoT applications in layered fog heterogeneous platforms" (2023).

The approach combines k-means clustering with a genetic algorithm to minimize latency and optimize resource allocation in fog computing environments. The system groups fog nodes into clusters and then assigns IoT devices to these clusters based on resource requirements and proximity, ensuring quality of service (QoS) constraints are met.

## Features
- **4-Layer Fog Architecture**: Simulates fog nodes in four layers (L1-L4) with increasing resource capacities
- **K-means Clustering**: Groups fog nodes based on physical location and resource capabilities
- **Genetic Algorithm Optimization**: Assigns IoT devices to clusters to minimize distance and satisfy resource requirements
- **Performance Evaluation**: Measures resource satisfaction, delay satisfaction, and average distance
- **Visualization**: Provides graphical representations of clusters, device assignments, and optimization progress

## Requirements
- Python 3.7+
- NumPy
- Matplotlib
- Scikit-learn
- TQDM

## Installation
1. Clone the repository:
```bash
git clone https://github.com/maziyar-redox/implementing-genetic-clustering-algorithm
cd genetic-clustering-iot
```

2. Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage
Run the main script to execute the genetic-based clustering algorithm:

```bash
python genetic_clustering_iot.py
```

The script will:
1. Generate fake fog nodes and IoT devices
2. Cluster fog nodes using k-means
3. Optimize device assignments using the genetic algorithm
4. Display performance metrics
5. Generate visualization plots

### Customization
You can modify the following parameters in the script:
- Number of fog nodes per layer
- Number of IoT devices
- Number of clusters
- Genetic algorithm parameters (population size, generations, mutation rate)

## Project Structure
```
genetic_clustering_iot/
├── genetic_clustering_iot.py  # Main implementation script
├── requirements.txt           # Required Python packages
└── README.md                  # Project documentation
```

## Results
The implementation produces the following outputs:

1. **Console Output**:
   - Optimization results including total distance, resource satisfaction, and delay satisfaction metrics

2. **Fitness History Plot**:
   - Shows how the solution improves over generations

3. **Cluster Visualization**:
   - Displays fog nodes in different layers, cluster centers, and device assignments

### Example Results
```
Generated 375 fog nodes and 20 IoT devices
Created 5 fog clusters

Optimization Results:
Best Fitness (Total Distance + Penalty): 1234.56
Average Distance: 45.67
Resource Satisfaction: 95.0%
Delay Satisfaction: 90.0%
```

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Citation
If you use this implementation in your research, please cite the original paper:
```
Abedpour, K., Hosseini Shirvani, M., & Abedpour, E. (2023). A genetic-based clustering algorithm for efficient resource allocating of IoT applications in layered fog heterogeneous platforms. Cluster Computing.
```