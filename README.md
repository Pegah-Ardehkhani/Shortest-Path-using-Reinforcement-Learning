# Shortest Path Finder using Reinforcement Learning 🏍️

<p align="center"> 
  <img width="600" height="400" src="https://github.com/Pegah-Ardehkhani/Shortest-Path-using-Reinforcement-Learning/blob/main/Learned_Path_Pic.png"> 
</p>

This project provides a framework to find the shortest path between nodes in a graph using reinforcement learning algorithms. The main algorithms implemented are **Q-learning**, **SARSA**, and **SARSA(λ)**. These algorithms are used to find optimal paths by learning from the graph's structure, where edges have weights representing distances or costs.

## Features

- **Q-learning**: A model-free reinforcement learning algorithm to learn an optimal policy for pathfinding.
- **SARSA**: A model-free on-policy reinforcement learning algorithm, similar to Q-learning but differs in how it updates the Q-values.
- **SARSA(λ)**: A variant of SARSA with eligibility traces, providing faster convergence for certain problems.
- **Graph Visualization**: The library includes built-in support to visualize graphs before and after running reinforcement learning algorithms, showing the learned path from the start node to the destination.

## Installation

1. Clone the repository or download the source files.
2. Ensure you have the following Python libraries installed:
   - `networkx` - for creating and manipulating graphs.
   - `matplotlib` - for visualizing graphs.
   - `numpy` - for numerical computations.
   - `random` - for random number generation.
   
   You can install these dependencies via `pip`:

   ```bash
   pip install networkx matplotlib numpy
   ```

3. Place the Python script (e.g., `pathfinder.py`) in your project folder.

## Usage

To use the **PathFinder** framework to find the shortest path in a graph, you need to:

1. **Define your graph**: The graph is defined using a list of edges, where each edge is a tuple `(u, v, weight)`, representing an undirected edge from node `u` to node `v` with the specified weight (or distance).

2. **Run the pathfinding algorithm**: Call the `shortestpathfinder()` function with the desired parameters.

### Example

```python
if __name__ == '__main__':
    # Define edges with weights (distance between nodes)
    edges = [
        (0, 1, 1), (0, 3, 1), (4, 2, 2), (4, 1, 2), 
        (4, 8, 1), (4, 9, 2), (2, 3, 1), (2, 6, 2), 
        (2, 5, 1), (5, 6, 2), (7, 8, 1), (7, 5, 2), 
        (2, 10, 3), (10, 6, 1), (8, 9, 3), (8, 11, 1), 
        (9, 11, 1)  # Only one direction for the edges
    ]
    
    # Run the pathfinder for the defined edges using Q-learning
    shortestpathfinder(
        edges, 
        start_node=0, 
        aim_node=11, 
        RL_algorithm='q_learning',  # Choose from 'q_learning', 'sarsa', 'sarsa_lambda'
        epsilon=0.02,  # Exploration rate
        plot=True,  # Enable graph visualization
        num_epoch=300  # Number of training episodes
    )
```

### Arguments

The main function `shortestpathfinder()` accepts the following arguments:

- **edges** (list): List of edges in the form of tuples `(u, v, weight)` representing the graph.
- **start_node** (int, optional): The node from which to start the pathfinding. Default is 0.
- **aim_node** (int, optional): The destination node for pathfinding. Default is 10.
- **RL_algorithm** (str, optional): The reinforcement learning algorithm to use (`'q_learning'`, `'sarsa'`, or `'sarsa_lambda'`). Default is `'q_learning'`.
- **num_epoch** (int, optional): Number of episodes for training the algorithm. Default is 500.
- **gamma** (float, optional): Discount factor for future rewards. Default is 0.8.
- **epsilon** (float, optional): Exploration rate for the epsilon-greedy strategy. Default is 0.05.
- **alpha** (float, optional): Learning rate. Default is 0.1.
- **lambda_** (float, optional): Eligibility trace decay factor for SARSA(λ) (0 ≤ λ ≤ 1). Default is 0.9.
- **plot** (bool, optional): Whether to visualize the graph and the learned path. Default is `True`.

### Output

1. **Graph Visualization**: The script will display the initial graph and then show the graph again with the learned path highlighted, depending on the chosen algorithm.
2. **Learned Path**: The script will print the nodes in the learned path.
3. **Path Length**: The total length (sum of edge weights) of the learned path will be printed.

### Example Output

```plaintext
Episode 1/300
Starting state: 0
Current state: 0 -> Next state: 1
  Edge weight (distance): 1  Reward: -1
  Updated Q-table: 0.0900
...
Episode 300 completed. Path length: 10.5
--------------------------------------------------
Time taken to complete q_learning: 15.24 seconds
Learned path from node 0 to node 11: [0, 1, 3, 5, 6, 10, 11]
Path length: 10.5
```

### Visualization

After the algorithm finishes training, the program will visualize the graph using `matplotlib`. Nodes will be color-coded:
- **Start node**: Red.
- **Goal node**: Yellow.
- **Other nodes**: Blue.
- **Path edges**: Green (found by the algorithm), gray (if not part of the path).

## Algorithm Details

- **Q-learning**: The agent learns a policy by interacting with the environment (graph), updating Q-values to choose the most rewarding path.
- **SARSA**: Similar to Q-learning, but updates are based on the action actually taken (on-policy).
- **SARSA(λ)**: An extension of SARSA with eligibility traces, allowing faster learning.

## Conclusion

The **PathFinder** framework offers a simple yet powerful way to apply reinforcement learning to pathfinding problems in graphs. By visualizing both the initial graph and the learned path, users can gain insights into the algorithm's performance and decision-making process.
