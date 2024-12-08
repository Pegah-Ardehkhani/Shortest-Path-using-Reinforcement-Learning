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

- **Q-learning**
Q-learning is an off-policy reinforcement learning algorithm where the agent learns the optimal action-value function without requiring the environment to follow the same policy. The agent updates the Q-values by using the Bellman equation:

$Q(s, a) \leftarrow Q(s, a) + \alpha \left[ R + \gamma \max_{a'} Q(s', a') - Q(s, a) \right]$

Where:
- $( Q(s, a) )$ is the Q-value for state $( s )$ and action $( a )$,
- $( \alpha )$ is the learning rate,
- $( R )$ is the reward received after taking action $( a )$ in state $( s )$,
- $( \gamma )$ is the discount factor for future rewards,
- $( \max_{a'} Q(s', a') )$ is the maximum Q-value for the next state $( s' )$.

The agent explores the environment using an **epsilon-greedy** strategy, balancing exploration (random actions) with exploitation (choosing the action with the highest Q-value).

- **SARSA**
SARSA (State-Action-Reward-State-Action) is an on-policy algorithm, meaning it updates Q-values based on the action the agent actually takes, rather than the optimal action (like Q-learning). The SARSA update rule is:

$Q(s, a) \leftarrow Q(s, a) + \alpha \left[ R + \gamma Q(s', a') - Q(s, a) \right]$

Where:
- $( Q(s, a) )$ is the Q-value for state $( s )$ and action $( a )$,
- $( R )$ is the reward received,
- $( \gamma )$ is the discount factor,
- $( Q(s', a') )$ is the Q-value for the next state $( s' )$ and action $( a' )$.

SARSA uses an **epsilon-greedy** policy to choose actions, exploring with a probability of $( \epsilon )$ and exploiting the best action with probability $( 1 - \epsilon )$.

- **SARSA(λ)**
SARSA(λ) is an extension of SARSA that incorporates **eligibility traces** to allow faster learning by updating multiple state-action pairs during a single transition. The eligibility trace acts as a memory, allowing previous state-action pairs to contribute to the update. The update rule is:

$Q(s, a) \leftarrow Q(s, a) + \alpha \left[ R + \gamma Q(s', a') - Q(s, a) \right] \cdot E(s, a)$

$E(s, a) \leftarrow \gamma \lambda E(s, a)$

Where:
- $( E(s, a) )$ is the eligibility trace for state $( s )$ and action $( a )$,
- $( \lambda )$ is the trace decay factor $(0 ≤ \( \lambda ) ≤ 1)$, controlling how much influence previous state-action pairs have on updates.

SARSA(λ) enhances learning by updating not just the current state-action pair but also previously visited state-action pairs, speeding up the convergence process.

## Conclusion

The **PathFinder** framework offers a simple yet powerful way to apply reinforcement learning to pathfinding problems in graphs. By visualizing both the initial graph and the learned path, users can gain insights into the algorithm's performance and decision-making process.
