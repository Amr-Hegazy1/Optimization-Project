# Multi-Robot Path Planning Optimization

Multi-robot path planning system using metaheuristic optimization. The system coordinates multiple robots to explore an environment while maintaining network connectivity and avoiding obstacles.

## Features

- **Simulated Annealing Optimization**: Iterative improvement using probabilistic acceptance
- **Real-time Visualization**: Animated GUI showing robot paths, coverage, and connectivity
- **Extensible Architecture**: Easy to add new optimization algorithms (GA, PSO, etc.)
- **Constraint Handling**: Energy budget, obstacle avoidance, network connectivity
- **Performance Metrics**: Coverage analysis, connectivity tracking, objective function evolution

## Quick Start

```bash
# Install dependencies
pip install numpy matplotlib

# Run optimization
python main.py
```

The GUI will show real-time optimization progress with animated robot paths.

## Configuration

All parameters are centralized in `config.py` for easy customization:

**Environment:**
- `MAP_WIDTH, MAP_HEIGHT`: Map dimensions (default: 20×20)
- `ROBOT_INITIAL_POSITIONS`: Starting positions for each robot
- `PATH_LENGTH`: Number of steps in generated paths (default: 110)

**Constraints:**
- `ENERGY_BUDGET`: Maximum distance per robot (default: 100)
- `COMMUNICATION_RADIUS`: Communication range (default: 15.0)
- `CONNECTIVITY_THRESHOLD`: Network connectivity threshold (default: 15.0)

**Objective Weights:**
- `ALPHA`: Coverage weight (default: 1.0)
- `BETA`: Connectivity weight (default: 0.5)
- `GAMMA`: Disconnection penalty (default: 2.0)
- `ZETA`: Obstacle penalty (default: 5.0)

**Simulated Annealing:**
- `SA_INITIAL_TEMPERATURE`: Starting temperature (default: 100.0)
- `SA_COOLING_RATE`: Temperature reduction rate (default: 0.995)
- `SA_MIN_TEMPERATURE`: Stopping threshold (default: 0.5)
- `SA_MAX_ITERATIONS`: Maximum iterations (default: 5000)

**Visualization:**
- `VISUALIZATION_STEP_SIZE`: Animation frame skipping (default: 1)
- `ANIMATION_SPEED_MS`: Milliseconds per frame (default: 50)

## Project Structure

```
├── main.py                       # Main entry point
├── config.py                     # Configuration parameters
├── simulated_annealing.py        # SA algorithm implementation
├── visualization.py              # GUI visualization
├── optimization/                 # Optimization framework
│   ├── base_optimizer.py         # Base class for algorithms
│   └── base_visualizer.py        # Base class for visualizers
└── latex/                        # Thesis LaTeX files
    ├── GUC_Thesis.tex
    ├── chapters/
    ├── Figures/
    ├── References/
    └── Sections/
```

## Architecture

The codebase uses an object-oriented design with base classes:

- **BaseOptimizer**: Abstract class for optimization algorithms
  - Defines interface: `run()`, `generate_neighbor()`, `acceptance_criterion()`
  - Makes it easy to add new algorithms (GA, PSO, ACO, etc.)

- **BaseVisualizer**: Abstract class for visualization
  - Provides common components (maps, coverage plots, animations)
  - Algorithms add specific plots (e.g., temperature for SA)

This design allows adding new optimization techniques with minimal code.

## How It Works

The system maximizes an objective function balancing coverage and connectivity:

```
f = α·Coverage + β·Connectivity - γ·Disconnection - ζ·Obstacles
```

**Simulated Annealing** iteratively improves solutions by:
1. Generating neighbor solutions (random movement modifications)
2. Always accepting improvements
3. Accepting worse solutions probabilistically (allows escaping local optima)
4. Gradually reducing temperature to focus search

**Constraints enforced:**
- Energy budget per robot
- 4-connected movement (Manhattan distance ≤ 1)
- Obstacle and collision avoidance
- Map boundaries

## Adding New Algorithms

To implement a new optimization algorithm (e.g., Genetic Algorithm):

1. Create a class inheriting from `BaseOptimizer`
2. Implement required methods: `run()`, `generate_neighbor()`, `acceptance_criterion()`
3. Create a visualizer inheriting from `BaseVisualizer`
4. Add algorithm-specific plots

Example structure:
```python
from optimization.base_optimizer import BaseOptimizer

class GeneticAlgorithm(BaseOptimizer):
    def run(self, initial_solution):
        # Implement GA logic
        pass
```

The base classes handle visualization integration, state tracking, and common functionality.

---

**Author**: Amr Hegazy  
**Institution**: German University in Cairo (GUC)
