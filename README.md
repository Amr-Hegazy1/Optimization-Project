# Multi-Robot Path Planning Simulation

This project implements a multi-robot path planning optimization system using Simulated Annealing. It includes a professional GUI visualization tool for analyzing robot paths, coverage, and network connectivity.

## Features

- **Multi-Robot Path Planning**: Coordinate multiple robots to explore an environment
- **Simulated Annealing Optimization**: Find near-optimal paths using SA algorithm
- **Interactive GUI Visualization**: Professional, animated visualization of results
- **Comprehensive Metrics**: Coverage, connectivity, and performance statistics
- **Constraint Handling**: Energy budget, obstacle avoidance, collision detection

## Installation

### Prerequisites

```bash
pip install numpy matplotlib
```

### Optional Dependencies

For advanced features:
```bash
pip install scipy  # For additional optimization methods
```

## Running the Simulation

To run the complete optimization with visualization:

```bash
python main.py
```

### What Happens

1. **Initial Path Generation**: Creates a random, feasible initial path for all robots
2. **Simulated Annealing**: Optimizes the paths over multiple iterations
3. **Console Visualization**: Displays text-based coverage map and statistics
4. **GUI Launch**: Opens interactive visualization window with:
   - Animated robot movement
   - Real-time coverage and connectivity charts
   - Detailed performance metrics
   - Playback controls

### Visualization Controls

- **Play/Pause/Reset**: Control animation playback
- **Speed Slider**: Adjust animation speed (10-200 ms/frame)
- **Zoom/Pan**: Use matplotlib toolbar for detailed inspection
- **Statistics Panel**: View real-time metrics and robot positions

For detailed visualization documentation, see [VISUALIZATION_GUIDE.md](VISUALIZATION_GUIDE.md).

## Parameters

You can adjust the simulation behavior by modifying parameters in `main.py`:

### Environment Parameters
*   `N`, `M`: Map dimensions (default: 20x20)
*   `ROBOTS_POSITIONS`: Initial positions of the robots
*   `K`: Number of steps in the path (default: 110)

### Constraint Parameters
*   `ENERGY_BUDGET`: Maximum cells each robot can traverse (default: 100)
*   `COMMUNICATION_RADIUS`: Communication range R_c (default: 15.0)
*   `CONNECTIVITY_THRESHOLD`: Threshold for network connectivity (default: 15.0)

### Objective Function Weights
*   `ALPHA`: Coverage weight (default: 1.0)
*   `BETA`: Connectivity weight (default: 0.5)
*   `GAMMA`: Disconnection penalty weight (default: 2.0)
*   `ZETA`: Obstacle encounter penalty weight (default: 5.0)

### Simulated Annealing Parameters

In `simulated_annealing.py`:
*   `initial_temperature`: Starting temperature (default: 100.0)
*   `cooling_rate`: Geometric cooling schedule rate (default: 0.995)
*   `min_temperature`: Minimum temperature threshold (default: 0.5)
*   `max_iterations`: Maximum optimization iterations (default: 5000)

## Project Structure

```
Optimization Project/
│
├── main.py                      # Main execution script
├── simulated_annealing.py       # SA optimization algorithm
├── visualization.py             # GUI visualization class
├── README.md                    # This file
├── VISUALIZATION_GUIDE.md       # Detailed visualization documentation
│
├── GUC_Thesis.tex              # LaTeX thesis document
├── chapters/                    # Thesis chapters
├── Figures/                     # Thesis figures
├── References/                  # Bibliography
└── Sections/                    # Thesis sections
```

## Algorithm Overview

### Objective Function

The system maximizes:

```
f = α·Coverage + β·Connectivity - γ·P_disconnect - ζ·P_obstacle
```

Where:
- **Coverage**: Number of unique cells explored by all robots
- **Connectivity**: Sum of distance-weighted communication links
- **P_disconnect**: Penalty for robots disconnected from main network
- **P_obstacle**: Penalty for obstacle encounters

### Constraints

1. **Map Bounds**: Robots must stay within [0, N) × [0, M)
2. **Motion**: Only 4-connected moves (Manhattan distance ≤ 1 per step)
3. **Energy Budget**: Total distance traveled ≤ ℓ for each robot
4. **Obstacle Avoidance**: Cannot occupy obstacle cells
5. **Collision Avoidance**: No two robots at same position simultaneously

### Simulated Annealing Process

1. Start with random feasible solution
2. Generate neighbor by randomly modifying movements
3. Accept better solutions always
4. Accept worse solutions with probability exp(Δf/T)
5. Gradually reduce temperature T
6. Repeat until convergence or max iterations

## Output

### Console Output
- Optimization progress (every 10 iterations)
- Final cost breakdown
- Text-based coverage map with robot positions

### GUI Output
- Interactive animation of robot paths
- Real-time coverage and connectivity graphs
- Detailed performance metrics
- Exportable final visualization image

## Examples

### Example 1: Basic Run
```bash
python main.py
```

### Example 2: Custom Parameters
```python
# In main.py, modify:
N, M = 30, 30  # Larger map
K = 150        # Longer paths
ALPHA = 2.0    # Prioritize coverage
```

### Example 3: Save Visualization Only
```python
from visualization import RobotPathVisualizer

viz = RobotPathVisualizer(...)
viz._update_visualization(viz.K - 1)  # Jump to final state
viz.save_final_state('results.png')
```

## Troubleshooting

### No Feasible Initial Solution
- Reduce `K` (path length)
- Increase `ENERGY_BUDGET`
- Reduce number of obstacles in map

### Poor Optimization Results
- Increase `max_iterations` in SimulatedAnnealing
- Adjust `initial_temperature` or `cooling_rate`
- Tune objective function weights (α, β, γ, ζ)

### Visualization Issues
- Ensure tkinter is installed: `python -c "import tkinter"`
- Update matplotlib: `pip install --upgrade matplotlib`
- See [VISUALIZATION_GUIDE.md](VISUALIZATION_GUIDE.md) for details

## Future Work

- [ ] Implement additional optimization algorithms (Genetic Algorithm, PSO)
- [ ] Add dynamic obstacle support
- [ ] Multi-objective optimization (Pareto frontier)
- [ ] Real-time replanning capabilities
- [ ] 3D environment support
- [ ] ROS integration for real robots

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## License

This project is part of a thesis at GUC (German University in Cairo).

## References

See `References/myref.bib` for academic references and related work.

---

**Author**: Amr Hegazy  
**Institution**: German University in Cairo (GUC)  
**Project**: Multi-Robot Path Planning Optimization
