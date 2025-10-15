# Multi-Robot Path Planning Simulation

This script simulates a multi-robot path planning problem. It defines a cost function to evaluate the quality of paths for multiple robots, considering factors like area coverage, inter-robot connectivity, and obstacle avoidance.

## Running the script

To run the simulation, simply execute the `main.py` file:

```bash
python main.py
```

The script will then:
1.  Generate a dummy solution (a set of paths for the robots).
2.  Check if the solution is feasible based on a set of constraints (map bounds, motion, energy, etc.).
3.  Calculate the cost of the solution using a detailed objective function.
4.  Print a breakdown of the cost components (coverage, connectivity, penalties).
5.  Display a visualization of the coverage map in the console.

## Parameters

The script's behavior can be adjusted by modifying the global variables at the top of `main.py`, such as:
*   `N`, `M`: Map dimensions.
*   `ROBOTS_POSITIONS`: Initial positions of the robots.
*   `ENERGY_BUDGET`: Maximum distance a robot can travel.
*   `COMMUNICATION_RADIUS`: The range within which robots can communicate.
*   `ALPHA`, `BETA`, `GAMMA`, `ZETA`: Weights for the different components of the objective function.
