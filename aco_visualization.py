"""
Professional GUI Visualization for Ant Colony Optimization
Provides interactive visualization of robot paths, coverage, and connectivity
Shows ACO-specific pheromone evolution and best solution
"""

import numpy as np
import matplotlib.pyplot as plt
from optimization.base_visualizer import BaseVisualizer


class AntColonyVisualizer(BaseVisualizer):
    """
    Visualizes the Ant Colony Optimization process in real-time.
    Shows objective function evolution, pheromone levels, and best solution found.
    
    Extends BaseVisualizer with ACO-specific pheromone plot.
    """
    
    def __init__(self, initial_positions, map_grid, 
                 communication_radius, connectivity_threshold,
                 alpha, beta, gamma, visualization_step_size=1):
        """
        Initialize the ACO optimization visualizer.
        
        Args:
            initial_positions: list of initial (x, y) positions
            map_grid: 2D numpy array representing the map
            communication_radius: R_c for connectivity
            connectivity_threshold: threshold for network connectivity
            alpha, beta, gamma: objective function weights
            visualization_step_size: number of steps to skip in animation (default: 1)
        """
        # ACO-specific tracking
        self.avg_pheromone_levels = []
        self.max_pheromone_levels = []
        self.display_avg_pheromone = None
        self.display_max_pheromone = None
        
        # Call parent constructor (this will call _create_gui which calls _create_visualization_panel)
        super().__init__(initial_positions, map_grid, communication_radius, 
                        connectivity_threshold, alpha, beta, gamma, 
                        visualization_step_size)
    
    def get_algorithm_name(self):
        """Return the algorithm name for display."""
        return "Ant Colony Optimization"
    
    def _create_visualization_panel(self, parent):
        """Create the matplotlib visualization panel."""
        # Create figure with subplots
        self.fig = plt.Figure(figsize=(19.0, 10.6), facecolor='white')
        
        # Grid specification: 2 rows, 3 columns with custom spacing
        gs = self.fig.add_gridspec(
            2,
            3,
            height_ratios=[1.0, 1.0],
            width_ratios=[0.75, 1.85, 1.0],
            hspace=0.28,
            wspace=0.30,
            left=0.045,
            right=0.985,
            top=0.965,
            bottom=0.05,
        )

        # Left side spanning both rows: Cost function evolution (minimization)
        self.ax_obj = self.fig.add_subplot(gs[:, 0])
        self.ax_obj.set_title('Best Cost Evolution (lower=better)', fontsize=13, fontweight='bold', pad=10)
        self.ax_obj.set_xlabel('Iteration', fontsize=11)
        self.ax_obj.set_ylabel('Cost Value', fontsize=11)
        self.ax_obj.grid(True, alpha=0.3, linestyle='--')
        
        # Use base class helper to create common map axes
        self.ax_current, self.ax_best, self.ax_coverage, self.ax_stats = \
            self._create_common_map_axes(gs, self.fig)
        
        # Embed in tkinter
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        self.canvas = FigureCanvasTkAgg(self.fig, parent)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill='both', expand=True)
        
        # Initialize plots
        self._initialize_aco_plots()
        self._initialize_common_plots()
        self._initialize_robot_markers()
    
    def _initialize_aco_plots(self):
        """Initialize ACO-specific plots (objective function)."""
        # Objective function plot - only best solution
        self.best_line, = self.ax_obj.plot([], [], 'g-', linewidth=2.5, label='Best Solution')
        self.ax_obj.legend(loc='upper right', fontsize=10)
    
    def update_optimization(self, iteration, current_cost, best_cost, best_path, 
                          avg_pheromone=None, max_pheromone=None, current_path=None):
        """
        Update visualization with new optimization data.
        
        Implementation of abstract method from BaseVisualizer.
        Handles ACO-specific pheromone parameters.
        
        Args:
            iteration: current iteration number
            current_cost: cost of current solution
            best_cost: cost of best solution found
            best_path: best path array found
            avg_pheromone: average pheromone level (optional)
            max_pheromone: maximum pheromone level (optional)
            current_path: current path array (optional)
        """
        # Store ACO-specific data
        if avg_pheromone is not None:
            self.avg_pheromone_levels.append(avg_pheromone)
        if max_pheromone is not None:
            self.max_pheromone_levels.append(max_pheromone)
        
        # Update common data through parent method
        self._update_common_data(iteration, current_cost, best_cost, best_path, current_path)
        
        # Store display state including pheromone levels
        self.pending_display_state = {
            "iteration": iteration,
            "current_cost": current_cost,
            "best_cost": best_cost,
            "avg_pheromone": avg_pheromone,
            "max_pheromone": max_pheromone,
        }
        
        # Update all plots
        self._update_plots()
    
    def _update_plots(self):
        """Update all plots with current data."""
        if len(self.iterations) == 0:
            return
        
        # Update objective function plot - only best solution
        self.best_line.set_data(self.iterations, self.best_costs)
        self.ax_obj.relim()
        self.ax_obj.autoscale_view()
        
        # Update statistics
        self._update_statistics()
        
        # Update status
        self._update_status_bar()
        
        self.canvas.draw_idle()
        self.root.update()
    
    def _update_statistics(self):
        """Update statistics panel with ACO-specific information."""
        self.ax_stats.clear()
        self.ax_stats.axis('off')
        
        if self.display_iteration is None:
            return
        
        # Create compact statistics text
        stats_text = ["ANT COLONY OPTIMIZATION"]
        stats_text.append(f"Iteration:  {self.display_iteration:,}")
        
        stats_text.append("")
        stats_text.append("COST VALUES (lower=better)")
        if self.display_best_cost is not None:
            stats_text.append(f"Best:      {self.display_best_cost:8.6f}")
        else:
            stats_text.append("Best:           --")
        
        if len(self.best_costs) > 1 and self.display_best_cost is not None:
            initial_cost = self.best_costs[0]
            current_best = self.display_best_cost
            improvement = initial_cost - current_best
            improvement_pct = (improvement / abs(initial_cost) * 100) if initial_cost != 0 else 0
            stats_text.append("")
            stats_text.append("IMPROVEMENT")
            stats_text.append(f"Δ Value:   {improvement:+8.6f}")
            stats_text.append(f"Δ Percent: {improvement_pct:+7.1f}%")
        
        if self.best_path is not None:
            visited = set()
            for r in range(self.R):
                for pos in self.best_path[r]:
                    pos_tuple = tuple(pos) if isinstance(pos, np.ndarray) else pos
                    if self.map_grid[pos_tuple[0], pos_tuple[1]] == 0:
                        visited.add(pos_tuple)
            coverage_pct = (len(visited) / (self.N * self.M)) * 100
            stats_text.append("")
            stats_text.append("BEST PATH COVERAGE")
            stats_text.append(f"Cells:    {len(visited):6d}")
            stats_text.append(f"Coverage: {coverage_pct:6.1f}%")
        
        text_str = '\n'.join(stats_text)
        self.ax_stats.text(
            0.02,
            0.98,
            text_str,
            transform=self.ax_stats.transAxes,
            fontsize=8,
            verticalalignment='top',
            fontfamily='monospace',
            bbox=dict(
                boxstyle='round,pad=0.6',
                facecolor='#f8f9fa',
                edgecolor='#dee2e6',
                alpha=0.95,
                linewidth=1.5,
            ),
        )
    
    def _update_status_bar(self, final=False, final_coverage=None):
        """Update status bar with ACO-specific information."""
        if len(self.iterations) == 0:
            status_text = "Status: Waiting for optimization to start..."
        else:
            prefix = "Status: ✓ Optimization Complete!" if final else "Status: Optimizing..."
            iter_value = self.display_iteration if self.display_iteration is not None else self.iterations[-1]
            cost_value = self.display_best_cost if self.display_best_cost is not None else self.best_costs[-1]
            status_text = (
                f"{prefix} | Iter: {iter_value} | "
                f"Best Cost: {cost_value:.6f}"
            )
            if final and final_coverage is not None:
                status_text += f" | Final Coverage: {final_coverage:.1f}%"
                status_text += f" | Total Iterations: {self.iterations[-1]}"
        self.status_label.config(text=status_text)
