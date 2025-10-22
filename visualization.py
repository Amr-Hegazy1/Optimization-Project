"""
Professional GUI Visualization for Multi-Robot Path Planning
Provides interactive visualization of robot paths, coverage, and connectivity
Shows both optimization process and final solution
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, FancyBboxPatch
from matplotlib.animation import FuncAnimation
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import tkinter as tk
from tkinter import ttk, messagebox
import math
import threading
import time
from optimization.base_visualizer import BaseVisualizer


class OptimizationVisualizer(BaseVisualizer):
    """
    Visualizes the Simulated Annealing optimization process in real-time.
    Shows objective function evolution, temperature cooling, and best solution found.
    
    Extends BaseVisualizer with SA-specific temperature plot.
    """
    
    def __init__(self, initial_positions, map_grid, 
                 communication_radius, connectivity_threshold,
                 alpha, beta, gamma, visualization_step_size=1):
        """
        Initialize the SA optimization visualizer.
        
        Args:
            initial_positions: list of initial (x, y) positions
            map_grid: 2D numpy array representing the map
            communication_radius: R_c for connectivity
            connectivity_threshold: threshold for network connectivity
            alpha, beta, gamma: objective function weights
            visualization_step_size: number of steps to skip in animation (default: 1)
        """
        # SA-specific tracking
        self.temperatures = []
        self.initial_temperature = None  # Will be set on first update
        self.display_temperature = None
        
        # Call parent constructor (this will call _create_gui which calls _create_visualization_panel)
        super().__init__(initial_positions, map_grid, communication_radius, 
                        connectivity_threshold, alpha, beta, gamma, 
                        visualization_step_size)
        
        # Set animation speed from config if available
        try:
            import config
            self.animation_speed = config.ANIMATION_SPEED_MS
        except (ImportError, AttributeError):
            pass  # Use default from base class
    
    def get_algorithm_name(self):
        """Return the algorithm name for display."""
        return "Simulated Annealing"
    
    def _create_visualization_panel(self, parent):
        """Create the matplotlib visualization panel with SA-specific temperature plot."""
        # Create figure with subplots - redesigned for clarity
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
        
        # Top left: Temperature over iterations (SA-specific)
        self.ax_temp = self.fig.add_subplot(gs[0, 0])
        self.ax_temp.set_title('Temperature Cooling Schedule', fontsize=13, fontweight='bold', pad=10)
        self.ax_temp.set_xlabel('Iteration', fontsize=11)
        self.ax_temp.set_ylabel('Temperature', fontsize=11)
        self.ax_temp.grid(True, alpha=0.3, linestyle='--')
        
        # Bottom left: Cost function evolution (minimization)
        self.ax_obj = self.fig.add_subplot(gs[1, 0])
        self.ax_obj.set_title('Cost Function Evolution (lower=better)', fontsize=13, fontweight='bold', pad=10)
        self.ax_obj.set_xlabel('Iteration', fontsize=11)
        self.ax_obj.set_ylabel('Cost Value', fontsize=11)
        self.ax_obj.grid(True, alpha=0.3, linestyle='--')
        
        # Use base class helper to create common map axes
        self.ax_current, self.ax_best, self.ax_coverage, self.ax_stats = \
            self._create_common_map_axes(gs, self.fig)
        
        # Embed in tkinter
        self.canvas = FigureCanvasTkAgg(self.fig, parent)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Initialize plots
        self._initialize_sa_plots()
        self._initialize_common_plots()
        self._initialize_robot_markers()
    
    def _initialize_sa_plots(self):
        """Initialize SA-specific plots (temperature and objective function)."""
        # Temperature plot
        self.temp_line, = self.ax_temp.plot([], [], 'r-', linewidth=2.5, label='Temperature')
        self.ax_temp.legend(loc='upper right', fontsize=10)
        
        # Objective function plot
        self.current_line, = self.ax_obj.plot([], [], 'b-', linewidth=1.5, alpha=0.7, label='Current Solution')
        self.best_line, = self.ax_obj.plot([], [], 'g-', linewidth=2.5, label='Best Solution')
        self.ax_obj.legend(loc='lower right', fontsize=10)
    
    def update_optimization(self, iteration, temperature, current_cost, best_cost, best_path, current_path=None):
        """
        Update visualization with new optimization data.
        
        Implementation of abstract method from BaseVisualizer.
        Handles SA-specific temperature parameter.
        """
        # Store initial temperature on first update
        if self.initial_temperature is None:
            self.initial_temperature = temperature
        
        # Store SA-specific data
        self.temperatures.append(temperature)
        
        # Update common data through parent method
        self._update_common_data(iteration, current_cost, best_cost, best_path, current_path)
        
        # Store display state including temperature
        self.pending_display_state = {
            "iteration": iteration,
            "temperature": temperature,
            "current_cost": current_cost,
            "best_cost": best_cost,
        }
        
        # Update all plots
        self._update_plots()
    
    def _update_plots(self):
        """Update all plots with current data."""
        if len(self.iterations) == 0:
            return
            
        # Update temperature plot (SA-specific)
        self.temp_line.set_data(self.iterations, self.temperatures)
        self.ax_temp.relim()
        self.ax_temp.autoscale_view()
        
        # Update objective function plot
        self.current_line.set_data(self.iterations, self.current_costs)
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
        """Update statistics panel with SA-specific information."""
        self.ax_stats.clear()
        self.ax_stats.axis('off')
        
        if self.display_iteration is None:
            return
        
        # Create compact statistics text
        stats_text = ["SIMULATED ANNEALING"]
        stats_text.append(f"Iteration:  {self.display_iteration:,}")
        temp_line = f"Temperature: {self.display_temperature:.2f}" if self.display_temperature is not None else "Temperature: --"
        if self.initial_temperature is not None:
            temp_line += f"  (start {self.initial_temperature:.2f})"
        stats_text.append(temp_line)
        stats_text.append("")
        stats_text.append("COST VALUES (lower=better)")
        if self.display_current_cost is not None:
            stats_text.append(f"Current:   {self.display_current_cost:8.6f}")
        else:
            stats_text.append("Current:        --")
        if self.display_best_cost is not None:
            stats_text.append(f"Best:      {self.display_best_cost:8.6f}")
        else:
            stats_text.append("Best:           --")
        
        if len(self.best_costs) > 1 and self.display_best_cost is not None:
            initial_cost = self.best_costs[0]
            current_best = self.display_best_cost
            improvement = initial_cost - current_best  # For minimization: positive means improvement
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
        """Update status bar with SA-specific information."""
        if len(self.iterations) == 0:
            status_text = "Status: Waiting for optimization to start..."
        else:
            prefix = "Status: ✓ Optimization Complete!" if final else "Status: Optimizing..."
            status_text = (
                f"{prefix} | Iteration: {self.display_iteration or self.iterations[-1]} | "
                f"Temperature: {(self.display_temperature if self.display_temperature is not None else self.temperatures[-1]):.2f} | "
                f"Best Cost: {(self.display_best_cost if self.display_best_cost is not None else self.best_costs[-1]):.6f}"
            )
            if final and final_coverage is not None:
                status_text += f" | Final Coverage: {final_coverage:.1f}%"
                status_text += f" | Total Iterations: {self.iterations[-1]}"
        self.status_label.config(text=status_text)


class RobotPathVisualizer:
    """
    Professional GUI visualization for multi-robot path planning optimization.
    
    Features:
    - Animated robot movement along optimized paths
    - Real-time coverage and connectivity visualization
    - Performance metrics dashboard
    - Interactive controls (play, pause, speed control)
    - Multi-panel layout with statistics
    """
    
    def __init__(self, path_array, initial_positions, map_grid, 
                 communication_radius, connectivity_threshold,
                 alpha, beta, gamma):
        """
        Initialize the visualization.
        
        Args:
            path_array: numpy array of robot paths [R x K x 2]
            initial_positions: list of initial (x, y) positions
            map_grid: 2D numpy array representing the map
            communication_radius: R_c for connectivity
            connectivity_threshold: threshold for network connectivity
            alpha, beta, gamma: objective function weights
        """
        self.path_array = path_array
        self.initial_positions = initial_positions
        self.map_grid = map_grid
        self.communication_radius = communication_radius
        self.connectivity_threshold = connectivity_threshold
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        
        self.R = len(path_array)  # Number of robots
        self.K = len(path_array[0])  # Path length
        self.N, self.M = map_grid.shape  # Map dimensions
        
        # Animation state
        self.current_step = 0
        self.is_playing = False
        self.animation = None
        self.speed = 50  # milliseconds per frame
        
        # Color scheme for robots
        self.robot_colors = plt.cm.tab10(np.linspace(0, 1, self.R))
        
        # Statistics tracking
        self.coverage_history = []
        self.connectivity_history = []
        self.visited_cells = set()
        
        # Create GUI
        self._create_gui()
        
    def _create_gui(self):
        """Create the main GUI window with all components."""
        self.root = tk.Tk()
        self.root.title("Multi-Robot Path Planning Visualization")
        self.root.geometry("1600x900")
        self.root.configure(bg='#f0f0f0')
        
        # Create main container
        main_container = ttk.Frame(self.root)
        main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Left panel: Visualization
        left_panel = ttk.Frame(main_container)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Right panel: Controls and Statistics
        right_panel = ttk.Frame(main_container, width=350)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, padx=(10, 0))
        right_panel.pack_propagate(False)
        
        self._create_visualization_panel(left_panel)
        self._create_control_panel(right_panel)
        self._create_statistics_panel(right_panel)
        
    def _create_visualization_panel(self, parent):
        """Create the matplotlib visualization panel."""
        # Create figure with subplots
        self.fig = plt.Figure(figsize=(12, 10), facecolor='white')
        self.fig.suptitle('Multi-Robot Path Planning Visualization', 
                         fontsize=16, fontweight='bold', y=0.98)
        
        # Main map view
        self.ax_map = self.fig.add_subplot(2, 2, (1, 3))
        self.ax_map.set_title('Environment Map & Robot Paths', fontsize=12, fontweight='bold')
        self.ax_map.set_xlabel('Y Coordinate', fontsize=10)
        self.ax_map.set_ylabel('X Coordinate', fontsize=10)
        self.ax_map.set_aspect('equal')
        self.ax_map.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        
        # Coverage over time
        self.ax_coverage = self.fig.add_subplot(2, 2, 2)
        self.ax_coverage.set_title('Coverage Over Time', fontsize=11, fontweight='bold')
        self.ax_coverage.set_xlabel('Time Step', fontsize=9)
        self.ax_coverage.set_ylabel('Coverage (%)', fontsize=9)
        self.ax_coverage.grid(True, alpha=0.3)
        
        # Connectivity over time
        self.ax_connectivity = self.fig.add_subplot(2, 2, 4)
        self.ax_connectivity.set_title('Network Connectivity', fontsize=11, fontweight='bold')
        self.ax_connectivity.set_xlabel('Time Step', fontsize=9)
        self.ax_connectivity.set_ylabel('Connectivity Score', fontsize=9)
        self.ax_connectivity.grid(True, alpha=0.3)
        
        self.fig.tight_layout(rect=[0, 0, 1, 0.96])
        
        # Embed in tkinter
        self.canvas = FigureCanvasTkAgg(self.fig, parent)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Add toolbar
        toolbar_frame = ttk.Frame(parent)
        toolbar_frame.pack(fill=tk.X)
        toolbar = NavigationToolbar2Tk(self.canvas, toolbar_frame)
        toolbar.update()
        
        # Initialize visualization
        self._initialize_map()
        
    def _create_control_panel(self, parent):
        """Create control panel with playback controls."""
        control_frame = ttk.LabelFrame(parent, text="Playback Controls", padding=10)
        control_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Playback buttons
        button_frame = ttk.Frame(control_frame)
        button_frame.pack(fill=tk.X, pady=5)
        
        self.play_button = ttk.Button(button_frame, text="▶ Play", 
                                      command=self._play, width=12)
        self.play_button.pack(side=tk.LEFT, padx=2)
        
        self.pause_button = ttk.Button(button_frame, text="⏸ Pause", 
                                       command=self._pause, width=12, state=tk.DISABLED)
        self.pause_button.pack(side=tk.LEFT, padx=2)
        
        self.reset_button = ttk.Button(button_frame, text="⏮ Reset", 
                                       command=self._reset, width=12)
        self.reset_button.pack(side=tk.LEFT, padx=2)
        
        # Speed control
        speed_frame = ttk.Frame(control_frame)
        speed_frame.pack(fill=tk.X, pady=10)
        
        ttk.Label(speed_frame, text="Animation Speed:").pack(anchor=tk.W)
        
        # Create label first
        self.speed_label = ttk.Label(speed_frame, text=f"{self.speed} ms/frame")
        self.speed_label.pack(anchor=tk.W, pady=(0, 5))
        
        # Then create scale (which may trigger callback)
        self.speed_scale = ttk.Scale(speed_frame, from_=10, to=200, 
                                     orient=tk.HORIZONTAL, command=self._update_speed)
        self.speed_scale.set(self.speed)
        self.speed_scale.pack(fill=tk.X, pady=5)
        
        # Progress bar
        ttk.Label(control_frame, text="Progress:").pack(anchor=tk.W, pady=(10, 0))
        self.progress = ttk.Progressbar(control_frame, mode='determinate', 
                                       maximum=self.K-1)
        self.progress.pack(fill=tk.X, pady=5)
        
        self.step_label = ttk.Label(control_frame, text=f"Step: 0 / {self.K-1}")
        self.step_label.pack(anchor=tk.W)
        
    def _create_statistics_panel(self, parent):
        """Create statistics panel showing metrics."""
        stats_frame = ttk.LabelFrame(parent, text="Performance Metrics", padding=10)
        stats_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create scrollable frame
        canvas = tk.Canvas(stats_frame, bg='white')
        scrollbar = ttk.Scrollbar(stats_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Real-time metrics
        metrics_frame = ttk.Frame(scrollable_frame)
        metrics_frame.pack(fill=tk.X, pady=5)
        
        # Coverage metrics
        ttk.Label(metrics_frame, text="Coverage Metrics", 
                 font=('Arial', 10, 'bold')).pack(anchor=tk.W, pady=(5, 2))
        
        self.coverage_label = ttk.Label(metrics_frame, text="Coverage: 0%")
        self.coverage_label.pack(anchor=tk.W, padx=10)
        
        self.cells_explored_label = ttk.Label(metrics_frame, text="Cells Explored: 0")
        self.cells_explored_label.pack(anchor=tk.W, padx=10)
        
        ttk.Separator(metrics_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)
        
        # Connectivity metrics
        ttk.Label(metrics_frame, text="Connectivity Metrics", 
                 font=('Arial', 10, 'bold')).pack(anchor=tk.W, pady=(5, 2))
        
        self.connectivity_label = ttk.Label(metrics_frame, text="Connectivity: 0.00")
        self.connectivity_label.pack(anchor=tk.W, padx=10)
        
        self.network_status_label = ttk.Label(metrics_frame, text="Network: Connected")
        self.network_status_label.pack(anchor=tk.W, padx=10)
        
        ttk.Separator(metrics_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)
        
        # Robot information
        ttk.Label(metrics_frame, text="Robot Information", 
                 font=('Arial', 10, 'bold')).pack(anchor=tk.W, pady=(5, 2))
        
        self.robot_labels = []
        for r in range(self.R):
            robot_frame = ttk.Frame(metrics_frame)
            robot_frame.pack(fill=tk.X, padx=10, pady=2)
            
            # Color indicator
            color_hex = '#{:02x}{:02x}{:02x}'.format(
                int(self.robot_colors[r][0]*255),
                int(self.robot_colors[r][1]*255),
                int(self.robot_colors[r][2]*255)
            )
            color_label = tk.Label(robot_frame, text="  ", bg=color_hex, 
                                  relief=tk.RAISED, width=2)
            color_label.pack(side=tk.LEFT)
            
            info_label = ttk.Label(robot_frame, text=f"Robot {r+1}: (0, 0)")
            info_label.pack(side=tk.LEFT, padx=5)
            self.robot_labels.append(info_label)
        
        ttk.Separator(metrics_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)
        
        # Objective function breakdown
        ttk.Label(metrics_frame, text="Objective Function", 
                 font=('Arial', 10, 'bold')).pack(anchor=tk.W, pady=(5, 2))
        
        self.obj_total_label = ttk.Label(metrics_frame, text="Total: 0.00", 
                                         font=('Arial', 9, 'bold'))
        self.obj_total_label.pack(anchor=tk.W, padx=10)
        
        self.obj_coverage_label = ttk.Label(metrics_frame, 
                                            text=f"Coverage (α={self.alpha}): 0.00")
        self.obj_coverage_label.pack(anchor=tk.W, padx=15)
        
        self.obj_connectivity_label = ttk.Label(metrics_frame, 
                                                text=f"Connectivity (β={self.beta}): 0.00")
        self.obj_connectivity_label.pack(anchor=tk.W, padx=15)
        
        self.obj_disconnect_label = ttk.Label(metrics_frame, 
                                              text=f"Disconnection (γ={self.gamma}): 0.00")
        self.obj_disconnect_label.pack(anchor=tk.W, padx=15)
        
        # Note: Obstacle penalty removed - handled by hard constraints
        
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
    def _initialize_map(self):
        """Initialize the map visualization."""
        self.ax_map.clear()
        self.ax_map.set_title('Environment Map & Robot Paths', fontsize=12, fontweight='bold')
        self.ax_map.set_xlabel('Y Coordinate', fontsize=10)
        self.ax_map.set_ylabel('X Coordinate', fontsize=10)
        self.ax_map.set_xlim(-1, self.M)
        self.ax_map.set_ylim(self.N, -1)  # Inverted for correct orientation
        self.ax_map.set_aspect('equal')
        self.ax_map.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        
        # Draw grid background
        for i in range(self.N):
            for j in range(self.M):
                if self.map_grid[i, j] == 2:  # Obstacle
                    rect = Rectangle((j-0.5, i-0.5), 1, 1, 
                                   facecolor='#333333', edgecolor='black', linewidth=0.5)
                    self.ax_map.add_patch(rect)
                else:  # Free or unexplored
                    rect = Rectangle((j-0.5, i-0.5), 1, 1, 
                                   facecolor='#f5f5f5', edgecolor='#cccccc', linewidth=0.3)
                    self.ax_map.add_patch(rect)
        
        # Draw robot paths (faded)
        self.path_lines = []
        for r in range(self.R):
            path = self.path_array[r]
            y_coords = [p[1] for p in path]
            x_coords = [p[0] for p in path]
            line, = self.ax_map.plot(y_coords, x_coords, '--', 
                                    color=self.robot_colors[r], alpha=0.3, linewidth=1.5)
            self.path_lines.append(line)
        
        # Initialize robot markers
        self.robot_markers = []
        self.robot_trails = []
        for r in range(self.R):
            x, y = self.initial_positions[r]
            marker = Circle((y, x), 0.4, color=self.robot_colors[r], 
                          ec='black', linewidth=2, zorder=10)
            self.ax_map.add_patch(marker)
            self.robot_markers.append(marker)
            
            # Trail line
            trail, = self.ax_map.plot([], [], '-', color=self.robot_colors[r], 
                                     linewidth=2, alpha=0.7, zorder=5)
            self.robot_trails.append(trail)
        
        # Initialize communication links
        self.comm_links = []
        
        # Create legend
        legend_elements = []
        for r in range(self.R):
            legend_elements.append(Line2D([0], [0], marker='o', color='w', 
                                        markerfacecolor=self.robot_colors[r], 
                                        markersize=10, label=f'Robot {r+1}',
                                        markeredgecolor='black', markeredgewidth=1.5))
        legend_elements.append(Rectangle((0, 0), 1, 1, fc='#333333', label='Obstacle'))
        legend_elements.append(Rectangle((0, 0), 1, 1, fc='#90EE90', alpha=0.6, label='Explored'))
        legend_elements.append(Line2D([0], [0], color='cyan', linewidth=2, 
                                    alpha=0.5, label='Communication'))
        
        self.ax_map.legend(handles=legend_elements, loc='upper right', 
                          fontsize=8, framealpha=0.9)
        
        self.canvas.draw()
        
    def _update_visualization(self, step):
        """Update visualization for a given timestep."""
        self.current_step = step
        
        # Update progress bar
        self.progress['value'] = step
        self.step_label.config(text=f"Step: {step} / {self.K-1}")
        
        # Clear previous communication links
        for link in self.comm_links:
            link.remove()
        self.comm_links = []
        
        # Update robot positions and trails
        positions_t = []
        for r in range(self.R):
            x, y = self.path_array[r][step]
            positions_t.append((x, y))
            
            # Update marker position
            self.robot_markers[r].center = (y, x)
            
            # Update trail
            trail_y = [self.path_array[r][t][1] for t in range(step + 1)]
            trail_x = [self.path_array[r][t][0] for t in range(step + 1)]
            self.robot_trails[r].set_data(trail_y, trail_x)
            
            # Update visited cells
            if self.map_grid[x, y] == 0:  # Unexplored
                self.visited_cells.add((x, y))
                # Highlight explored cell
                rect = Rectangle((y-0.5, x-0.5), 1, 1, 
                               facecolor='#90EE90', edgecolor='#cccccc', 
                               linewidth=0.3, alpha=0.6, zorder=1)
                self.ax_map.add_patch(rect)
            
            # Update robot position label
            self.robot_labels[r].config(text=f"Robot {r+1}: ({x}, {y})")
        
        # Draw communication links
        for i in range(self.R):
            for j in range(i + 1, self.R):
                dist = math.sqrt((positions_t[i][0] - positions_t[j][0])**2 + 
                               (positions_t[i][1] - positions_t[j][1])**2)
                if dist <= self.communication_radius:
                    link, = self.ax_map.plot([positions_t[i][1], positions_t[j][1]], 
                                            [positions_t[i][0], positions_t[j][0]], 
                                            'c-', linewidth=2, alpha=0.5, zorder=3)
                    self.comm_links.append(link)
        
        # Update statistics
        self._update_statistics(step, positions_t)
        
        # Update plots
        self.canvas.draw_idle()
        
    def _update_statistics(self, step, positions_t):
        """Update statistical displays."""
        # Coverage
        coverage_pct = (len(self.visited_cells) / (self.N * self.M)) * 100
        self.coverage_history.append(coverage_pct)
        self.coverage_label.config(text=f"Coverage: {coverage_pct:.2f}%")
        self.cells_explored_label.config(text=f"Cells Explored: {len(self.visited_cells)}/{self.N*self.M}")
        
        # Connectivity
        connectivity_score = 0
        for i in range(self.R):
            for j in range(i + 1, self.R):
                dist = math.sqrt((positions_t[i][0] - positions_t[j][0])**2 + 
                               (positions_t[i][1] - positions_t[j][1])**2)
                if dist <= self.communication_radius:
                    connectivity_score += 1.0 / (1.0 + dist)
        
        self.connectivity_history.append(connectivity_score)
        self.connectivity_label.config(text=f"Connectivity: {connectivity_score:.2f}")
        
        # Network status (simplified check)
        max_possible_links = self.R * (self.R - 1) // 2
        num_active_links = len(self.comm_links)
        if num_active_links >= self.R - 1:  # Minimum spanning tree
            self.network_status_label.config(text="Network: Connected ✓", 
                                           foreground='green')
        else:
            self.network_status_label.config(text="Network: Disconnected ✗", 
                                           foreground='red')
        
        # Update objective function (calculate incrementally)
        coverage_term = self.alpha * len(self.visited_cells)
        connectivity_term = self.beta * connectivity_score
        
        self.obj_total_label.config(text=f"Total: {coverage_term + connectivity_term:.2f}")
        self.obj_coverage_label.config(text=f"Coverage (α={self.alpha}): {coverage_term:.2f}")
        self.obj_connectivity_label.config(text=f"Connectivity (β={self.beta}): {connectivity_term:.2f}")
        
    def _play(self):
        """Start animation playback."""
        if not self.is_playing:
            self.is_playing = True
            self.play_button.config(state=tk.DISABLED)
            self.pause_button.config(state=tk.NORMAL)
            self._animate()
            
    def _pause(self):
        """Pause animation playback."""
        self.is_playing = False
        self.play_button.config(state=tk.NORMAL)
        self.pause_button.config(state=tk.DISABLED)
        
    def _reset(self):
        """Reset animation to beginning."""
        self.is_playing = False
        self.play_button.config(state=tk.NORMAL)
        self.pause_button.config(state=tk.DISABLED)
        self.current_step = 0
        self.visited_cells.clear()
        self.coverage_history.clear()
        self.connectivity_history.clear()
        self._initialize_map()
        self._update_visualization(0)
        
    def _animate(self):
        """Animate one frame."""
        if self.is_playing and self.current_step < self.K - 1:
            self.current_step += 1
            self._update_visualization(self.current_step)
            self.root.after(self.speed, self._animate)
        else:
            self.is_playing = False
            self.play_button.config(state=tk.NORMAL)
            self.pause_button.config(state=tk.DISABLED)
            
    def _update_speed(self, value):
        """Update animation speed."""
        self.speed = int(float(value))
        self.speed_label.config(text=f"{self.speed} ms/frame")
        
    def show(self):
        """Display the visualization window."""
        # Initial update
        self._update_visualization(0)
        
        # Start GUI event loop
        self.root.mainloop()
        
    def save_final_state(self, filename='visualization_final.png'):
        """Save the final state as an image."""
        self._update_visualization(self.K - 1)
        self.fig.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to {filename}")


def visualize_optimization_results(path_array, initial_positions, map_grid,
                                   communication_radius, connectivity_threshold,
                                   alpha, beta, gamma):
    """
    Convenience function to create and show visualization.
    
    Args:
        path_array: Robot paths
        initial_positions: Initial robot positions
        map_grid: Environment map
        communication_radius: Communication range
        connectivity_threshold: Connectivity threshold
        alpha, beta, gamma: Objective weights
    """
    viz = RobotPathVisualizer(
        path_array=path_array,
        initial_positions=initial_positions,
        map_grid=map_grid,
        communication_radius=communication_radius,
        connectivity_threshold=connectivity_threshold,
        alpha=alpha,
        beta=beta,
        gamma=gamma
    )
    viz.show()
    return viz


if __name__ == "__main__":
    # Test visualization with dummy data
    print("This module should be imported and used with actual optimization results.")
    print("Example usage:")
    print("  from visualization import visualize_optimization_results")
    print("  viz = visualize_optimization_results(path_array, initial_positions, ...)")
