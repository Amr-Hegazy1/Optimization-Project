"""
Base class for optimization visualization.

Provides common visualization components (maps, coverage, objective plots)
that can be extended by algorithm-specific visualizers.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.lines import Line2D
import tkinter as tk
from tkinter import ttk
from abc import ABC, abstractmethod
import math


class BaseVisualizer(ABC):
    """
    Abstract base class for optimization visualization.
    
    Provides common visualization components:
    - Environment map views (current and best solutions)
    - Coverage tracking and plotting
    - Objective function evolution
    - Statistics panel
    - Animation controls
    
    Algorithm-specific plots (e.g., temperature for SA, population for GA)
    should be added by subclasses.
    """
    
    def __init__(self, initial_positions, map_grid, 
                 communication_radius, connectivity_threshold,
                 alpha, beta, gamma, visualization_step_size=1):
        """
        Initialize the base visualizer.
        
        Args:
            initial_positions: list of initial (x, y) positions
            map_grid: 2D numpy array representing the map
            communication_radius: R_c for connectivity
            connectivity_threshold: threshold for network connectivity
            alpha, beta, gamma: objective function weights
            visualization_step_size: steps to skip in animation (default: 1)
        """
        self.initial_positions = initial_positions
        self.map_grid = map_grid
        self.communication_radius = communication_radius
        self.connectivity_threshold = connectivity_threshold
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.visualization_step_size = visualization_step_size
        
        self.R = len(initial_positions)  # Number of robots
        self.N, self.M = map_grid.shape  # Map dimensions
        
        # Optimization tracking - common to all algorithms
        self.iterations = []
        self.current_costs = []
        self.best_costs = []
        self.best_path = None
        self.best_cost = -np.inf
        
        # Animation state
        self.current_solution_path = None
        self.animation_step = 0
        self.is_animating = False
        self.pending_current_path = None
        self.pending_best_path = None
        self.pending_current_update = False
        self.pending_best_update = False
        self.pending_display_state = None
        self.display_iteration = None
        self.display_current_cost = None
        self.display_best_cost = None
        
        # Synchronization
        self.animation_complete = True
        self.waiting_for_animation = False
        
        # Fast mode replay
        self.is_replaying = False
        self.replay_index = 0
        self.replay_history = []
        
        # Robot visualization
        self.current_robot_markers = []
        self.current_robot_trails = []
        self.current_robot_texts = []
        self.current_visited = set()
        
        self.best_robot_markers = []
        self.best_robot_trails = []
        self.best_robot_texts = []
        self.best_visited = set()
        
        # Coverage tracking
        self.current_coverage_data = []
        
        # Network connections
        self.current_network_lines = []
        self.best_network_lines = []
        
        # Color scheme
        self.robot_colors = plt.cm.tab10(np.linspace(0, 1, self.R))
        
        # Create GUI
        self._create_gui()
    
    def _create_gui(self):
        """Create the main GUI window."""
        self.root = tk.Tk()
        self.root.title(f"{self.get_algorithm_name()} Optimization - Real-time Visualization")
        self.root.geometry("1800x1000")
        self.root.configure(bg='#f0f0f0')
        
        # Create main container
        main_container = ttk.Frame(self.root)
        main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Title
        title_label = tk.Label(main_container, 
                              text=f"Multi-Robot Path Planning - {self.get_algorithm_name()} Optimization",
                              font=('Arial', 16, 'bold'),
                              bg='#f0f0f0')
        title_label.pack(pady=(0, 5))
        
        # Control panel
        self._create_control_panel(main_container)
        
        # Create visualization panel
        self._create_visualization_panel(main_container)
        
        # Status bar
        self.status_label = tk.Label(main_container, 
                                     text="Status: Waiting for optimization to start...",
                                     font=('Arial', 10),
                                     bg='#f0f0f0',
                                     anchor=tk.W)
        self.status_label.pack(fill=tk.X, pady=(10, 0))
    
    def _create_control_panel(self, parent):
        """Create control panel with animation controls."""
        control_frame = ttk.LabelFrame(parent, text="Visualization Controls", padding=10)
        control_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Control frame
        controls_frame = ttk.Frame(control_frame)
        controls_frame.pack(fill=tk.X)
        
        # Network toggle
        self.show_network_var = tk.BooleanVar(value=True)
        self.network_checkbox = ttk.Checkbutton(controls_frame, 
                                                text="Show Network Connections",
                                                variable=self.show_network_var,
                                                command=self._toggle_network)
        self.network_checkbox.pack(side=tk.LEFT, padx=(0, 20))
        
        # Step size control
        ttk.Label(controls_frame, text="Visualization Step Size:", font=('Arial', 10, 'bold')).pack(side=tk.LEFT, padx=(20, 10))
        
        self.decrease_step_btn = ttk.Button(controls_frame, text="◄ Decrease", 
                                            command=self._decrease_step_size, width=12)
        self.decrease_step_btn.pack(side=tk.LEFT, padx=5)
        
        self.step_size_display = tk.Label(controls_frame, 
                                         text=f"Step: {self.visualization_step_size}",
                                         font=('Arial', 10),
                                         bg='#f0f0f0',
                                         width=15,
                                         relief=tk.SUNKEN,
                                         bd=2)
        self.step_size_display.pack(side=tk.LEFT, padx=5)
        
        self.increase_step_btn = ttk.Button(controls_frame, text="Increase ►", 
                                            command=self._increase_step_size, width=12)
        self.increase_step_btn.pack(side=tk.LEFT, padx=5)
        
        self.reset_step_size_btn = ttk.Button(controls_frame, text="Reset Step", 
                                              command=self._reset_step_size, width=12)
        self.reset_step_size_btn.pack(side=tk.LEFT, padx=5)
        self.reset_step_size_btn.pack(side=tk.LEFT, padx=5)
    
    @abstractmethod
    def _create_visualization_panel(self, parent):
        """
        Create the matplotlib visualization panel.
        
        Subclasses must implement this to create their specific layout.
        Should include:
        - Map views (current and best solutions)
        - Coverage plot
        - Objective function plot
        - Algorithm-specific plots (e.g., temperature for SA)
        - Statistics panel
        
        Must call _initialize_common_plots() and _initialize_robot_markers()
        at the end of implementation.
        """
        pass
    
    def _create_common_map_axes(self, gs, fig):
        """
        Create common map axes (current and best solution maps).
        Helper method for subclasses to use in their layout.
        
        Args:
            gs: GridSpec object
            fig: Figure object
            
        Returns:
            tuple: (ax_current, ax_best, ax_coverage, ax_stats)
        """
        # Current solution map
        ax_current = fig.add_subplot(gs[0, 1])
        ax_current.set_title('Current Solution', fontsize=13, fontweight='bold', pad=10)
        ax_current.set_xlabel('Y Coordinate', fontsize=11)
        ax_current.set_ylabel('X Coordinate', fontsize=11)
        ax_current.set_aspect('equal')
        ax_current.grid(True, alpha=0.2, linestyle='--', linewidth=0.5)
        
        # Best solution map
        ax_best = fig.add_subplot(gs[1, 1])
        ax_best.set_title('Best Solution So Far', fontsize=13, fontweight='bold', pad=10)
        ax_best.set_xlabel('Y Coordinate', fontsize=11)
        ax_best.set_ylabel('X Coordinate', fontsize=11)
        ax_best.set_aspect('equal')
        ax_best.grid(True, alpha=0.2, linestyle='--', linewidth=0.5)
        
        # Coverage plot
        ax_coverage = fig.add_subplot(gs[0, 2])
        ax_coverage.set_title('Coverage Over Iterations', fontsize=12, fontweight='bold', pad=10)
        ax_coverage.set_xlabel('Iteration', fontsize=10)
        ax_coverage.set_ylabel('Coverage (%)', fontsize=10)
        ax_coverage.grid(True, alpha=0.3, linestyle='--')
        ax_coverage.set_ylim(0, 100)
        
        # Statistics panel
        ax_stats = fig.add_subplot(gs[1, 2])
        ax_stats.axis('off')
        ax_stats.set_title('Statistics', fontsize=12, fontweight='bold', pad=10)
        
        return ax_current, ax_best, ax_coverage, ax_stats
    
    def _initialize_common_plots(self):
        """Initialize common plots (objective function, coverage, maps)."""
        # Coverage plot
        self.coverage_line, = self.ax_coverage.plot([], [], color='#1f77b4', linewidth=2.5)
        
        # Initialize map views
        self._initialize_map_view(self.ax_current, "Current Solution")
        self._initialize_map_view(self.ax_best, "Best Solution So Far")
        
        self.canvas.draw()
    
    def _initialize_robot_markers(self):
        """Initialize robot markers and trails for animations."""
        # Current solution markers
        self.current_robot_markers = []
        self.current_robot_trails = []
        self.current_robot_texts = []
        
        for r in range(self.R):
            x, y = self.initial_positions[r]
            
            marker = Circle((y, x), 0.4, color=self.robot_colors[r], 
                          ec='black', linewidth=2, zorder=10, alpha=0.9)
            self.ax_current.add_patch(marker)
            self.current_robot_markers.append(marker)
            
            trail, = self.ax_current.plot([], [], '-', color=self.robot_colors[r], 
                                         linewidth=2, alpha=0.6, zorder=5)
            self.current_robot_trails.append(trail)
            
            text = self.ax_current.text(y + 0.7, x, f'R{r+1}', ha='left', va='center',
                                       fontsize=9, fontweight='bold', zorder=11,
                                       bbox=dict(boxstyle='round,pad=0.2', facecolor='white', 
                                                edgecolor=self.robot_colors[r], linewidth=1.5))
            self.current_robot_texts.append(text)
        
        # Best solution markers
        self.best_robot_markers = []
        self.best_robot_trails = []
        self.best_robot_texts = []
        
        for r in range(self.R):
            x, y = self.initial_positions[r]
            
            marker = Circle((y, x), 0.4, color=self.robot_colors[r], 
                          ec='black', linewidth=2, zorder=10, alpha=0.9)
            self.ax_best.add_patch(marker)
            self.best_robot_markers.append(marker)
            
            trail, = self.ax_best.plot([], [], '-', color=self.robot_colors[r], 
                                      linewidth=2, alpha=0.6, zorder=5)
            self.best_robot_trails.append(trail)
            
            text = self.ax_best.text(y + 0.7, x, f'R{r+1}', ha='left', va='center',
                                    fontsize=9, fontweight='bold', zorder=11,
                                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white', 
                                             edgecolor=self.robot_colors[r], linewidth=1.5))
            self.best_robot_texts.append(text)
    
    def _initialize_map_view(self, ax, title):
        """Initialize a map view with grid and obstacles."""
        ax.clear()
        ax.set_title(title, fontsize=12, fontweight='bold', pad=10)
        ax.set_xlabel('Y Coordinate', fontsize=10)
        ax.set_ylabel('X Coordinate', fontsize=10)
        ax.set_xlim(-1, self.M)
        ax.set_ylim(self.N, -1)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.5)
        
        # Draw grid background
        for i in range(self.N):
            for j in range(self.M):
                if self.map_grid[i, j] == 2:  # Obstacle
                    rect = Rectangle((j-0.5, i-0.5), 1, 1, 
                                   facecolor='#333333', edgecolor='#555555', linewidth=0.5)
                    ax.add_patch(rect)
                else:
                    rect = Rectangle((j-0.5, i-0.5), 1, 1, 
                                   facecolor='#fafafa', edgecolor='#e0e0e0', linewidth=0.3)
                    ax.add_patch(rect)
        
        # Show initial positions with faded markers
        for r, (x, y) in enumerate(self.initial_positions):
            circle = Circle((y, x), 0.35, color=self.robot_colors[r], 
                          ec='black', linewidth=1.5, zorder=10, alpha=0.3)
            ax.add_patch(circle)
    
    @abstractmethod
    def update_optimization(self, **kwargs):
        """
        Update visualization with new optimization data.
        
        Subclasses must implement this to handle algorithm-specific parameters.
        Should call _update_common_data() to update shared components.
        
        Required kwargs:
            iteration: current iteration number
            current_cost: cost of current solution
            best_cost: cost of best solution found
            best_path: best path array found
            current_path: current path array (optional)
        """
        pass
    
    @abstractmethod
    def _update_plots(self):
        """
        Update algorithm-specific plots with current data.
        
        This method should update all plots (objective function, temperature, etc.)
        based on the stored data (self.iterations, self.current_costs, etc.).
        Called during both real-time updates and fast mode replay.
        """
        pass
    
    def _update_common_data(self, iteration, current_cost, best_cost, best_path, current_path=None):
        """
        Update common optimization data.
        
        This should be called by subclasses in their update_optimization implementation.
        """
        self.animation_complete = False
        
        self.iterations.append(iteration)
        self.current_costs.append(current_cost)
        self.best_costs.append(best_cost)
        
        if best_cost > self.best_cost:
            self.best_cost = best_cost
            self.pending_best_path = best_path
            self.pending_best_update = True
        
        if current_path is not None:
            self.pending_current_path = current_path
            self.pending_current_update = True
        
        # If animation is idle, apply updates and restart
        if not self.is_animating:
            if self.pending_best_update or self.pending_current_update:
                self._apply_pending_visual_updates()
            if self.current_solution_path is not None or self.best_path is not None:
                self.animation_step = 0
                self.current_coverage_data = []
                self.is_animating = True
                self.root.after(0, self._animate_solutions)
    
    def _apply_pending_visual_updates(self):
        """Apply queued map resets once animation is idle."""
        if self.pending_best_update and self.pending_best_path is not None:
            for line in self.best_network_lines:
                try:
                    line.remove()
                except Exception:
                    pass
            self.best_network_lines = []
            self.best_visited.clear()
            self.best_path = self.pending_best_path
            self.pending_best_path = None
            self._initialize_map_view(self.ax_best, "Best Solution So Far")
            self._initialize_robot_markers_for_ax(
                self.ax_best,
                self.best_robot_markers,
                self.best_robot_trails,
                self.best_robot_texts,
            )
            self.pending_best_update = False
        
        if self.pending_current_update and self.pending_current_path is not None:
            for line in self.current_network_lines:
                try:
                    line.remove()
                except Exception:
                    pass
            self.current_network_lines = []
            self.current_visited.clear()
            self.current_solution_path = self.pending_current_path
            self.pending_current_path = None
            self._initialize_map_view(self.ax_current, "Current Solution")
            self._initialize_robot_markers_for_ax(
                self.ax_current,
                self.current_robot_markers,
                self.current_robot_trails,
                self.current_robot_texts,
            )
            self.pending_current_update = False
            self.coverage_line.set_data([], [])
            self.ax_coverage.set_xlim(0, 1)
            self.ax_coverage.set_ylim(0, 100)
        
        if self.pending_display_state is not None:
            for key, value in self.pending_display_state.items():
                setattr(self, f"display_{key}", value)
            self.pending_display_state = None
            self._update_statistics()
            self._update_status_bar()
            self.canvas.draw_idle()
    
    def _initialize_robot_markers_for_ax(self, ax, markers, trails, texts):
        """Reinitialize robot markers for a specific axes."""
        markers.clear()
        trails.clear()
        texts.clear()
        
        for r in range(self.R):
            x, y = self.initial_positions[r]
            
            marker = Circle((y, x), 0.4, color=self.robot_colors[r], 
                          ec='black', linewidth=2, zorder=10, alpha=0.9)
            ax.add_patch(marker)
            markers.append(marker)
            
            trail, = ax.plot([], [], '-', color=self.robot_colors[r], 
                            linewidth=2, alpha=0.6, zorder=5)
            trails.append(trail)
            
            text = ax.text(y + 0.7, x, f'R{r+1}', ha='left', va='center',
                          fontsize=9, fontweight='bold', zorder=11,
                          bbox=dict(boxstyle='round,pad=0.2', facecolor='white', 
                                   edgecolor=self.robot_colors[r], linewidth=1.5))
            texts.append(text)
    
    def wait_for_animation_complete(self):
        """Block until current animation is complete."""
        self.waiting_for_animation = True
        while not self.animation_complete and self.waiting_for_animation:
            self.root.update()
            import time
            time.sleep(0.001)
        self.waiting_for_animation = False
    
    def _animate_solutions(self):
        """Animate both current and best solutions simultaneously."""
        if self.current_solution_path is None and self.best_path is None:
            self.is_animating = False
            self.animation_complete = True
            return
        
        K = 0
        if self.current_solution_path is not None:
            K = max(K, len(self.current_solution_path[0]))
        if self.best_path is not None:
            K = max(K, len(self.best_path[0]))
        
        if self.animation_step < K:
            if self.current_solution_path is not None:
                self._update_robot_animation(
                    self.current_solution_path,
                    self.current_robot_markers,
                    self.current_robot_trails,
                    self.current_robot_texts,
                    self.current_visited,
                    self.ax_current,
                    self.animation_step,
                    f"Current Solution (Step {self.animation_step + 1}/{K})",
                    self.current_network_lines
                )
                self._update_current_solution_coverage_incremental(self.animation_step)
            
            if self.best_path is not None:
                self._update_robot_animation(
                    self.best_path,
                    self.best_robot_markers,
                    self.best_robot_trails,
                    self.best_robot_texts,
                    self.best_visited,
                    self.ax_best,
                    self.animation_step,
                    f"Best Solution (Step {self.animation_step + 1}/{K})",
                    self.best_network_lines
                )
            
            self.canvas.draw_idle()
            self.animation_step += self.visualization_step_size
            # Use a fixed delay of 50ms per animation frame
            self.root.after(50, self._animate_solutions)
        else:
            self.animation_step = 0
            self.is_animating = False
            self.animation_complete = True
            if self.pending_best_update or self.pending_current_update:
                self._apply_pending_visual_updates()
                if self.current_solution_path is not None or self.best_path is not None:
                    self.animation_step = 0
                    self.current_coverage_data = []
                    self.is_animating = True
                    self.animation_complete = False
                    self.root.after(0, self._animate_solutions)
    
    def _update_robot_animation(self, path_array, markers, trails, texts, visited_set, ax, step, title, network_lines=None):
        """Update robot positions for one animation frame."""
        if step >= len(path_array[0]):
            return
        
        if len(markers) != self.R:
            return
        
        if network_lines is not None:
            for line in network_lines:
                try:
                    line.remove()
                except:
                    pass
            network_lines.clear()
        
        current_positions = []
        
        for r in range(self.R):
            pos = path_array[r][step]
            pos_tuple = tuple(pos) if isinstance(pos, np.ndarray) else pos
            x, y = pos_tuple
            current_positions.append((x, y))
            
            markers[r].center = (y, x)
            texts[r].set_position((y + 0.7, x))
            
            trail_y = []
            trail_x = []
            for s in range(step + 1):
                p = path_array[r][s]
                p_tuple = tuple(p) if isinstance(p, np.ndarray) else p
                trail_y.append(p_tuple[1])
                trail_x.append(p_tuple[0])
            trails[r].set_data(trail_y, trail_x)
            
            prev_step = max(0, step - self.visualization_step_size)
            for s in range(prev_step, step + 1):
                if s < len(path_array[r]):
                    p = path_array[r][s]
                    p_tuple = tuple(p) if isinstance(p, np.ndarray) else p
                    cell_x, cell_y = p_tuple
                    
                    if self.map_grid[cell_x, cell_y] == 0 and p_tuple not in visited_set:
                        visited_set.add(p_tuple)
                        rect = Rectangle((cell_y-0.5, cell_x-0.5), 1, 1, 
                                       facecolor='#90EE90', edgecolor='#cccccc', 
                                       linewidth=0.3, alpha=0.5, zorder=1)
                        ax.add_patch(rect)
        
        if self.show_network_var.get() and network_lines is not None:
            for i in range(self.R):
                for j in range(i + 1, self.R):
                    x1, y1 = current_positions[i]
                    x2, y2 = current_positions[j]
                    dist = math.sqrt((x1 - x2)**2 + (y1 - y2)**2)
                    
                    if dist <= self.communication_radius:
                        if dist <= self.connectivity_threshold:
                            line, = ax.plot([y1, y2], [x1, x2], 
                                          'c-', linewidth=2.5, alpha=0.6, zorder=3)
                        else:
                            line, = ax.plot([y1, y2], [x1, x2], 
                                          'c--', linewidth=1.5, alpha=0.4, zorder=3)
                        network_lines.append(line)
        
        coverage = (len(visited_set) / (self.N * self.M)) * 100
        num_connections = 0
        if self.show_network_var.get():
            for i in range(self.R):
                for j in range(i + 1, self.R):
                    x1, y1 = current_positions[i]
                    x2, y2 = current_positions[j]
                    dist = math.sqrt((x1 - x2)**2 + (y1 - y2)**2)
                    if dist <= self.communication_radius:
                        num_connections += 1
        
        max_connections = self.R * (self.R - 1) // 2
        ax.set_title(f"{title} - Coverage: {coverage:.1f}% | Connections: {num_connections}/{max_connections}", 
                    fontsize=13, fontweight='bold', pad=10)
    
    def _update_current_solution_coverage_incremental(self, step):
        """Update coverage plot incrementally as animation progresses."""
        if self.current_solution_path is None:
            return
        
        if step >= len(self.current_solution_path[0]):
            return
        
        visited = set()
        for r in range(self.R):
            for t in range(min(step + 1, len(self.current_solution_path[r]))):
                pos = self.current_solution_path[r][t]
                pos_tuple = tuple(pos) if isinstance(pos, np.ndarray) else pos
                if self.map_grid[pos_tuple[0], pos_tuple[1]] == 0:
                    visited.add(pos_tuple)
        
        coverage_pct = (len(visited) / (self.N * self.M)) * 100
        self.current_coverage_data.append(coverage_pct)
        
        self.coverage_line.set_data(range(len(self.current_coverage_data)), self.current_coverage_data)
        K = len(self.current_solution_path[0])
        self.ax_coverage.set_xlim(0, max(K - 1, 1))
        self.ax_coverage.set_ylim(0, 100)
    
    @abstractmethod
    def _update_statistics(self):
        """
        Update statistics panel.
        
        Subclasses should implement this to show algorithm-specific statistics.
        """
        pass
    
    def _update_status_bar(self, final=False, final_coverage=None):
        """Update status bar text."""
        if len(self.iterations) == 0:
            status_text = "Status: Waiting for optimization to start..."
        else:
            prefix = "Status: ✓ Optimization Complete!" if final else "Status: Optimizing..."
            status_text = (
                f"{prefix} | Iteration: {self.display_iteration or self.iterations[-1]} | "
                f"Best Cost: {(self.display_best_cost if self.display_best_cost is not None else self.best_costs[-1]):.2f}"
            )
            if final and final_coverage is not None:
                status_text += f" | Final Coverage: {final_coverage:.1f}%"
                status_text += f" | Total Iterations: {self.iterations[-1]}"
        self.status_label.config(text=status_text)
    
    def finish_optimization(self):
        """Called when optimization is complete."""
        self.is_animating = False
        
        # Update display state
        self.display_iteration = self.iterations[-1] if self.iterations else None
        self.display_current_cost = self.current_costs[-1] if self.current_costs else None
        self.display_best_cost = self.best_costs[-1] if self.best_costs else None
        
        # Calculate final coverage
        final_coverage = 0
        if self.best_path is not None:
            visited = set()
            for r in range(self.R):
                for pos in self.best_path[r]:
                    pos_tuple = tuple(pos) if isinstance(pos, np.ndarray) else pos
                    if self.map_grid[pos_tuple[0], pos_tuple[1]] == 0:
                        visited.add(pos_tuple)
            final_coverage = (len(visited) / (self.N * self.M)) * 100
        
        self._update_status_bar(final=True, final_coverage=final_coverage)
        self.canvas.draw()
    
    def replay_optimization_history(self, history):
        """
        Replay optimization history in fast mode after optimization completes.
        
        Args:
            history (list): List of kwargs dictionaries from update_visualization calls
        """
        if not history:
            # No history to replay, just finish normally
            self.finish_optimization()
            return
        
        # Get update interval from config
        try:
            import config
            update_interval = config.FAST_MODE_UPDATE_INTERVAL if hasattr(config, 'FAST_MODE_UPDATE_INTERVAL') else 10
        except ImportError:
            update_interval = 10
        
        print(f"\n--- Fast Mode: Replaying {len(history)} optimization steps ---")
        print(f"Updating GUI every {update_interval} iterations")
        self._update_status_bar_text(f"Fast Mode: Replaying optimization history (update every {update_interval} iterations)...")
        
        # Reset coverage data for replay
        self.current_coverage_data = []
        
        # Replay history with automatic animation
        self.replay_index = 0
        self.replay_history = history
        self.replay_update_interval = update_interval
        self.is_replaying = True
        
        # Start replay animation
        self._replay_next_step()
    
    def _replay_next_step(self):
        """Replay the next step in the optimization history."""
        if self.replay_index >= len(self.replay_history):
            # Replay complete
            self.is_replaying = False
            print("--- Fast Mode Replay Complete ---")
            self.finish_optimization()
            return
        
        # Get next history entry
        kwargs = self.replay_history[self.replay_index]
        self.replay_index += 1
        
        # Always update the data (for all iterations)
        # This ensures plots have complete data even if we skip some GUI updates
        if hasattr(self, 'temperatures') and 'temperature' in kwargs:
            # SA-specific: track temperature
            if self.initial_temperature is None:
                self.initial_temperature = kwargs.get('temperature')
            self.temperatures.append(kwargs.get('temperature'))
        
        # Update common tracking data
        iteration = kwargs.get('iteration')
        current_cost = kwargs.get('current_cost')
        best_cost = kwargs.get('best_cost')
        best_path = kwargs.get('best_path')
        current_path = kwargs.get('current_path')
        
        if iteration is not None:
            self.iterations.append(iteration)
        if current_cost is not None:
            self.current_costs.append(current_cost)
        if best_cost is not None:
            self.best_costs.append(best_cost)
        if best_path is not None:
            self.best_path = best_path
            self.best_cost = best_cost
        
        # Update display state
        self.display_iteration = iteration
        self.display_current_cost = current_cost
        self.display_best_cost = best_cost
        if hasattr(self, 'display_temperature') and 'temperature' in kwargs:
            self.display_temperature = kwargs.get('temperature')
        
        # Only trigger full GUI update (with animation) every N iterations
        should_update_gui = (self.replay_index % self.replay_update_interval == 0 or 
                            self.replay_index == len(self.replay_history))
        
        if should_update_gui:
            # Update plots first
            self._update_plots()
            
            # For fast mode replay, show final state without animation
            # This is faster and still shows progress
            if current_path is not None:
                self.current_solution_path = current_path
                # Show final state of current solution
                self._update_map_snapshot(
                    current_path,
                    self.ax_current,
                    "Current Solution",
                    self.current_robot_markers,
                    self.current_robot_trails,
                    self.current_robot_texts,
                    self.current_visited,
                    self.current_network_lines
                )
            
            if best_path is not None:
                self.best_path = best_path
                # Show final state of best solution
                self._update_map_snapshot(
                    best_path,
                    self.ax_best,
                    "Best Solution So Far",
                    self.best_robot_markers,
                    self.best_robot_trails,
                    self.best_robot_texts,
                    self.best_visited,
                    self.best_network_lines
                )
            
            # Update coverage data for current solution
            if current_path is not None:
                # Count unique visited unexplored cells
                visited_cells = set()
                for r in range(self.R):
                    for pos in current_path[r]:
                        pos_tuple = tuple(pos) if isinstance(pos, np.ndarray) else pos
                        if self.map_grid[pos_tuple[0], pos_tuple[1]] == 0:
                            visited_cells.add(pos_tuple)
                
                coverage_pct = (len(visited_cells) / (self.N * self.M)) * 100
                
                # Update coverage plot - use iteration count as x-axis
                if iteration is not None:
                    self.current_coverage_data.append((iteration, coverage_pct))
                    if self.current_coverage_data and hasattr(self, 'coverage_line'):
                        iters, coverages = zip(*self.current_coverage_data)
                        self.coverage_line.set_data(iters, coverages)
                        self.ax_coverage.relim()
                        self.ax_coverage.autoscale_view()
                        # Update x-axis label
                        self.ax_coverage.set_xlabel('Iteration', fontsize=10)
            
            # Force canvas redraw
            self.canvas.draw_idle()
            
            # Schedule next step with delay to allow GUI to update
            self.root.after(10, self._replay_next_step)
        else:
            # Skip GUI update but update plots without animation
            # Only update plots every N/10 iterations to avoid overhead
            if self.replay_index % max(1, self.replay_update_interval // 10) == 0:
                self._update_plots()
            
            # Continue with next step using after to avoid stack overflow
            self.root.after(0, self._replay_next_step)
    
    def _update_status_bar_text(self, text):
        """Update status bar with custom text."""
        if hasattr(self, 'status_label'):
            self.status_label.config(text=f"Status: {text}")
    
    def _update_map_snapshot(self, path_array, ax, title, markers, trails, texts, visited_set, network_lines):
        """
        Update map to show final state of a path without animation.
        Used during fast mode replay for quick updates.
        """
        if path_array is None or len(path_array) == 0:
            return
        
        K = len(path_array[0])
        final_step = K - 1
        
        # Clear existing markers and trails
        for marker in markers:
            if marker is not None:
                try:
                    marker.remove()
                except:
                    pass
        for trail in trails:
            if trail is not None:
                try:
                    trail.remove()
                except:
                    pass
        for text in texts:
            if text is not None:
                try:
                    text.remove()
                except:
                    pass
        for line in network_lines:
            try:
                line.remove()
            except:
                pass
        
        markers.clear()
        trails.clear()
        texts.clear()
        network_lines.clear()
        visited_set.clear()
        
        # Redraw map
        self._initialize_map_view(ax, title)
        
        # Track visited cells first
        for r in range(self.R):
            path_r = path_array[r]
            for pos in path_r[:final_step+1]:
                pos_tuple = tuple(pos) if isinstance(pos, np.ndarray) else pos
                if self.map_grid[pos_tuple[0], pos_tuple[1]] == 0:
                    visited_set.add(pos_tuple)
        
        # Draw visited cells as green rectangles
        for (i, j) in visited_set:
            rect = Rectangle((j-0.5, i-0.5), 1, 1, 
                           facecolor='#90EE90', edgecolor='#e0e0e0', 
                           linewidth=0.3, alpha=0.6, zorder=2)
            ax.add_patch(rect)
        
        # Draw complete trails for all robots
        for r in range(self.R):
            path_r = path_array[r]
            # Draw trail
            trail_x = [pos[1] for pos in path_r[:final_step+1]]
            trail_y = [pos[0] for pos in path_r[:final_step+1]]
            trail_line, = ax.plot(trail_x, trail_y, '-', 
                                 color=self.robot_colors[r], 
                                 linewidth=2, alpha=0.6, zorder=5)
            trails.append(trail_line)
            
            # Draw final position
            if final_step < len(path_r):
                final_pos = path_r[final_step]
                y, x = final_pos[1], final_pos[0]
                circle = Circle((y, x), 0.4, color=self.robot_colors[r], 
                              ec='black', linewidth=2, zorder=15)
                ax.add_patch(circle)
                markers.append(circle)
                
                # Add robot label
                text_obj = ax.text(y, x, f'R{r+1}', 
                                  ha='center', va='center', 
                                  fontsize=8, fontweight='bold',
                                  color='white', zorder=20)
                texts.append(text_obj)
        
        # Draw network connections at final step
        if self.show_network_var.get() and final_step < K:
            positions_t = [path_array[r][final_step] if final_step < len(path_array[r]) 
                          else path_array[r][-1] for r in range(self.R)]
            
            for i in range(self.R):
                for j in range(i + 1, self.R):
                    pos_i = positions_t[i]
                    pos_j = positions_t[j]
                    dist = np.sqrt((pos_i[0] - pos_j[0])**2 + (pos_i[1] - pos_j[1])**2)
                    
                    if dist <= self.communication_radius:
                        line, = ax.plot([pos_i[1], pos_j[1]], [pos_i[0], pos_j[0]], 
                                       'g--', alpha=0.3, linewidth=1.5, zorder=3)
                        network_lines.append(line)
    
    def show(self):
        """Display the visualization window."""
        self.root.mainloop()
    
    def save_figure(self, filename='optimization_results.png'):
        """Save the current figure."""
        self.fig.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {filename}")
    
    @abstractmethod
    def get_algorithm_name(self):
        """
        Get the name of the algorithm being visualized.
        
        Returns:
            str: Algorithm name (e.g., "Simulated Annealing")
        """
        pass
    
    # Animation control methods
    def _toggle_network(self):
        """Toggle network connections visibility."""
        pass
    
    def _decrease_step_size(self):
        """Decrease visualization step size."""
        if self.visualization_step_size > 1:
            self.visualization_step_size -= 1
            self.step_size_display.config(text=f"Step: {self.visualization_step_size}")
    
    def _increase_step_size(self):
        """Increase visualization step size."""
        self.visualization_step_size += 1
        self.step_size_display.config(text=f"Step: {self.visualization_step_size}")
    
    def _reset_step_size(self):
        """Reset visualization step size to default."""
        self.visualization_step_size = 1
        self.step_size_display.config(text=f"Step: {self.visualization_step_size}")
