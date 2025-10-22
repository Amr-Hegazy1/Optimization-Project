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
                 alpha, beta, gamma, zeta, visualization_step_size=1):
        """
        Initialize the base visualizer.
        
        Args:
            initial_positions: list of initial (x, y) positions
            map_grid: 2D numpy array representing the map
            communication_radius: R_c for connectivity
            connectivity_threshold: threshold for network connectivity
            alpha, beta, gamma, zeta: objective function weights
            visualization_step_size: steps to skip in animation (default: 1)
        """
        self.initial_positions = initial_positions
        self.map_grid = map_grid
        self.communication_radius = communication_radius
        self.connectivity_threshold = connectivity_threshold
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.zeta = zeta
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
        self.animation_speed = 50  # milliseconds per step (can be overridden by config)
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
        control_frame = ttk.LabelFrame(parent, text="Animation Controls", padding=10)
        control_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Speed control
        speed_frame = ttk.Frame(control_frame)
        speed_frame.pack(fill=tk.X)
        
        ttk.Label(speed_frame, text="Animation Speed:", font=('Arial', 10, 'bold')).pack(side=tk.LEFT, padx=(0, 10))
        
        self.slower_btn = ttk.Button(speed_frame, text="◄◄ Slower", 
                                     command=self._slower_animation, width=12)
        self.slower_btn.pack(side=tk.LEFT, padx=5)
        
        self.speed_display = tk.Label(speed_frame, 
                                     text=f"{self.animation_speed} ms/frame",
                                     font=('Arial', 10),
                                     bg='#f0f0f0',
                                     width=15,
                                     relief=tk.SUNKEN,
                                     bd=2)
        self.speed_display.pack(side=tk.LEFT, padx=5)
        
        self.faster_btn = ttk.Button(speed_frame, text="Faster ►►", 
                                     command=self._faster_animation, width=12)
        self.faster_btn.pack(side=tk.LEFT, padx=5)
        
        self.reset_speed_btn = ttk.Button(speed_frame, text="Reset Speed", 
                                          command=self._reset_speed, width=12)
        self.reset_speed_btn.pack(side=tk.LEFT, padx=5)
        
        # Network toggle
        self.show_network_var = tk.BooleanVar(value=True)
        self.network_checkbox = ttk.Checkbutton(speed_frame, 
                                                text="Show Network Connections",
                                                variable=self.show_network_var,
                                                command=self._toggle_network)
        self.network_checkbox.pack(side=tk.LEFT, padx=20)
        
        # Step size control
        step_size_frame = ttk.Frame(control_frame)
        step_size_frame.pack(fill=tk.X, pady=(10, 0))
        
        ttk.Label(step_size_frame, text="Visualization Step Size:", font=('Arial', 10, 'bold')).pack(side=tk.LEFT, padx=(0, 10))
        
        self.decrease_step_btn = ttk.Button(step_size_frame, text="◄ Decrease", 
                                            command=self._decrease_step_size, width=12)
        self.decrease_step_btn.pack(side=tk.LEFT, padx=5)
        
        self.step_size_display = tk.Label(step_size_frame, 
                                         text=f"Step: {self.visualization_step_size}",
                                         font=('Arial', 10),
                                         bg='#f0f0f0',
                                         width=15,
                                         relief=tk.SUNKEN,
                                         bd=2)
        self.step_size_display.pack(side=tk.LEFT, padx=5)
        
        self.increase_step_btn = ttk.Button(step_size_frame, text="Increase ►", 
                                            command=self._increase_step_size, width=12)
        self.increase_step_btn.pack(side=tk.LEFT, padx=5)
        
        self.reset_step_size_btn = ttk.Button(step_size_frame, text="Reset Step", 
                                              command=self._reset_step_size, width=12)
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
        ax_coverage.set_title('Coverage Over Time (Current Solution)', fontsize=12, fontweight='bold', pad=10)
        ax_coverage.set_xlabel('Time Step', fontsize=10)
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
            self.root.after(self.animation_speed, self._animate_solutions)
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
    def _slower_animation(self):
        """Slow down animation speed."""
        self.animation_speed = min(500, self.animation_speed + 25)
        self.speed_display.config(text=f"{self.animation_speed} ms/frame")
    
    def _faster_animation(self):
        """Speed up animation speed."""
        if self.animation_speed > 50:
            self.animation_speed = max(1, self.animation_speed - 25)
        elif self.animation_speed > 10:
            self.animation_speed = max(1, self.animation_speed - 10)
        else:
            self.animation_speed = max(1, self.animation_speed - 1)
        self.speed_display.config(text=f"{self.animation_speed} ms/frame")
    
    def _reset_speed(self):
        """Reset animation speed to default."""
        self.animation_speed = 50
        self.speed_display.config(text=f"{self.animation_speed} ms/frame")
    
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
