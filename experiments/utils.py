"""
Shared helpers for running optimization experiments across SA, GA, and ACO.

These utilities keep config and BaseOptimizer globals in sync so each
experiment can safely tweak robot counts, path lengths, and objective weights
without leaking state between runs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

import config
from optimization import base_optimizer as bo

Position = Tuple[int, int]
PathArray = np.ndarray


# Snapshot defaults so we can restore them after experiments if needed.
DEFAULTS: Dict[str, object] = {
    "R": config.R,
    "PATH_LENGTH": config.PATH_LENGTH,
    "ALPHA": config.ALPHA,
    "BETA": config.BETA,
    "GAMMA": config.GAMMA,
    "ENERGY_BUDGET": config.ENERGY_BUDGET,
    "COMMUNICATION_RADIUS": config.COMMUNICATION_RADIUS,
    "CONNECTIVITY_THRESHOLD": config.CONNECTIVITY_THRESHOLD,
    "ROBOT_INITIAL_POSITIONS": tuple(config.ROBOT_INITIAL_POSITIONS),
}


def _set_attr(obj: object, name: str, value: object) -> None:
    """Set attribute on an object/module if it exists."""
    if hasattr(obj, name):
        setattr(obj, name, value)


def generate_positions(num_robots: int) -> List[Position]:
    """
    Return deterministic initial positions for the requested robot count.

    Uses the configured positions when possible; fills additional robots on a
    coarse grid so they start well-spaced when more robots are requested.
    """
    existing = list(config.ROBOT_INITIAL_POSITIONS)
    if num_robots <= len(existing):
        return existing[:num_robots]

    positions = existing[:]
    step = max(1, min(config.MAP_HEIGHT, config.MAP_WIDTH) // 4)
    for x in range(0, config.MAP_HEIGHT, step):
        for y in range(0, config.MAP_WIDTH, step):
            if len(positions) >= num_robots:
                break
            if (x, y) not in positions:
                positions.append((x, y))
        if len(positions) >= num_robots:
            break

    # Fallback: fill remaining positions sequentially.
    i = 0
    while len(positions) < num_robots:
        x, y = divmod(i, config.MAP_WIDTH)
        if (x, y) not in positions:
            positions.append((x, y))
        i += 1
    return positions


def apply_scenario(
    *,
    num_robots: int,
    path_length: int,
    alpha: float,
    beta: float,
    gamma: float,
    energy_budget: int | None = None,
    communication_radius: float | None = None,
    connectivity_threshold: float | None = None,
    robot_positions: Sequence[Position] | None = None,
) -> List[Position]:
    """
    Synchronize experiment parameters across config and BaseOptimizer globals.

    Returns the robot positions used for the scenario so callers can keep a
    local reference when constructing initial solutions.
    """
    positions = list(robot_positions) if robot_positions is not None else generate_positions(num_robots)
    energy = energy_budget if energy_budget is not None else path_length
    comms = communication_radius if communication_radius is not None else config.COMMUNICATION_RADIUS
    connectivity = connectivity_threshold if connectivity_threshold is not None else config.CONNECTIVITY_THRESHOLD

    # Update config module.
    config.R = num_robots
    config.PATH_LENGTH = path_length
    config.ALPHA = alpha
    config.BETA = beta
    config.GAMMA = gamma
    config.ENERGY_BUDGET = energy
    config.COMMUNICATION_RADIUS = comms
    config.CONNECTIVITY_THRESHOLD = connectivity
    config.ROBOT_INITIAL_POSITIONS = positions

    # Keep BaseOptimizer globals aligned with config.
    _set_attr(bo, "R", num_robots)
    _set_attr(bo, "PATH_LENGTH", path_length)
    _set_attr(bo, "ALPHA", alpha)
    _set_attr(bo, "BETA", beta)
    _set_attr(bo, "GAMMA", gamma)
    _set_attr(bo, "ENERGY_BUDGET", energy)
    _set_attr(bo, "COMMUNICATION_RADIUS", comms)
    _set_attr(bo, "CONNECTIVITY_THRESHOLD", connectivity)
    _set_attr(bo, "ROBOT_INITIAL_POSITIONS", positions)
    _set_attr(bo, "MAP", config.MAP)
    _set_attr(bo, "MAP_HEIGHT", config.MAP_HEIGHT)
    _set_attr(bo, "MAP_WIDTH", config.MAP_WIDTH)
    _set_attr(bo, "MOVES", config.MOVES)

    # Keep ant_colony module globals aligned (it imports config via `from config import *`).
    try:
        import ant_colony as aco_mod  # noqa: WPS433

        for name, value in [
            ("R", num_robots),
            ("PATH_LENGTH", path_length),
            ("MAP_HEIGHT", config.MAP_HEIGHT),
            ("MAP_WIDTH", config.MAP_WIDTH),
            ("MOVES", config.MOVES),
            ("ROBOT_INITIAL_POSITIONS", positions),
            ("ALPHA", alpha),
            ("BETA", beta),
            ("GAMMA", gamma),
            ("ENERGY_BUDGET", energy),
            ("COMMUNICATION_RADIUS", comms),
            ("CONNECTIVITY_THRESHOLD", connectivity),
            ("MAP", config.MAP),
            ("CELL_UNEXPLORED", getattr(config, "CELL_UNEXPLORED", 0)),
            ("CELL_FREE", getattr(config, "CELL_FREE", 1)),
            ("CELL_OBSTACLE", getattr(config, "CELL_OBSTACLE", 2)),
            ("CELL_ROBOT", getattr(config, "CELL_ROBOT", 3)),
        ]:
            _set_attr(aco_mod, name, value)
    except ImportError:
        pass

    return positions


def reset_defaults() -> None:
    """Restore config/BaseOptimizer globals to their original values."""
    apply_scenario(
        num_robots=DEFAULTS["R"],
        path_length=DEFAULTS["PATH_LENGTH"],
        alpha=DEFAULTS["ALPHA"],
        beta=DEFAULTS["BETA"],
        gamma=DEFAULTS["GAMMA"],
        energy_budget=DEFAULTS["ENERGY_BUDGET"],
        communication_radius=DEFAULTS["COMMUNICATION_RADIUS"],
        connectivity_threshold=DEFAULTS["CONNECTIVITY_THRESHOLD"],
        robot_positions=DEFAULTS["ROBOT_INITIAL_POSITIONS"],
    )


def build_initial_movements(path_length: int, positions: Sequence[Position]) -> np.ndarray:
    """Construct a feasible initial solution and return its movement encoding."""
    initial_path = bo.BaseOptimizer.create_dummy_solution(
        initial_positions=positions,
        path_length=path_length,
    )
    return bo.BaseOptimizer.positions_to_movements(initial_path, positions)


def coverage_percentage(path_array: Sequence[Sequence[Position]]) -> float:
    """Compute percent of grid cells visited at least once across all robots."""
    visited = set()
    for robot_path in path_array:
        for x, y in robot_path:
            if config.MAP[x, y] != config.CELL_OBSTACLE:
                visited.add((x, y))
    total_cells = config.MAP_HEIGHT * config.MAP_WIDTH
    return (len(visited) / total_cells) * 100.0


@dataclass
class ExperimentResult:
    name: str
    cost: float
    coverage: float
    iterations: int
    extra: Dict[str, object]

    def as_row(self) -> str:
        """Return a compact string representation for console output."""
        extra_bits = ", ".join(f"{k}={v}" for k, v in self.extra.items())
        return f"{self.name:25s} | cost={self.cost:9.4f} | cover={self.coverage:6.2f}% | iters={self.iterations:4d} | {extra_bits}"
