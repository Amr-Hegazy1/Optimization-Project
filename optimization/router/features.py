from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

import config

Position = Tuple[int, int]


@dataclass(frozen=True)
class FeatureSpec:
    names: List[str]


DEFAULT_FEATURE_SPEC = FeatureSpec(
    names=[
        "n_robots",
        "path_length",
        "alpha",
        "beta",
        "gamma",
        "energy_budget",
        "communication_radius",
        "connectivity_threshold",
        "map_height",
        "map_width",
        "obstacle_frac",
        "unexplored_frac",
        "start_mean_pairwise_manhattan",
        "start_min_pairwise_manhattan",
        "start_mean_pairwise_euclidean",
        "start_min_pairwise_euclidean",
        "start_connectivity_edge_frac",
    ]
)


def _pairwise_distances(positions: Sequence[Position]) -> Tuple[np.ndarray, np.ndarray]:
    if len(positions) < 2:
        return np.array([], dtype=float), np.array([], dtype=float)
    manhattan: List[float] = []
    euclidean: List[float] = []
    for i in range(len(positions)):
        x1, y1 = positions[i]
        for j in range(i + 1, len(positions)):
            x2, y2 = positions[j]
            manhattan.append(abs(x1 - x2) + abs(y1 - y2))
            euclidean.append(float(((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5))
    return np.array(manhattan, dtype=float), np.array(euclidean, dtype=float)


def _edge_fraction_under_threshold(positions: Sequence[Position], threshold: float) -> float:
    if len(positions) < 2:
        return 0.0
    edges = 0
    total = 0
    thr2 = float(threshold) ** 2
    for i in range(len(positions)):
        x1, y1 = positions[i]
        for j in range(i + 1, len(positions)):
            x2, y2 = positions[j]
            total += 1
            if (x1 - x2) ** 2 + (y1 - y2) ** 2 <= thr2:
                edges += 1
    return edges / max(1, total)


def extract_features(
    *,
    map_grid: np.ndarray,
    robot_positions: Sequence[Position],
    n_robots: int,
    path_length: int,
    alpha: float,
    beta: float,
    gamma: float,
    energy_budget: float,
    communication_radius: float,
    connectivity_threshold: float,
    feature_spec: FeatureSpec = DEFAULT_FEATURE_SPEC,
) -> Dict[str, float]:
    map_height, map_width = map_grid.shape
    obstacle_frac = float(np.mean(map_grid == getattr(config, "CELL_OBSTACLE", 2)))
    unexplored_frac = float(np.mean(map_grid == getattr(config, "CELL_UNEXPLORED", 0)))

    manhattan, euclidean = _pairwise_distances(robot_positions)
    start_mean_manhattan = float(manhattan.mean()) if manhattan.size else 0.0
    start_min_manhattan = float(manhattan.min()) if manhattan.size else 0.0
    start_mean_euclidean = float(euclidean.mean()) if euclidean.size else 0.0
    start_min_euclidean = float(euclidean.min()) if euclidean.size else 0.0
    edge_frac = _edge_fraction_under_threshold(robot_positions, connectivity_threshold)

    values: Dict[str, float] = {
        "n_robots": float(n_robots),
        "path_length": float(path_length),
        "alpha": float(alpha),
        "beta": float(beta),
        "gamma": float(gamma),
        "energy_budget": float(energy_budget),
        "communication_radius": float(communication_radius),
        "connectivity_threshold": float(connectivity_threshold),
        "map_height": float(map_height),
        "map_width": float(map_width),
        "obstacle_frac": obstacle_frac,
        "unexplored_frac": unexplored_frac,
        "start_mean_pairwise_manhattan": start_mean_manhattan,
        "start_min_pairwise_manhattan": start_min_manhattan,
        "start_mean_pairwise_euclidean": start_mean_euclidean,
        "start_min_pairwise_euclidean": start_min_euclidean,
        "start_connectivity_edge_frac": float(edge_frac),
    }

    # Ensure all requested features exist.
    for name in feature_spec.names:
        values.setdefault(name, 0.0)
    return values


def vectorize_features(feature_dict: Dict[str, float], feature_spec: FeatureSpec = DEFAULT_FEATURE_SPEC) -> np.ndarray:
    return np.array([feature_dict[name] for name in feature_spec.names], dtype=float)


def extract_features_from_config(feature_spec: FeatureSpec = DEFAULT_FEATURE_SPEC) -> Dict[str, float]:
    return extract_features(
        map_grid=np.array(config.MAP),
        robot_positions=tuple(config.ROBOT_INITIAL_POSITIONS),
        n_robots=int(config.R),
        path_length=int(config.PATH_LENGTH),
        alpha=float(config.ALPHA),
        beta=float(config.BETA),
        gamma=float(config.GAMMA),
        energy_budget=float(config.ENERGY_BUDGET),
        communication_radius=float(config.COMMUNICATION_RADIUS),
        connectivity_threshold=float(config.CONNECTIVITY_THRESHOLD),
        feature_spec=feature_spec,
    )
