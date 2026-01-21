from __future__ import annotations

import math
from dataclasses import MISSING
from typing import TYPE_CHECKING, Literal

import isaaclab.sim as sim_utils
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.sensors import patterns
from isaaclab.utils import configclass

from .edge_cylinder import GreedyconcatEdgeCylinder, PluckerEdgeCylinder, RansacEdgeCylinder, RayEdgeCylinder
from .virtual_obstacle_base import VirtualObstacleCfg


@configclass
class EdgeCylinderCfg(VirtualObstacleCfg):
    """The class to use for the edge cylinder detector."""

    class_type: type = MISSING
    """The class to use for the edge detector."""
    angle_threshold: float = 70.0
    """The angle threshold to consider an edge as sharp."""

    cylinder_radius: float = 0.2
    """The radius of the edge cylinder, which is used to treat the edge cylinders as a virtual obstacle."""
    num_grid_cells: int = 64**3
    """The number of grid cells to use for spatial partitioning of the edge cylinders."""

    visualizer: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/edgeMarkers",
        markers={
            "cylinder": sim_utils.CylinderCfg(
                radius=1,
                height=1,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 0.9), opacity=0.2),
            )
        },
    )


@configclass
class PluckerEdgeCylinderCfg(EdgeCylinderCfg):
    class_type: type = PluckerEdgeCylinder


@configclass
class RansacEdgeCylinderCfg(EdgeCylinderCfg):
    class_type: type = RansacEdgeCylinder

    max_iter: int = 500
    point_distance_threshold: float = 0.04
    min_points: int = 5
    cluster_eps: float = 0.08


@configclass
class GreedyconcatEdgeCylinderCfg(EdgeCylinderCfg):
    class_type: type = GreedyconcatEdgeCylinder

    adjacent_angle_threshold: float = 30.0
    point_distance_threshold: float = 0.06
    min_points: int = 5


@configclass
class RayEdgeCylinderCfg(VirtualObstacleCfg):
    class_type: type = RayEdgeCylinder

    cylinder_radius: float = 0.2
    num_grid_cells: int = 64**3
    max_iter: int = 500
    point_distance_threshold: float = 0.005
    min_points: int = 15
    cluster_eps: float = 0.08

    ray_pattern: patterns.GridPatternCfg = patterns.GridPatternCfg(
        resolution=0.01,
        size=[6, 6],
        direction=(0.0, 0.0, -1.0),
    )

    ray_offset_pos: list[float] = [0.0, 0.0, 1.0]

    ray_rotate_axes: list[list[float]] = [
        [1.0, 1.0, 0.0],
        [-1.0, 1.0, 0.0],
        [1.0, -1.0, 0.0],
        [-1.0, -1.0, 0.0],
    ]

    ray_rotate_angle: list[float] = [math.pi * 0.25, math.pi * 0.25, math.pi * 0.25, math.pi * 0.25]

    max_ray_depth: float = 8.0

    depth_canny_thresholds: list[float] = [250, 300]

    normal_canny_thresholds: list[float] = [80, 250]

    cutoff_z_height: float = 0.1

    visualizer: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/edgeMarkers",
        markers={
            "cylinder": sim_utils.CylinderCfg(
                radius=1,
                height=1,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 0.9), opacity=0.2),
            )
        },
    )

    points_visualizer: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/edgePoints",
        markers={
            "sphere": sim_utils.SphereCfg(
                radius=0.01,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.5, 0.5)),
            ),
        },
    )
