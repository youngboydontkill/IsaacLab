from __future__ import annotations

import torch
import trimesh
from abc import ABC, abstractmethod
from dataclasses import MISSING
from typing import TYPE_CHECKING

from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.utils import configclass


@configclass
class VirtualObstacleCfg:
    """Configuration for a virtual obstacle."""

    class_type: type = MISSING
    """The class to use for the virtual obstacle."""

    visualizer: VisualizationMarkersCfg = MISSING
    """The visualizer configuration for the virtual obstacle."""


class VirtualObstacleBase(ABC):
    def __init__(self, cfg: VirtualObstacleCfg):
        self.cfg = cfg

    @abstractmethod
    def generate(self, mesh: trimesh.Trimesh, device: torch.device | str = "cpu") -> None:
        raise NotImplementedError("This method should be implemented by subclasses.")

    @abstractmethod
    def disable_visualizer(self) -> None:
        raise NotImplementedError("This method should be implemented by subclasses.")

    @abstractmethod
    def visualize(self):
        raise NotImplementedError("This method should be implemented by subclasses.")

    @abstractmethod
    def get_points_penetration_offset(self, points: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("This method should be implemented by subclasses.")
