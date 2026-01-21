import numpy as np
import torch
from collections import defaultdict

import warp as wp

from .kernels import points_penetrate_cylinder_kernel


class CylinderSpatialGrid:
    """Handle cylinders and sort them into a spatial grid for penetration queries."""

    def __init__(
        self,
        cylinders: torch.Tensor | np.ndarray,
        num_grid_cells: int = 64**3,
        device: str | torch.device = "cuda",
    ):
        self.cylinders_np = cylinders if isinstance(cylinders, np.ndarray) else cylinders.cpu().numpy()
        assert self.cylinders_np.shape[1] == 7
        self.num_grid_cells = num_grid_cells
        self.device = device

        self.num_cylinders = self.cylinders_np.shape[0]

        self._compute_bounding_box()
        self._create_grid()

    def _compute_bounding_box(self):
        xyz = np.concatenate([self.cylinders_np[:, :3], self.cylinders_np[:, 3:6]], axis=0)
        r_max = self.cylinders_np[:, 6].max()
        bbox_min = xyz.min(axis=0) - r_max
        bbox_max = xyz.max(axis=0) + r_max
        extent = bbox_max - bbox_min

        scale = extent / extent.max()
        res_f = scale * (self.num_grid_cells ** (1 / 3))
        grid_res = np.maximum(np.round(res_f), 1).astype(int)

        self.grid_res = grid_res
        self.total_num_cells = np.prod(grid_res)
        self.cell_size = extent / grid_res
        self.bbox_min = bbox_min
        self.bbox_max = bbox_max

    def get_flat_grid_idx(self, ix, iy, iz):
        if ix < 0 or ix >= self.grid_res[0] or iy < 0 or iy >= self.grid_res[1] or iz < 0 or iz >= self.grid_res[2]:
            return -1
        if self.grid_res[0] <= 0 or self.grid_res[1] <= 0 or self.grid_res[2] <= 0:
            return -1
        if self.grid_res[0] * self.grid_res[1] * self.grid_res[2] <= 0:
            return -1
        return ix * self.grid_res[1] * self.grid_res[2] + iy * self.grid_res[2] + iz

    def _create_grid(self):
        grid = defaultdict(list)
        for idx, cyl in enumerate(self.cylinders_np):
            r = cyl[6]
            start = cyl[:3]
            end = cyl[3:6]
            bbox_min = np.minimum(start, end) - r
            bbox_max = np.maximum(start, end) + r
            min_cell = np.floor((bbox_min - self.bbox_min) / self.cell_size).astype(int)
            max_cell = np.floor((bbox_max - self.bbox_min) / self.cell_size).astype(int)

            for ix in range(min_cell[0], max_cell[0] + 1):
                for iy in range(min_cell[1], max_cell[1] + 1):
                    for iz in range(min_cell[2], max_cell[2] + 1):
                        flat_grid_idx = self.get_flat_grid_idx(ix, iy, iz)
                        if flat_grid_idx < 0 or flat_grid_idx >= self.total_num_cells:
                            continue
                        grid[flat_grid_idx].append(idx)

        self.cell_offsets = np.zeros(self.total_num_cells + 1, dtype=np.int32)
        self.cell_indices = []
        for i in range(self.total_num_cells):
            self.cell_offsets[i] = len(self.cell_indices)
            self.cell_indices.extend(grid[i])
        self.cell_offsets[-1] = len(self.cell_indices)
        self.cell_indices = np.array(self.cell_indices, dtype=np.int32)

        self.cell_offsets_wp = wp.array(self.cell_offsets, dtype=wp.int32)
        self.cell_indices_wp = wp.array(self.cell_indices, dtype=wp.int32)
        self.cell_size_wp = wp.vec3(*self.cell_size)
        self.bbox_min_wp = wp.vec3(*self.bbox_min)
        self.grid_res_wp = wp.vec3i(*self.grid_res)
        self.cylinder_start_wp = wp.array(self.cylinders_np[:, :3], dtype=wp.vec3)
        self.cylinder_end_wp = wp.array(self.cylinders_np[:, 3:6], dtype=wp.vec3)
        self.cylinder_thickness_wp = wp.array(self.cylinders_np[:, 6], dtype=wp.float32)

    def get_points_penetration_offset(self, points: torch.Tensor) -> torch.Tensor:
        assert points.shape[1] == 3
        points_wp = wp.from_torch(points, dtype=wp.vec3)
        penetration_offset = torch.zeros(points.shape[0], 3, device=points.device, dtype=points.dtype)
        penetration_offset_wp = wp.from_torch(penetration_offset, dtype=wp.vec3)
        output_device = points.device

        wp.launch(
            points_penetrate_cylinder_kernel,
            dim=points.shape[0],
            inputs=[
                points_wp,
                self.cylinder_start_wp,
                self.cylinder_end_wp,
                self.cylinder_thickness_wp,
                self.cell_offsets_wp,
                self.cell_indices_wp,
                self.grid_res_wp,
                self.bbox_min_wp,
                self.cell_size_wp,
                penetration_offset_wp,
            ],
            device=str(points.device),
        )

        return penetration_offset.to(output_device)
