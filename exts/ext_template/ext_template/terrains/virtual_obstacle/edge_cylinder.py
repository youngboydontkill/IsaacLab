from __future__ import annotations

import math
import numpy as np
import os
import random
import torch
import trimesh
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from numpy.linalg import norm
from typing import TYPE_CHECKING

import cv2
from pxr import UsdGeom, UsdPhysics
from sklearn.cluster import DBSCAN

import isaaclab.utils.math as math_utils
from isaaclab.markers import VisualizationMarkers
from isaaclab.sensors import patterns
from isaaclab.utils.warp import convert_to_warp_mesh, raycast_mesh

from ext_template.utils.warp.cylinder import CylinderSpatialGrid

from .virtual_obstacle_base import VirtualObstacleBase

if TYPE_CHECKING:
    from .edge_cylinder_cfg import (
        EdgeCylinderCfg,
        GreedyconcatEdgeCylinderCfg,
        PluckerEdgeCylinderCfg,
        RansacEdgeCylinderCfg,
        RayEdgeCylinderCfg,
    )


class EdgeCylinder(VirtualObstacleBase):
    """Base class for edge detectors."""

    def __init__(self, cfg: EdgeCylinderCfg):
        self.cfg: EdgeCylinderCfg = cfg
        self.angle_threshold = cfg.angle_threshold

    def generate(self, mesh: trimesh.Trimesh, device="cpu") -> None:
        angles = mesh.face_adjacency_angles
        threshold = np.deg2rad(self.angle_threshold)
        sharp_mask = angles > threshold
        if not np.any(sharp_mask):
            edge_end_points = np.empty((0, 6), dtype=np.float32)
            print("[WARNING] No sharp edges detected.")
        else:
            sharp_edges = mesh.face_adjacency_edges[sharp_mask]
            v = mesh.vertices
            edge_coords = np.hstack([v[sharp_edges[:, 0]], v[sharp_edges[:, 1]]])
            edge_end_points = self.process_edges(edge_coords)
            print(f"Detected {edge_end_points.shape[0]} edges after processing.")
        self.device = device if isinstance(device, torch.device) else torch.device(device)
        self.edges_pyt = torch.tensor(edge_end_points, dtype=torch.float32, device=self.device)
        if edge_end_points.size > 0:
            self.cylinders = CylinderSpatialGrid(
                cylinders=np.concatenate(
                    [
                        edge_end_points,
                        np.ones_like(edge_end_points[:, :1]) * self.cfg.cylinder_radius,
                    ],
                    axis=1,
                ),
                num_grid_cells=self.cfg.num_grid_cells,
                device=self.device,
            )
        else:
            self.cylinders = None

    def disable_visualizer(self):
        if hasattr(self, "_cylinder_visualizer"):
            self._cylinder_visualizer.set_visibility(False)

    def visualize(self):
        if self.edges_pyt.numel() == 0:
            return

        if not hasattr(self, "_cylinder_visualizer"):
            self._cylinder_visualizer = VisualizationMarkers(self.cfg.visualizer)
            self._cylinder_rotate_y_90 = math_utils.quat_from_angle_axis(
                angle=torch.tensor([np.pi / 2], device=self.device),
                axis=torch.tensor([[0.0, 1.0, 0.0]], device=self.device),
            )

        trans = (self.edges_pyt[:, :3] + self.edges_pyt[:, 3:6]) / 2
        direction = self.edges_pyt[:, 3:6] - self.edges_pyt[:, :3]
        default_direction = torch.zeros_like(direction)
        default_direction[:, 0] = 1.0
        normalized_direction = direction / torch.norm(direction, dim=-1, keepdim=True)
        axis = torch.cross(default_direction, normalized_direction, dim=-1)
        dot_prod_ = torch.sum(default_direction * normalized_direction, dim=-1)
        angle = torch.acos(torch.clamp(dot_prod_, -1.0, 1.0))
        quat = math_utils.quat_from_angle_axis(
            angle,
            axis,
        )
        quat = math_utils.quat_mul(quat, self._cylinder_rotate_y_90.expand(quat.shape[0], -1))
        scales = torch.ones(len(self.edges_pyt), 3, device=self.device)
        scales[:, 0] = self.cfg.cylinder_radius
        scales[:, 1] = self.cfg.cylinder_radius
        scales[:, 2] = torch.norm(direction, dim=-1)
        self._cylinder_visualizer.visualize(
            translations=trans,
            orientations=quat,
            scales=scales,
        )
        self._cylinder_visualizer.set_visibility(True)

    def get_points_penetration_offset(self, points):
        return (
            self.cylinders.get_points_penetration_offset(points)
            if self.cylinders is not None
            else torch.zeros_like(points, device=self.device)
        )

    def process_edges(self, edge_coords: np.ndarray) -> np.ndarray:
        return edge_coords


class PluckerEdgeCylinder(EdgeCylinder):
    def __init__(self, cfg: PluckerEdgeCylinderCfg):
        self.cfg: PluckerEdgeCylinderCfg = cfg
        self.angle_threshold = cfg.angle_threshold

    def process_edges(self, edge_coords: np.ndarray) -> np.ndarray:
        p0 = edge_coords[:, :3]
        p1 = edge_coords[:, 3:]

        d = p1 - p0
        lengths = np.linalg.norm(d, axis=1, keepdims=True)
        d_norm = d / lengths

        for i, vec in enumerate(d_norm):
            for comp in vec:
                if abs(comp) < 1e-8:
                    continue
                if comp < 0:
                    d_norm[i] = -vec
                break

        m = np.cross(d_norm, p0)

        para = np.hstack((d_norm, m))
        para = np.round(para, 6)

        unique_rows, inv_idx = np.unique(para, axis=0, return_inverse=True)
        groups = {}
        for i, g in enumerate(inv_idx):
            groups.setdefault(g, []).append(i)
        same_para_groups = [idx_list for idx_list in groups.values()]
        new_edge_coords = []

        for group in same_para_groups:
            d_norm_group = d_norm[group[0]]
            p0_group = p0[group[0]]
            t_group = np.zeros((len(group), 2))
            t_group[:, 0] = np.dot(p0[group, :] - p0_group, d_norm_group)
            t_group[:, 1] = np.dot(p1[group, :] - p0_group, d_norm_group)
            events = []
            for t in t_group:
                if t[0] < t[1]:
                    events.append((t[0], 1))
                    events.append((t[1], -1))
                else:
                    events.append((t[1], 1))
                    events.append((t[0], -1))
            events.sort(key=lambda x: (x[0], -x[1]))
            count = 0
            prev = None
            raw_results = []

            for point, delta in events:
                if prev is not None and point > prev:
                    if count == 1:
                        raw_results.append([prev, point])
                count += delta
                prev = point

            if not raw_results:
                continue
            else:
                merged = [raw_results[0]]
                for curr in raw_results[1:]:
                    last = merged[-1]
                    if abs(last[1] - curr[0]) < 1e-6:
                        last[1] = curr[1]
                    else:
                        merged.append(curr)
                for segment in merged:
                    pa = p0_group + segment[0] * d_norm_group
                    pb = p0_group + segment[1] * d_norm_group
                    new_edge_coords.append(np.array([pa[0], pa[1], pa[2], pb[0], pb[1], pb[2]]))

        new_edge_coords = np.array(new_edge_coords, dtype=np.float32)
        return new_edge_coords


class RansacEdgeCylinder(EdgeCylinder):
    def __init__(self, cfg: RansacEdgeCylinderCfg):
        self.cfg: RansacEdgeCylinderCfg = cfg
        self.angle_threshold = cfg.angle_threshold

    def process_edges(self, edge_coords: np.ndarray) -> np.ndarray:
        endpoints = np.vstack((edge_coords[:, :3], edge_coords[:, 3:]))
        unique_endpoints = np.unique(endpoints, axis=0)
        max_iters = self.cfg.max_iter
        thresh = self.cfg.point_distance_threshold

        db = DBSCAN(eps=self.cfg.cluster_eps, min_samples=2, metric="euclidean")
        labels = db.fit_predict(unique_endpoints)

        groups = [unique_endpoints[labels == lbl] for lbl in np.unique(labels)]
        all_segments = []

        cfg_dict = dict(
            max_iter=self.cfg.max_iter,
            point_distance_threshold=self.cfg.point_distance_threshold,
            min_points=self.cfg.min_points,
        )
        with ProcessPoolExecutor(max_workers=max(1, os.cpu_count() - 2)) as executor:
            futures = {executor.submit(_fit_segments_for_group, grp, cfg_dict): i for i, grp in enumerate(groups)}
            for future in as_completed(futures):
                segs = future.result()
                if segs:
                    all_segments.extend(segs)

        return np.array(all_segments)


class GreedyconcatEdgeCylinder(EdgeCylinder):
    def __init__(self, cfg: GreedyconcatEdgeCylinderCfg):
        self.cfg: GreedyconcatEdgeCylinderCfg = cfg
        self.angle_threshold = cfg.angle_threshold

    def process_edges(self, edge_coords: np.ndarray) -> np.ndarray:
        line_pts = edge_coords.reshape(-1, 3)
        V, inv_idx = np.unique(line_pts, axis=0, return_inverse=True)
        E_pairs = inv_idx.reshape(-1, 2)

        adj_list = {i: set() for i in range(V.shape[0])}
        for u, v in E_pairs:
            if u != v:
                adj_list[u].add(v)
                adj_list[v].add(u)

        num_edges_V = np.array([len(adj_list[i]) for i in range(V.shape[0])], dtype=int)
        available_edges = set(np.where(num_edges_V > 0)[0])

        cos_threshold = np.cos(np.deg2rad(self.cfg.adjacent_angle_threshold))

        processed_edge_coords = []

        def compute_max_distance_to_line_vec(V, v_set):
            A = V[v_set[0]]
            B = V[v_set[-1]]
            AB = B - A
            norm_AB = np.linalg.norm(AB)
            if norm_AB == 0:
                return v_set[0], 0.0
            pts = V[v_set]
            dists = np.linalg.norm(np.cross(pts - A, pts - B), axis=1) / norm_AB
            max_idx = np.argmax(dists)
            return v_set[max_idx], dists[max_idx]

        while available_edges:
            selected_vertex = random.choice(list(available_edges))
            v_set = [selected_vertex]

            if adj_list[selected_vertex]:
                neighbor = next(iter(adj_list[selected_vertex]))
                v_set.append(neighbor)
                adj_list[selected_vertex].remove(neighbor)
                adj_list[neighbor].remove(selected_vertex)
                for vid in [selected_vertex, neighbor]:
                    num_edges_V[vid] -= 1
                    if num_edges_V[vid] == 0:
                        available_edges.discard(vid)

            while True:
                find_neighbor = False
                start, end = v_set[0], v_set[-1]

                neighbors = list(adj_list[start])
                if neighbors:
                    dirs = V[start] - V[neighbors]
                    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
                    start_dir = V[v_set[1]] - V[start]
                    start_dir /= np.linalg.norm(start_dir)
                    dots = dirs @ start_dir
                    idx = np.where(dots > cos_threshold)[0]
                    if idx.size > 0:
                        n = neighbors[idx[0]]
                        v_set.insert(0, n)
                        adj_list[start].remove(n)
                        adj_list[n].remove(start)
                        for vid in [start, n]:
                            num_edges_V[vid] -= 1
                            if num_edges_V[vid] == 0:
                                available_edges.discard(vid)
                        find_neighbor = True

                neighbors = list(adj_list[end])
                if neighbors:
                    dirs = V[neighbors] - V[end]
                    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
                    end_dir = V[end] - V[v_set[-2]]
                    end_dir /= np.linalg.norm(end_dir)
                    dots = dirs @ end_dir
                    idx = np.where(dots > cos_threshold)[0]
                    if idx.size > 0:
                        n = neighbors[idx[0]]
                        v_set.append(n)
                        adj_list[end].remove(n)
                        adj_list[n].remove(end)
                        for vid in [end, n]:
                            num_edges_V[vid] -= 1
                            if num_edges_V[vid] == 0:
                                available_edges.discard(vid)
                        find_neighbor = True

                if not find_neighbor:
                    break

            while len(v_set) >= self.cfg.min_points:
                for i in range(len(v_set) - 1):
                    max_vi, max_dist = compute_max_distance_to_line_vec(V, v_set[i:])
                    if max_dist < 0.05:
                        break
                if len(v_set) - i >= self.cfg.min_points:
                    processed_edge_coords.append(np.concatenate([V[v_set[i]], V[v_set[-1]]]))
                v_set = v_set[: i + 1]

        return np.array(processed_edge_coords, dtype=np.float32)
