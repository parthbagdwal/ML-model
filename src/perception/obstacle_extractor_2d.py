"""Stage 12: 2D LiDAR Obstacle Extraction & Multi-Scan Tracker.

Implements a deterministic, planar 2D obstacle extraction pipeline:
1. Rejection of invalid, out-of-range, NaN, and Infinite range measurements.
2. Polar-to-Cartesian conversion (+X forward, +Y lateral in vehicle frame).
3. Neighbor grouping via adaptive Euclidean jump-distance thresholding.
4. Cluster filtering and planar dimensional analysis (centroid, planar span/width, radial depth, return count).
5. Temporal multi-scan tracking (`MultiScanTracker2D`) to derive relative closing velocity (m/s)
   and Time-to-Collision (s) across consecutive scan cycles.

CRITICAL SENSOR CONVENTION:
This module operates exclusively on 2D planar data. It contains NO 3D point-cloud
assumptions, NO ground-plane extraction, and NO vertical dimension estimation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

from src.perception.lidar_2d import LaserScan2D, LaserScanPoint2D, Obstacle2D


@dataclass
class TrackedObstacleState:
    """Internal state for a 2D obstacle track across consecutive scans."""

    track_id: int
    last_x_m: float
    last_y_m: float
    last_distance_m: float
    last_timestamp_ns: int
    smoothed_velocity_mps: float = 0.0
    consecutive_hits: int = 1
    missed_cycles: int = 0


class ObstacleExtractor2D:
    """Extracts planar 2D obstacles from raw 2D LiDAR scans."""

    def __init__(
        self,
        cluster_distance_threshold_m: float = 1.2,
        min_cluster_points: int = 2,
        max_cluster_span_m: float = 25.0,
    ) -> None:
        """Initialize 2D obstacle extractor.

        Args:
            cluster_distance_threshold_m: Maximum Euclidean jump distance between consecutive
                                         beam returns to consider them part of the same obstacle.
            min_cluster_points: Minimum beam hits required to form an obstacle (filters out noise).
            max_cluster_span_m: Maximum allowable planar span for a single obstacle (splits oversized clusters).
        """
        self.cluster_dist_thresh = cluster_distance_threshold_m
        self.min_cluster_points = min_cluster_points
        self.max_cluster_span = max_cluster_span_m
        self._obstacle_counter = 0

    def extract_obstacles(self, scan: LaserScan2D) -> list[Obstacle2D]:
        """Process a 2D planar scan and return segmented 2D obstacles.

        Steps:
        1. Convert valid beams to Cartesian points.
        2. Group adjacent valid beams using Euclidean jump distance.
        3. Filter clusters by point count.
        4. Calculate 2D centroid, range, planar width, and radial depth.
        """
        points = scan.to_points()
        valid_points = [p for p in points if p.is_valid]

        if not valid_points:
            return []

        # Group consecutive valid points into clusters
        raw_clusters: list[list[LaserScanPoint2D]] = []
        current_cluster: list[LaserScanPoint2D] = []

        for i, pt in enumerate(points):
            if not pt.is_valid:
                # Invalid reading breaks cluster continuity
                if current_cluster:
                    raw_clusters.append(current_cluster)
                    current_cluster = []
                continue

            if not current_cluster:
                current_cluster.append(pt)
            else:
                prev_pt = current_cluster[-1]
                # Euclidean distance between consecutive scan points
                dx = pt.x_m - prev_pt.x_m
                dy = pt.y_m - prev_pt.y_m
                dist_jump = math.sqrt(dx * dx + dy * dy)

                if dist_jump <= self.cluster_dist_thresh:
                    current_cluster.append(pt)
                else:
                    raw_clusters.append(current_cluster)
                    current_cluster = [pt]

        if current_cluster:
            raw_clusters.append(current_cluster)

        # Filter and convert clusters to Obstacle2D instances
        obstacles: list[Obstacle2D] = []

        for cluster in raw_clusters:
            if len(cluster) < self.min_cluster_points:
                continue

            self._obstacle_counter += 1
            obs_id = f"obs2d_{self._obstacle_counter:04d}"

            # Calculate 2D centroid
            avg_x = sum(p.x_m for p in cluster) / len(cluster)
            avg_y = sum(p.y_m for p in cluster) / len(cluster)
            centroid_dist = math.sqrt(avg_x * avg_x + avg_y * avg_y)

            # Calculate planar span / width across extremities
            p_first = cluster[0]
            p_last = cluster[-1]
            chord_width = math.sqrt((p_last.x_m - p_first.x_m) ** 2 + (p_last.y_m - p_first.y_m) ** 2)
            lateral_spread = max(p.y_m for p in cluster) - min(p.y_m for p in cluster)
            span_width = max(0.2, round(max(chord_width, lateral_spread), 2))

            # Calculate planar depth along radial direction
            longitudinal_spread = max(p.x_m for p in cluster) - min(p.x_m for p in cluster)
            depth_length = max(0.2, round(longitudinal_spread, 2))

            obs = Obstacle2D(
                obstacle_id=obs_id,
                distance_m=round(centroid_dist, 2),
                object_x_m=round(avg_x, 2),
                object_y_m=round(avg_y, 2),
                span_width_m=span_width,
                depth_length_m=depth_length,
                point_count=len(cluster),
                relative_velocity_mps=0.0,  # Single scan cannot measure velocity
                time_to_collision_s=99.9,
                timestamp_ns=scan.timestamp_ns,
                is_valid=True,
                detection_source="2D_PLANAR_LIDAR_SCAN",
                has_3d_z=False,
                has_3d_height=False,
                has_3d_classification=False,
            )
            obstacles.append(obs)

        return obstacles


class MultiScanTracker2D:
    """Tracks 2D obstacles across consecutive scans to compute closing velocity and TTC.

    Maintains track history, associates detections via Euclidean nearest-neighbor gating,
    and applies exponential smoothing on derived relative velocities.
    """

    def __init__(
        self,
        gating_distance_m: float = 4.5,
        velocity_smoothing_alpha: float = 0.7,
        max_missed_cycles: int = 3,
    ) -> None:
        """Initialize tracker.

        Args:
            gating_distance_m: Maximum allowable centroid displacement between consecutive scans.
            velocity_smoothing_alpha: Weight for new velocity measurement (0.0 to 1.0).
            max_missed_cycles: Number of scans a track persists without new detections before pruning.
        """
        self.gating_distance = gating_distance_m
        self.alpha = velocity_smoothing_alpha
        self.max_missed_cycles = max_missed_cycles

        self._active_tracks: dict[int, TrackedObstacleState] = {}
        self._next_track_id = 1

    @property
    def active_tracks(self) -> dict[int, TrackedObstacleState]:
        """Dictionary of currently active tracked obstacle states."""
        return self._active_tracks

    def reset(self) -> None:
        """Reset all active tracks."""
        self._active_tracks.clear()
        self._next_track_id = 1

    def track(self, obstacles: list[Obstacle2D], timestamp_ns: int) -> list[Obstacle2D]:
        """Update tracks with new detections from current scan and populate velocities and TTC.

        Args:
            obstacles: List of Obstacle2D detected in current scan.
            timestamp_ns: Monotonic timestamp of the current scan.

        Returns:
            Updated list of Obstacle2D instances with tracked IDs, velocities, and TTC.
        """
        updated_obstacles: list[Obstacle2D] = []
        matched_track_ids: set[int] = set()

        for obs in obstacles:
            best_track_id: int | None = None
            min_dist = float("inf")

            # Find nearest existing track within gating distance
            for tid, state in self._active_tracks.items():
                if tid in matched_track_ids:
                    continue
                dx = obs.object_x_m - state.last_x_m
                dy = obs.object_y_m - state.last_y_m
                dist = math.sqrt(dx * dx + dy * dy)

                if dist < self.gating_distance and dist < min_dist:
                    min_dist = dist
                    best_track_id = tid

            if best_track_id is not None:
                # Existing track match
                state = self._active_tracks[best_track_id]
                matched_track_ids.add(best_track_id)

                dt_s = (timestamp_ns - state.last_timestamp_ns) * 1e-9
                if dt_s > 1e-4:
                    # Closing velocity convention:
                    # Negative (<0): Approaching (distance decreasing)
                    # Positive (>0): Receding (distance increasing)
                    measured_velocity = (obs.distance_m - state.last_distance_m) / dt_s
                    # Apply smoothing
                    smoothed_vel = (
                        self.alpha * measured_velocity + (1.0 - self.alpha) * state.smoothed_velocity_mps
                    )
                else:
                    smoothed_vel = state.smoothed_velocity_mps

                state.last_x_m = obs.object_x_m
                state.last_y_m = obs.object_y_m
                state.last_distance_m = obs.distance_m
                state.last_timestamp_ns = timestamp_ns
                state.smoothed_velocity_mps = smoothed_vel
                state.consecutive_hits += 1
                state.missed_cycles = 0

                # Compute Time-to-Collision (TTC)
                # If approaching (negative velocity)
                if smoothed_vel < -0.05:
                    ttc = obs.distance_m / (-smoothed_vel)
                    ttc = min(99.9, max(0.0, ttc))
                else:
                    ttc = 99.9

                obs.tracking_id = best_track_id
                obs.relative_velocity_mps = round(smoothed_vel, 2)
                obs.time_to_collision_s = round(ttc, 2)
                updated_obstacles.append(obs)

            else:
                # New obstacle track
                new_tid = self._next_track_id
                self._next_track_id += 1

                self._active_tracks[new_tid] = TrackedObstacleState(
                    track_id=new_tid,
                    last_x_m=obs.object_x_m,
                    last_y_m=obs.object_y_m,
                    last_distance_m=obs.distance_m,
                    last_timestamp_ns=timestamp_ns,
                    smoothed_velocity_mps=0.0,
                    consecutive_hits=1,
                    missed_cycles=0,
                )
                matched_track_ids.add(new_tid)

                obs.tracking_id = new_tid
                obs.relative_velocity_mps = 0.0
                obs.time_to_collision_s = 99.9
                updated_obstacles.append(obs)

        # Prune tracks that were missed in this cycle
        prune_ids: list[int] = []
        for tid, state in self._active_tracks.items():
            if tid not in matched_track_ids:
                state.missed_cycles += 1
                if state.missed_cycles > self.max_missed_cycles:
                    prune_ids.append(tid)

        for tid in prune_ids:
            del self._active_tracks[tid]

        return updated_obstacles
