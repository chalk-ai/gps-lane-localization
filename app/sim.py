"""Simulates a car driving a real street grid, and the sensors a production car would carry.

Ground truth comes from a route through the OSM road graph, a lane plan (curb lane before right
turns, the far lane before left turns, occasional lane changes), and a curvature-limited speed
profile with stops. The sensors are:

- GPS at 1 Hz: white noise, a slowly wandering bias, multipath jumps, and no fix (or a wild one)
  in the shadow of the I-80 overpass.
- Wheel odometry and a gyro at 10 Hz, with a speed scale error and a gyro bias.
- A lane camera at 10 Hz: the car's offset from the centre of whichever lane it is in, and whether
  the markings on each side are solid (road edge) or dashed (another lane).
- A landmark detector at 2 Hz: range and bearing to poles and signs at the roadside, with known
  identities. True landmark positions are never given to an estimator; the live drive gets
  only the landmark map a mapping fleet built with SLAM.
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass

import numpy as np

from roadmap import LANE_WIDTH_M, Edge, RoadMap, point_to_segment

DT = 0.1
GPS_EVERY = 10  # samples between GPS fixes (1 Hz)
KF_EVERY = 5  # samples between landmark observations / SLAM keyframes (2 Hz)


@dataclass(frozen=True)
class GpsProfile:
    name: str
    white_sigma: float
    bias_sigma: float
    bias_tau_s: float
    multipath_rate_hz: float
    multipath_magnitude: tuple[float, float]


GPS_PROFILES = {
    "open-sky": GpsProfile("open-sky", 2.0, 1.0, 60.0, 0.0, (0.0, 0.0)),
    "urban-canyon": GpsProfile("urban-canyon", 3.0, 4.0, 25.0, 1 / 40, (8.0, 18.0)),
    "harsh": GpsProfile("harsh", 5.0, 7.0, 20.0, 1 / 20, (10.0, 30.0)),
}

# How the sensors misbehave, shared by the simulator and (as noise models) by the estimators.
SPEED_SCALE_ERROR = 0.015
SPEED_NOISE = 0.05
GYRO_NOISE = math.radians(0.3)
LANE_CAM_NOISE = 0.10
LANDMARK_RANGE_NOISE = 0.3
LANDMARK_BEARING_NOISE = math.radians(1.5)
LANDMARK_MAX_RANGE = 30.0
OVERPASS_SHADOW_M = 14.0


@dataclass
class Drive:
    t: np.ndarray
    # Ground truth at 10 Hz.
    x: np.ndarray
    y: np.ndarray
    heading: np.ndarray
    speed: np.ndarray
    edge: np.ndarray
    lane: np.ndarray
    route: list[int]
    # Sensors.
    gps_idx: np.ndarray
    gps: np.ndarray  # (n, 2); NaN where there was no fix
    gps_shadow: np.ndarray  # bool per fix: inside the overpass shadow
    odo_speed: np.ndarray
    odo_yaw_rate: np.ndarray
    cam_offset: np.ndarray  # NaN when the camera lost the markings
    cam_left_solid: np.ndarray
    cam_right_solid: np.ndarray
    landmarks: np.ndarray  # (m, 2) true positions, for display and scoring only
    obs: list[tuple[int, int, float, float]]  # (sample index, landmark id, range, bearing)


def _turn(e1: Edge, e2: Edge) -> float:
    d = e2.heading - e1.heading
    return (d + math.pi) % (2 * math.pi) - math.pi


def plan_route(rm: RoadMap, start: int, goal: int) -> list[int] | None:
    """Shortest edge sequence from start to goal; turns cost extra, U-turns are not allowed."""
    dist = {start: 0.0}
    prev: dict[int, int] = {}
    heap = [(0.0, start)]
    while heap:
        d, e = heapq.heappop(heap)
        if e == goal:
            path = [e]
            while path[-1] in prev:
                path.append(prev[path[-1]])
            return path[::-1]
        if d > dist.get(e, math.inf):
            continue
        cur = rm.edges[e]
        for nxt in rm.out_edges.get(cur.v, []):
            ne = rm.edges[nxt]
            if ne.v == cur.u:
                continue
            turn = abs(_turn(cur, ne))
            cost = d + ne.length + (12.0 if turn > math.radians(40) else 0.0)
            if cost < dist.get(nxt, math.inf):
                dist[nxt], prev[nxt] = cost, e
                heapq.heappush(heap, (cost, nxt))
    return None


def make_landmarks(rm: RoadMap, seed: int) -> np.ndarray:
    """Poles and signs along both sides of every street, shared by every drive in the area."""
    rng = np.random.default_rng(seed)
    marks = []
    seen_ways: set[tuple[int, int, int]] = set()
    for e in rm.edges:
        # Both directions of a two-way street share one set of roadside landmarks.
        key = (e.way_id, min(e.u, e.v), max(e.u, e.v))
        if key in seen_ways:
            continue
        seen_ways.add(key)
        d = np.array(e.b) - np.array(e.a)
        u = d / max(np.linalg.norm(d), 1e-9)
        r = np.array([u[1], -u[0]])
        pos = rng.uniform(0, 18)
        while pos < e.length:
            side = e.right_edge + rng.uniform(1.0, 3.0) if rng.random() < 0.6 else e.left_edge - rng.uniform(1.0, 3.0)
            marks.append(np.array(e.a) + u * pos + r * side)
            pos += rng.uniform(14, 24)
    return np.array(marks)


def _dist_to_freeway(rm: RoadMap, pts: np.ndarray) -> np.ndarray:
    best = np.full(len(pts), np.inf)
    for line in rm.freeway:
        for a, b in zip(line[:-1], line[1:]):
            ab = b - a
            t = np.clip(((pts - a) @ ab) / max(ab @ ab, 1e-9), 0, 1)
            d = np.linalg.norm(pts - (a + t[:, None] * ab), axis=1)
            best = np.minimum(best, d)
    return best


def pick_route(rm: RoadMap, rng: np.random.Generator, under_freeway: bool) -> list[int]:
    """A 1.6 to 3 km route. With under_freeway, one that passes under I-80 so GPS has to fail."""
    candidates = [e.id for e in rm.edges if e.lanes >= 2]
    best: list[int] | None = None
    for _ in range(300):
        start, goal = rng.choice(candidates, 2, replace=False)
        route = plan_route(rm, int(start), int(goal))
        if route is None:
            continue
        length = sum(rm.edges[e].length for e in route)
        if not 1600 <= length <= 3000:
            continue
        if not under_freeway:
            return route
        pts = np.array([rm.edges[e].a for e in route])
        if _dist_to_freeway(rm, pts).min() < OVERPASS_SHADOW_M:
            return route
        best = best or route
    if best is None:
        raise RuntimeError("no route found")
    return best


def _moving_average(a: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return a
    pad = window // 2
    padded = np.concatenate([np.repeat(a[:1], pad, axis=0), a, np.repeat(a[-1:], pad, axis=0)])
    kernel = np.ones(window) / window
    if a.ndim == 1:
        return np.convolve(padded, kernel, mode="valid")[: len(a)]
    return np.stack([np.convolve(padded[:, i], kernel, mode="valid")[: len(a)] for i in range(a.shape[1])], 1)


def simulate(rm: RoadMap, landmarks: np.ndarray, seed: int, profile: GpsProfile, under_freeway: bool) -> Drive:
    rng = np.random.default_rng(seed)
    route = pick_route(rm, rng, under_freeway)
    edges = [rm.edges[e] for e in route]

    # Lane plan on a 1 m grid along the route.
    lengths = np.array([e.length for e in edges])
    starts = np.concatenate([[0.0], np.cumsum(lengths)[:-1]])
    total = float(lengths.sum())
    s = np.arange(0.0, total, 1.0)
    edge_at = np.searchsorted(starts, s, side="right") - 1
    lane_at = np.zeros(len(s), dtype=int)
    lane = int(rng.integers(0, edges[0].lanes))
    for i, e in enumerate(edges):
        lane = min(lane, e.lanes - 1)
        if e.lanes > 1 and rng.random() < e.length / 220:
            lane = int(np.clip(lane + rng.choice([-1, 1]), 0, e.lanes - 1))
        mask = edge_at == i
        lane_at[mask] = lane
        # Before a turn, move to the curb lane (right turn) or the far lane (left turn).
        if i + 1 < len(edges):
            turn = _turn(e, edges[i + 1])
            if abs(turn) > math.radians(40):
                lane = 0 if turn < 0 else e.lanes - 1
                near_end = mask & (s > starts[i] + e.length - 60)
                lane_at[near_end] = lane
    target_offset = np.array([edges[edge_at[i]].lane_offset(lane_at[i]) for i in range(len(s))])
    offset = _moving_average(target_offset, 31)

    # Round the corners of the centreline, then place the car at its lateral offset.
    centre = np.array(
        [
            np.array(edges[edge_at[i]].a)
            + (s[i] - starts[edge_at[i]])
            / lengths[edge_at[i]]
            * (np.array(edges[edge_at[i]].b) - np.array(edges[edge_at[i]].a))
            for i in range(len(s))
        ]
    )
    centre = _moving_average(centre, 9)
    tangent = np.gradient(centre, axis=0)
    tangent /= np.maximum(np.linalg.norm(tangent, axis=1, keepdims=True), 1e-9)
    right = np.stack([tangent[:, 1], -tangent[:, 0]], axis=1)
    path = centre + offset[:, None] * right

    # Speed: limited by lateral acceleration in bends, with acceleration limits and two stops.
    heading_s = np.unwrap(np.arctan2(np.gradient(path[:, 1]), np.gradient(path[:, 0])))
    curvature = np.abs(_moving_average(np.gradient(heading_s), 5))
    v = np.minimum(rng.uniform(10.5, 13.0), np.sqrt(2.0 / np.maximum(curvature, 1e-4)))
    stops = sorted(rng.choice(np.arange(200, max(201, len(s) - 200)), 2, replace=False))
    for st in stops:
        v[st] = 0.0
    v[0] = v[-1] = 0.0
    for i in range(1, len(v)):
        v[i] = min(v[i], math.sqrt(v[i - 1] ** 2 + 2 * 1.8))
    for i in range(len(v) - 2, -1, -1):
        v[i] = min(v[i], math.sqrt(v[i + 1] ** 2 + 2 * 2.5))
    dt_s = 1.0 / np.maximum(0.5 * (v[:-1] + v[1:]), 0.3)
    t_grid = np.concatenate([[0.0], np.cumsum(dt_s)])
    for st in stops:
        t_grid[st:] += rng.uniform(4.0, 8.0)
    t = np.arange(0.0, t_grid[-1], DT)
    s_t = np.interp(t, t_grid, s)

    x = np.interp(s_t, s, path[:, 0])
    y = np.interp(s_t, s, path[:, 1])
    heading = np.interp(s_t, s, heading_s)
    speed = np.gradient(s_t, DT)
    yaw_rate = np.gradient(heading, DT)
    idx = np.clip(np.round(s_t).astype(int), 0, len(s) - 1)
    edge_t = np.array([route[edge_at[i]] for i in idx])
    lane_t = lane_at[idx].copy()
    n = len(t)

    # Odometry.
    gyro_bias = math.radians(rng.uniform(0.25, 0.5)) * rng.choice([-1, 1])
    odo_speed = speed * (1 + SPEED_SCALE_ERROR) + rng.normal(0, SPEED_NOISE, n)
    odo_speed = np.maximum(odo_speed, 0)
    odo_yaw_rate = yaw_rate + gyro_bias + rng.normal(0, GYRO_NOISE, n)

    # GPS: white noise + Gauss-Markov bias + multipath jumps + overpass shadow.
    truth_xy = np.stack([x, y], axis=1)
    gps_idx = np.arange(0, n, GPS_EVERY)
    shadow_d = _dist_to_freeway(rm, truth_xy[gps_idx])
    bias = rng.normal(0, profile.bias_sigma, 2)
    phi = math.exp(-1.0 / profile.bias_tau_s)
    gps = np.zeros((len(gps_idx), 2))
    multipath = np.zeros(2)
    multipath_left = 0
    shadow = shadow_d < OVERPASS_SHADOW_M
    for k in range(len(gps_idx)):
        bias = phi * bias + rng.normal(0, profile.bias_sigma * math.sqrt(1 - phi * phi), 2)
        if multipath_left <= 0 and rng.random() < profile.multipath_rate_hz:
            a = rng.uniform(0, 2 * math.pi)
            multipath = rng.uniform(*profile.multipath_magnitude) * np.array([math.cos(a), math.sin(a)])
            multipath_left = int(rng.integers(4, 11))
        jump = multipath if multipath_left > 0 else 0.0
        multipath_left -= 1
        gps[k] = truth_xy[gps_idx[k]] + bias + jump + rng.normal(0, profile.white_sigma, 2)
        if shadow[k]:
            gps[k] = np.nan if rng.random() < 0.7 else truth_xy[gps_idx[k]] + rng.normal(0, 25.0, 2)

    # Lane camera. The true lane is whichever lane centre the car is physically nearest, so a
    # sample halfway through a lane change counts against the lane it is crossing into.
    cam_offset = np.full(n, np.nan)
    cam_left = np.zeros(n, dtype=bool)
    cam_right = np.zeros(n, dtype=bool)
    for i in range(n):
        e = rm.edges[edge_t[i]]
        lateral, _, _ = point_to_segment(truth_xy[i], e.a, e.b)
        k = int(np.clip(round((e.lane0_offset - lateral) / LANE_WIDTH_M), 0, e.lanes - 1))
        lane_t[i] = k
        if rng.random() < 0.05:
            continue
        cam_offset[i] = lateral - e.lane_offset(k) + rng.normal(0, LANE_CAM_NOISE)
        cam_left[i] = k == e.lanes - 1
        cam_right[i] = k == 0

    # Landmark detections (range, bearing) of the shared roadside landmarks.
    obs: list[tuple[int, int, float, float]] = []
    for i in range(0, n, KF_EVERY):
        rel = landmarks - truth_xy[i]
        rng_m = np.linalg.norm(rel, axis=1)
        bearing = np.arctan2(rel[:, 1], rel[:, 0]) - heading[i]
        bearing = (bearing + math.pi) % (2 * math.pi) - math.pi
        for j in np.nonzero((rng_m < LANDMARK_MAX_RANGE) & (np.abs(bearing) < math.radians(80)))[0]:
            obs.append(
                (
                    i,
                    int(j),
                    float(rng_m[j] + rng.normal(0, LANDMARK_RANGE_NOISE)),
                    float(bearing[j] + rng.normal(0, LANDMARK_BEARING_NOISE)),
                )
            )

    return Drive(
        t=t,
        x=x,
        y=y,
        heading=heading,
        speed=speed,
        edge=edge_t,
        lane=lane_t,
        route=route,
        gps_idx=gps_idx,
        gps=gps,
        gps_shadow=shadow,
        odo_speed=odo_speed,
        odo_yaw_rate=odo_yaw_rate,
        cam_offset=cam_offset,
        cam_left_solid=cam_left,
        cam_right_solid=cam_right,
        landmarks=landmarks,
        obs=obs,
    )
