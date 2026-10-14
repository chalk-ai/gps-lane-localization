"""The classic localization stack, one technique at a time.

1. kalman_filter: an EKF fusing 1 Hz GPS with 10 Hz wheel speed and gyro (state x, y, heading,
   gyro bias), with Mahalanobis gating to reject multipath outliers. It knows nothing about roads.
2. map_match: hidden-Markov map matching (Newson & Krumm, 2009). States are candidate road
   segments near each 1 Hz position; emissions score distance to the road, transitions score
   how well the driving distance between candidates agrees with the straight-line distance.
   Viterbi picks the road sequence, and positions are snapped onto it.
3. graph_slam: least-squares pose-graph SLAM over 2 Hz keyframes and every landmark seen,
   constrained by odometry, GPS (with a robust loss) and range/bearing observations. Landmark
   positions are unknowns, estimated along with the trajectory.
4. lane_localize: lane-level localization on the lane map. A second HMM picks the lane from the
   lane camera's solid/dashed boundaries and lane-change events, and the camera's in-lane offset
   places the car within it.
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix, diags
from scipy.sparse.linalg import spsolve

from roadmap import LANE_WIDTH_M, Edge, RoadMap, point_to_segment
from sim import (
    DT,
    GPS_EVERY,
    GYRO_NOISE,
    KF_EVERY,
    LANDMARK_BEARING_NOISE,
    LANDMARK_RANGE_NOISE,
    SPEED_NOISE,
    Drive,
)

# The estimators assume a generic GPS spec rather than the simulator's true noise profile.
GPS_SIGMA = 5.0


def wrap(a: np.ndarray | float) -> np.ndarray | float:
    return (np.asarray(a) + np.pi) % (2 * np.pi) - np.pi


@dataclass
class Track:
    xy: np.ndarray  # (n, 2) at 10 Hz
    heading: np.ndarray
    edge: np.ndarray | None = None  # matched road edge per sample, when the stage knows roads
    lane: np.ndarray | None = None
    cov: np.ndarray | None = None  # (n, 2, 2) position covariance, for the Kalman filter


# --- 1. Kalman filter ---------------------------------------------------------------------------


def kalman_filter(d: Drive) -> Track:
    n = len(d.t)
    fixes = {int(i): k for k, i in enumerate(d.gps_idx)}
    valid = ~np.isnan(d.gps[:, 0])
    # GPS noise makes a heading from two nearby fixes meaningless, so the filter waits until
    # odometry says the car has driven 40 m, then fits a line through the fixes from that stretch.
    travelled = np.cumsum(d.odo_speed) * DT
    start = int(np.searchsorted(travelled, 40.0))
    window = [k for k, i in enumerate(d.gps_idx) if i <= start and valid[k]]
    if len(window) >= 2:
        pts = d.gps[window]
        direction = np.linalg.svd(pts - pts.mean(0))[2][0]
        if direction @ (pts[-1] - pts[0]) < 0:
            direction = -direction
        heading0 = math.atan2(direction[1], direction[0])
        p0 = pts[-1]
    else:
        first = int(np.argmax(valid))
        heading0, p0, start = 0.0, d.gps[first], int(d.gps_idx[first])
    x = np.array([p0[0], p0[1], heading0, 0.0])
    P = np.diag([GPS_SIGMA**2, GPS_SIGMA**2, math.radians(25) ** 2, math.radians(1.0) ** 2])
    out = np.zeros((n, 4))
    cov = np.zeros((n, 2, 2))
    R = np.eye(2) * GPS_SIGMA**2
    rejected_in_a_row = 0
    last_fix: tuple[int, np.ndarray] | None = None
    for i in range(n):
        if i < start:
            # Before the filter starts, report the latest GPS fix.
            k = min(i // GPS_EVERY, len(d.gps) - 1)
            fix = d.gps[k] if valid[k] else p0
            out[i] = (fix[0], fix[1], heading0, 0.0)
            cov[i] = np.eye(2) * GPS_SIGMA**2
            continue
        v, w = d.odo_speed[i], d.odo_yaw_rate[i] - x[3]
        c, s = math.cos(x[2]), math.sin(x[2])
        x = x + np.array([v * c * DT, v * s * DT, w * DT, 0.0])
        F = np.eye(4)
        F[0, 2], F[1, 2], F[2, 3] = -v * s * DT, v * c * DT, -DT
        sv = SPEED_NOISE + 0.02 * v
        Q = np.diag([(sv * DT) ** 2, (sv * DT) ** 2, (2 * GYRO_NOISE * DT) ** 2, (math.radians(0.01)) ** 2])
        P = F @ P @ F.T + Q
        k = fixes.get(i)
        if k is not None and valid[k]:
            innovation = d.gps[k] - x[:2]
            S = P[:2, :2] + R
            m2 = float(innovation @ np.linalg.solve(S, innovation))
            # Reject multipath outliers, but not forever: a long run of rejections means the
            # filter itself has drifted and must re-anchor.
            if m2 < 16.0 or rejected_in_a_row >= 4:
                K = P[:, :2] @ np.linalg.inv(S)
                x = x + K @ innovation
                P = (np.eye(4) - K[:, :2] @ np.eye(2, 4)) @ P
                rejected_in_a_row = 0
            else:
                rejected_in_a_row += 1
            # Course over ground: the direction between consecutive fixes is a heading measurement
            # once the car has moved far enough for GPS noise to matter less than the motion.
            if last_fix is not None:
                moved = float(travelled[i] - travelled[last_fix[0]])
                delta = d.gps[k] - last_fix[1]
                if moved > 8.0 and np.linalg.norm(delta) > 4.0:
                    course = math.atan2(delta[1], delta[0])
                    r_course = (2.0 * GPS_SIGMA / moved) ** 2
                    innov = float(wrap(course - x[2]))
                    s_course = P[2, 2] + r_course
                    if innov * innov / s_course < 9.0:
                        K = P[:, 2] / s_course
                        x = x + K * innov
                        P = P - np.outer(K, P[2, :])
            last_fix = (i, d.gps[k].copy())
        x[3] = float(np.clip(x[3], -math.radians(2.0), math.radians(2.0)))
        x[2] = wrap(x[2])
        out[i] = x
        cov[i] = P[:2, :2]
    return Track(xy=out[:, :2], heading=out[:, 2], cov=cov)


# --- 2. Map matching ----------------------------------------------------------------------------


class RoadGraph:
    """Shortest driving distances between junctions, cached per source junction."""

    def __init__(self, rm: RoadMap) -> None:
        super().__init__()
        self.rm = rm
        self._cache: dict[int, dict[int, float]] = {}

    def from_node(self, u: int, limit: float) -> dict[int, float]:
        cached = self._cache.get(u)
        if cached is not None:
            return cached
        dist = {u: 0.0}
        heap = [(0.0, u)]
        while heap:
            d, node = heapq.heappop(heap)
            if d > dist.get(node, math.inf) or d > limit:
                continue
            for eid in self.rm.out_edges.get(node, []):
                e = self.rm.edges[eid]
                nd = d + e.length
                if nd < dist.get(e.v, math.inf):
                    dist[e.v] = nd
                    heapq.heappush(heap, (nd, e.v))
        self._cache[u] = dist
        return dist


def _carriageway_projection(p: np.ndarray, e: Edge) -> tuple[float, float]:
    """Distance from p to this direction's carriageway centre, and the along-edge position."""
    lateral, along, t = point_to_segment(p, e.a, e.b)
    overshoot = 0.0
    if t <= 0.0 or t >= 1.0:
        end = np.array(e.a if t <= 0.0 else e.b)
        ux, uy = (np.array(e.b) - np.array(e.a)) / max(e.length, 1e-9)
        overshoot = abs((p[0] - end[0]) * ux + (p[1] - end[1]) * uy)
    return math.hypot(lateral - e.carriageway_offset(), overshoot), along


def _snap(p: np.ndarray, e: Edge, lateral: float) -> np.ndarray:
    _, along, _ = point_to_segment(p, e.a, e.b)
    a, b = np.array(e.a), np.array(e.b)
    u = (b - a) / max(e.length, 1e-9)
    return a + u * along + np.array([u[1], -u[0]]) * lateral


def map_match(rm: RoadMap, graph: RoadGraph, track: Track, sigma: float) -> Track:
    """Viterbi over candidate roads at 1 Hz, then snaps each 10 Hz sample onto its road."""
    n = len(track.xy)
    steps = list(range(0, n, GPS_EVERY))
    beta = 8.0
    cand: list[list[tuple[int, float, float]]] = []  # per step: (edge, distance, along)
    for i in steps:
        p = track.xy[i]
        options = []
        for e in rm.edges:
            # Cheap box test before projecting.
            if not (
                min(e.a[0], e.b[0]) - 40 < p[0] < max(e.a[0], e.b[0]) + 40
                and min(e.a[1], e.b[1]) - 40 < p[1] < max(e.a[1], e.b[1]) + 40
            ):
                continue
            dist, along = _carriageway_projection(p, e)
            if dist < 35 and abs(wrap(e.heading - track.heading[i])) < math.radians(100):
                options.append((e.id, dist, along))
        options.sort(key=lambda o: o[1])
        cand.append(options[:8])

    score: list[np.ndarray] = []
    back: list[np.ndarray] = []
    for k, options in enumerate(cand):
        emis = np.array([-0.5 * (dist / sigma) ** 2 for _, dist, _ in options]) if options else np.zeros(0)
        if k == 0 or not len(score[-1]) or not len(options):
            score.append(emis)
            back.append(np.full(len(options), -1))
            continue
        straight = float(np.linalg.norm(track.xy[steps[k]] - track.xy[steps[k - 1]]))
        trans = np.full((len(cand[k - 1]), len(options)), -np.inf)
        for a, (ea, _, along_a) in enumerate(cand[k - 1]):
            edge_a = rm.edges[ea]
            reach = graph.from_node(edge_a.v, 600.0)
            for b, (eb, _, along_b) in enumerate(options):
                edge_b = rm.edges[eb]
                if ea == eb:
                    route = along_b - along_a
                    if route < -3.0:
                        continue
                    route = max(route, 0.0)
                elif edge_b.u in reach:
                    route = (edge_a.length - along_a) + reach[edge_b.u] + along_b
                else:
                    continue
                trans[a, b] = -abs(route - straight) / beta
        total = score[-1][:, None] + trans
        best = np.argmax(total, axis=0)
        new = total[best, np.arange(len(options))] + emis
        if np.all(np.isinf(new)):
            # No connected path: restart the chain here rather than fail the whole drive.
            score.append(emis)
            back.append(np.full(len(options), -1))
        else:
            score.append(new)
            back.append(best)

    chosen = np.full(len(steps), -1)
    k = len(steps) - 1
    while k >= 0:
        if not len(score[k]):
            k -= 1
            continue
        j = int(np.argmax(score[k]))
        while k >= 0 and j >= 0:
            chosen[k] = cand[k][j][0]
            j = int(back[k][j])
            k -= 1
    # Fill gaps (steps with no candidate) from the nearest matched step.
    known = np.nonzero(chosen >= 0)[0]
    for k in range(len(steps)):
        if chosen[k] < 0 and len(known):
            chosen[k] = chosen[known[np.argmin(np.abs(known - k))]]

    xy = np.zeros_like(track.xy)
    edge = np.zeros(n, dtype=int)
    for i in range(n):
        k0 = min(i // GPS_EVERY, len(steps) - 1)
        k1 = min(k0 + 1, len(steps) - 1)
        options = {int(chosen[k0]), int(chosen[k1])}
        best_e = min(options, key=lambda e: _carriageway_projection(track.xy[i], rm.edges[e])[0])
        e = rm.edges[best_e]
        edge[i] = best_e
        xy[i] = _snap(track.xy[i], e, e.carriageway_offset())
    return Track(xy=xy, heading=track.heading, edge=edge)


# --- 3. Graph SLAM ------------------------------------------------------------------------------


def levenberg_marquardt(
    fn: Callable[[np.ndarray], tuple[np.ndarray, csr_matrix]],
    x0: np.ndarray,
    huber: float,
    iterations: int,
) -> np.ndarray:
    """Sparse Levenberg-Marquardt on whitened residuals, with a Huber loss applied by
    iteratively reweighted least squares (residuals beyond `huber` sigmas are down-weighted)."""

    def robust_cost(r: np.ndarray) -> float:
        a = np.abs(r)
        return float(np.sum(np.where(a <= huber, 0.5 * a * a, huber * (a - 0.5 * huber))))

    x = x0.copy()
    r, jac = fn(x)
    cost = robust_cost(r)
    lam = 1e-3
    for _ in range(iterations):
        w = np.where(np.abs(r) <= huber, 1.0, huber / np.maximum(np.abs(r), 1e-12))
        jw = diags(w) @ jac
        h = (jac.T @ jw).tocsc()
        g = jac.T @ (w * r)
        diag = h.diagonal()
        improved = False
        while lam < 1e8:
            step = spsolve(h + diags(lam * np.maximum(diag, 1e-9)), -g)
            r_new, jac_new = fn(x + step)
            new_cost = robust_cost(r_new)
            if new_cost < cost:
                x, r, jac = x + step, r_new, jac_new
                improved = cost - new_cost > 1e-6 * cost
                cost = new_cost
                lam = max(lam / 3, 1e-9)
                break
            lam *= 5
        if not improved:
            break
    return x


@dataclass
class _DriveBlock:
    drive: Drive
    init: Track
    kf: np.ndarray
    pose0: int  # index of this drive's first pose variable
    bias: int  # index of this drive's gyro-bias variable; its wheel-speed scale is the next one
    odo: np.ndarray  # (m-1, 4): dx, dy in the earlier keyframe's frame, raw yaw change, duration


def _odometry(d: Drive, kf: np.ndarray, bias0: float) -> np.ndarray:
    odo = []
    for j in range(len(kf) - 1):
        th, dx, dy, dth_raw = 0.0, 0.0, 0.0, 0.0
        for i in range(kf[j], kf[j + 1]):
            dx += d.odo_speed[i] * math.cos(th) * DT
            dy += d.odo_speed[i] * math.sin(th) * DT
            th += (d.odo_yaw_rate[i] - bias0) * DT
            dth_raw += d.odo_yaw_rate[i] * DT
        odo.append((dx, dy, dth_raw, (kf[j + 1] - kf[j]) * DT))
    return np.array(odo)


def graph_slam(
    drives: list[tuple[Drive, Track]],
    landmark_prior: tuple[np.ndarray, np.ndarray, np.ndarray] | None,
) -> tuple[list[Track], np.ndarray, np.ndarray]:
    """Least-squares SLAM over one or more drives that observe the same landmarks.

    Variables are every drive's 2 Hz keyframe poses and gyro bias, plus the position of every
    landmark any drive saw. Factors are odometry between keyframes, GPS fixes (robust loss, so
    multipath jumps are down-weighted), range/bearing landmark observations and, when
    `landmark_prior` = (ids, positions, sigmas) is given, a prior on each mapped landmark. Returns the
    optimized tracks, landmark ids, and landmark positions.
    """
    lm_ids = sorted({o[1] for d, _ in drives for o in d.obs})
    lm_index = {lid: j for j, lid in enumerate(lm_ids)}
    blocks: list[_DriveBlock] = []
    n_pose_vars = 0
    for d, init in drives:
        kf = np.arange(0, len(d.t), KF_EVERY)
        blocks.append(_DriveBlock(d, init, kf, n_pose_vars, -1, _odometry(d, kf, 0.0)))
        n_pose_vars += 3 * len(kf)
    lm0 = n_pose_vars
    bias0 = lm0 + 2 * len(lm_ids)
    for b_i, block in enumerate(blocks):
        block.bias = bias0 + 2 * b_i
    n_vars = bias0 + 2 * len(blocks)

    x0 = np.zeros(n_vars)
    x0[bias0 + 1 :: 2] = 1.0  # wheel-speed scale
    lm_sum = np.zeros((len(lm_ids), 2))
    lm_count = np.zeros(len(lm_ids))
    for block in blocks:
        m = len(block.kf)
        x0[block.pose0 : block.pose0 + 3 * m : 3] = block.init.xy[block.kf, 0]
        x0[block.pose0 + 1 : block.pose0 + 3 * m : 3] = block.init.xy[block.kf, 1]
        x0[block.pose0 + 2 : block.pose0 + 3 * m : 3] = block.init.heading[block.kf]
        for i, lid, r, b in block.drive.obs:
            j = block.kf[i // KF_EVERY]
            th = block.init.heading[j]
            lm_sum[lm_index[lid]] += block.init.xy[j] + r * np.array([math.cos(th + b), math.sin(th + b)])
            lm_count[lm_index[lid]] += 1
    x0[lm0:bias0] = (lm_sum / np.maximum(lm_count, 1)[:, None]).ravel()
    prior_idx = np.zeros(0, dtype=int)
    prior_xy = np.zeros((0, 2))
    prior_sigma = np.zeros(0)
    if landmark_prior is not None:
        known = {int(lid): (xy, sig) for lid, xy, sig in zip(*landmark_prior)}
        pairs = [(lm_index[lid], *known[lid]) for lid in lm_ids if lid in known]
        if pairs:
            prior_idx = np.array([p[0] for p in pairs])
            prior_xy = np.array([p[1] for p in pairs])
            prior_sigma = np.array([p[2] for p in pairs])
            x0[lm0:bias0].reshape(-1, 2)[prior_idx] = prior_xy

    # Flattened factor tables, indexed into the global variable vector.
    odo_i0, odo_bias, odo_meas = [], [], []
    gps_i, gps_xy = [], []
    obs_pose, obs_lm, obs_r, obs_b = [], [], [], []
    for block in blocks:
        d = block.drive
        m = len(block.kf)
        odo_i0.extend(block.pose0 + 3 * np.arange(m - 1))
        odo_bias.extend([block.bias] * (m - 1))
        odo_meas.append(block.odo)
        for k, i in enumerate(d.gps_idx):
            if not np.isnan(d.gps[k, 0]):
                gps_i.append(block.pose0 + 3 * (int(i) // KF_EVERY))
                gps_xy.append(d.gps[k])
        for i, lid, r, b in d.obs:
            obs_pose.append(block.pose0 + 3 * (i // KF_EVERY))
            obs_lm.append(lm0 + 2 * lm_index[lid])
            obs_r.append(r)
            obs_b.append(b)
    odo_i0_a = np.array(odo_i0)
    odo_bias_a = np.array(odo_bias)
    odo_m = np.concatenate(odo_meas)
    gps_i_a = np.array(gps_i)
    gps_xy_a = np.array(gps_xy)
    obs_pose_a, obs_lm_a = np.array(obs_pose), np.array(obs_lm)
    obs_r_a, obs_b_a = np.array(obs_r), np.array(obs_b)
    bias_idx = np.array([b.bias for b in blocks])
    prior_var = lm0 + 2 * prior_idx

    sigma_t = 0.05 + 0.03 * np.hypot(odo_m[:, 0], odo_m[:, 1])
    sigma_th = GYRO_NOISE * 2
    n_odo, n_gps, n_obs, n_pri, n_blk = len(odo_i0_a), len(gps_i_a), len(obs_pose_a), len(prior_idx), len(blocks)

    def residuals_and_jacobian(v: np.ndarray) -> tuple[np.ndarray, csr_matrix]:
        rows: list[np.ndarray] = []
        cols: list[np.ndarray] = []
        vals: list[np.ndarray] = []

        def add(r0: np.ndarray, c: np.ndarray, val: np.ndarray) -> None:
            rows.append(r0)
            cols.append(c)
            vals.append(np.broadcast_to(val, r0.shape))

        # Odometry: the next keyframe, seen from this one, should be where the scaled wheel
        # odometry says, and the heading change should match the bias-corrected gyro.
        i0 = odo_i0_a
        x0_, y0_, t0_ = v[i0], v[i0 + 1], v[i0 + 2]
        dxw, dyw = v[i0 + 3] - x0_, v[i0 + 4] - y0_
        c, s_ = np.cos(t0_), np.sin(t0_)
        bx, by = c * dxw + s_ * dyw, -s_ * dxw + c * dyw
        scale, bias = v[odo_bias_a + 1], v[odo_bias_a]
        r_odo = np.stack(
            [
                (bx - scale * odo_m[:, 0]) / sigma_t,
                (by - scale * odo_m[:, 1]) / sigma_t,
                wrap(v[i0 + 5] - t0_ - (odo_m[:, 2] - bias * odo_m[:, 3])) / sigma_th,
            ],
            axis=1,
        ).ravel()
        ra, rb, rc = 3 * np.arange(n_odo), 3 * np.arange(n_odo) + 1, 3 * np.arange(n_odo) + 2
        for r0, coeffs in (
            (ra, [(i0, -c), (i0 + 1, -s_), (i0 + 2, by), (i0 + 3, c), (i0 + 4, s_), (odo_bias_a + 1, -odo_m[:, 0])]),
            (rb, [(i0, s_), (i0 + 1, -c), (i0 + 2, -bx), (i0 + 3, -s_), (i0 + 4, c), (odo_bias_a + 1, -odo_m[:, 1])]),
        ):
            for col, val in coeffs:
                add(r0, col, val / sigma_t)
        add(rc, i0 + 2, -1.0 / sigma_th)
        add(rc, i0 + 5, 1.0 / sigma_th)
        add(rc, odo_bias_a, odo_m[:, 3] / sigma_th)
        base = 3 * n_odo

        # GPS fixes.
        r_gps = (np.stack([v[gps_i_a], v[gps_i_a + 1]], axis=1) - gps_xy_a).ravel() / GPS_SIGMA
        add(base + 2 * np.arange(n_gps), gps_i_a, 1.0 / GPS_SIGMA)
        add(base + 2 * np.arange(n_gps) + 1, gps_i_a + 1, 1.0 / GPS_SIGMA)
        base += 2 * n_gps

        # Landmark range and bearing.
        dx = v[obs_lm_a] - v[obs_pose_a]
        dy = v[obs_lm_a + 1] - v[obs_pose_a + 1]
        q = np.maximum(dx * dx + dy * dy, 1e-6)
        rho = np.sqrt(q)
        r_lm = np.stack(
            [
                (rho - obs_r_a) / LANDMARK_RANGE_NOISE,
                wrap(np.arctan2(dy, dx) - v[obs_pose_a + 2] - obs_b_a) / LANDMARK_BEARING_NOISE,
            ],
            axis=1,
        ).ravel()
        rr, rbr = base + 2 * np.arange(n_obs), base + 2 * np.arange(n_obs) + 1
        sr, sb = LANDMARK_RANGE_NOISE, LANDMARK_BEARING_NOISE
        add(rr, obs_lm_a, dx / rho / sr)
        add(rr, obs_lm_a + 1, dy / rho / sr)
        add(rr, obs_pose_a, -dx / rho / sr)
        add(rr, obs_pose_a + 1, -dy / rho / sr)
        add(rbr, obs_lm_a, -dy / q / sb)
        add(rbr, obs_lm_a + 1, dx / q / sb)
        add(rbr, obs_pose_a, dy / q / sb)
        add(rbr, obs_pose_a + 1, -dx / q / sb)
        add(rbr, obs_pose_a + 2, -1.0 / sb)
        base += 2 * n_obs

        # Priors: mapped landmark positions, then each drive's gyro bias and wheel scale.
        r_pri = ((np.stack([v[prior_var], v[prior_var + 1]], axis=1) - prior_xy) / prior_sigma[:, None]).ravel()
        add(base + 2 * np.arange(n_pri), prior_var, 1.0 / prior_sigma)
        add(base + 2 * np.arange(n_pri) + 1, prior_var + 1, 1.0 / prior_sigma)
        base += 2 * n_pri
        r_blk = np.stack([v[bias_idx] / math.radians(1.0), (v[bias_idx + 1] - 1.0) / 0.03], 1).ravel()
        add(base + 2 * np.arange(n_blk), bias_idx, 1.0 / math.radians(1.0))
        add(base + 2 * np.arange(n_blk) + 1, bias_idx + 1, 1.0 / 0.03)
        base += 2 * n_blk

        res = np.concatenate([r_odo, r_gps, r_lm, r_pri, r_blk])
        jac = coo_matrix(
            (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(base, n_vars)
        ).tocsr()
        return res, jac

    v = levenberg_marquardt(residuals_and_jacobian, x0, huber=2.0, iterations=40)
    lms = v[lm0:bias0].reshape(-1, 2)

    tracks = []
    for block in blocks:
        d = block.drive
        n = len(d.t)
        m = len(block.kf)
        poses = v[block.pose0 : block.pose0 + 3 * m].reshape(m, 3)
        bias, scale = v[block.bias], v[block.bias + 1]
        # Between keyframes, dead-reckon from the optimized keyframe with the estimated gyro bias.
        xy = np.zeros((n, 2))
        heading = np.zeros(n)
        for j in range(m):
            px, py, th = poses[j]
            end = block.kf[j + 1] if j + 1 < m else n
            for i in range(block.kf[j], end):
                xy[i], heading[i] = (px, py), th
                px += scale * d.odo_speed[i] * math.cos(th) * DT
                py += scale * d.odo_speed[i] * math.sin(th) * DT
                th += (d.odo_yaw_rate[i] - bias) * DT
        tracks.append(Track(xy=xy, heading=wrap(heading)))
    return tracks, np.array(lm_ids), lms


@dataclass
class LandmarkMap:
    ids: np.ndarray
    xy: np.ndarray
    sigma: np.ndarray  # rough 1-sigma position error, shrinking with the number of drives that saw it
    passes: np.ndarray


def build_landmark_map(drives: list[Drive]) -> LandmarkMap:
    """Fuses per-drive SLAM landmark estimates from a mapping fleet into one landmark map.

    Each drive is solved on its own, so its landmarks carry that drive's GPS bias. Biases of
    different drives are independent, so averaging a landmark over the drives that saw it shrinks
    the error roughly with the square root of the number of passes.
    """
    sums: dict[int, np.ndarray] = {}
    weights: dict[int, float] = {}
    passes: dict[int, int] = {}
    for d in drives:
        (_,), ids, lms = graph_slam([(d, kalman_filter(d))], None)
        counts: dict[int, int] = {}
        for _, lid, _, _ in d.obs:
            counts[lid] = counts.get(lid, 0) + 1
        for lid, xy in zip(ids.tolist(), lms):
            w = float(counts[lid])
            sums[lid] = sums.get(lid, np.zeros(2)) + w * xy
            weights[lid] = weights.get(lid, 0.0) + w
            passes[lid] = passes.get(lid, 0) + 1
    ids = np.array(sorted(sums))
    xy = np.array([sums[i] / weights[i] for i in ids])
    n = np.array([passes[i] for i in ids])
    return LandmarkMap(ids=ids, xy=xy, sigma=4.0 / np.sqrt(n), passes=n)


# --- 4. Lane-level localization ------------------------------------------------------------------


def lane_localize(rm: RoadMap, d: Drive, source: Track, matched: Track) -> Track:
    """Picks a lane on each matched road with an HMM, then places the car inside it."""
    n = len(source.xy)
    assert matched.edge is not None
    max_lanes = max(rm.edges[int(e)].lanes for e in set(matched.edge.tolist()))
    neg = -1e9
    log_stay, log_move = math.log(0.995), math.log(0.0025)
    log_hit, log_miss = math.log(0.95), math.log(0.03)

    def emission(i: int, e: Edge) -> np.ndarray:
        lateral, _, _ = point_to_segment(source.xy[i], e.a, e.b)
        out = np.full(max_lanes, neg)
        for k in range(e.lanes):
            ll = -0.5 * ((lateral - e.lane_offset(k)) / 2.0) ** 2
            if not np.isnan(d.cam_offset[i]):
                ll += log_hit if d.cam_left_solid[i] == (k == e.lanes - 1) else log_miss
                ll += log_hit if d.cam_right_solid[i] == (k == 0) else log_miss
            out[k] = ll
        return out

    score = emission(0, rm.edges[int(matched.edge[0])])
    back = np.zeros((n, max_lanes), dtype=int)
    last_offset = d.cam_offset[0]
    for i in range(1, n):
        e = rm.edges[int(matched.edge[i])]
        prev_lanes = rm.edges[int(matched.edge[i - 1])].lanes
        # A lane change shows up as the camera's offset jumping by about a lane width: moving
        # right (towards the curb, lane index down) makes it jump from positive to negative.
        step = 0
        off = d.cam_offset[i]
        if not np.isnan(off) and not np.isnan(last_offset):
            if last_offset > 0.8 and off < -0.8:
                step = -1
            elif last_offset < -0.8 and off > 0.8:
                step = 1
        if not np.isnan(off):
            last_offset = off
        trans = np.full((max_lanes, max_lanes), neg)
        for a in range(prev_lanes):
            for b in range(e.lanes):
                # Lanes carry over by index when the road gains or drops lanes.
                stay_lane = min(a, e.lanes - 1)
                if step == 0:
                    trans[a, b] = log_stay if b == stay_lane else (log_move if abs(b - stay_lane) == 1 else neg)
                else:
                    want = min(max(a + step, 0), e.lanes - 1)
                    trans[a, b] = math.log(0.95) if b == want else (math.log(0.04) if b == stay_lane else neg)
        total = score[:, None] + trans
        back[i] = np.argmax(total, axis=0)
        score = total[back[i], np.arange(max_lanes)] + emission(i, e)

    lanes = np.zeros(n, dtype=int)
    lanes[-1] = int(np.argmax(score))
    for i in range(n - 1, 0, -1):
        lanes[i - 1] = back[i, lanes[i]]

    xy = np.zeros_like(source.xy)
    for i in range(n):
        e = rm.edges[int(matched.edge[i])]
        k = min(int(lanes[i]), e.lanes - 1)
        off = 0.0 if np.isnan(d.cam_offset[i]) else float(np.clip(d.cam_offset[i], -LANE_WIDTH_M / 2, LANE_WIDTH_M / 2))
        xy[i] = _snap(source.xy[i], e, e.lane_offset(k) + off)
    return Track(xy=xy, heading=source.heading, edge=matched.edge, lane=lanes)


# --- Scoring ------------------------------------------------------------------------------------


def score(
    rm: RoadMap, d: Drive, xy: np.ndarray, idx: np.ndarray, edge: np.ndarray | None, lane: np.ndarray | None
) -> dict[str, float]:
    """Error against ground truth at sample indices `idx` (positions `xy` are aligned with idx)."""
    truth = np.stack([d.x[idx], d.y[idx]], axis=1)
    delta = xy - truth
    c, s = np.cos(d.heading[idx]), np.sin(d.heading[idx])
    along = delta[:, 0] * c + delta[:, 1] * s
    lateral = -delta[:, 0] * s + delta[:, 1] * c
    err = np.hypot(delta[:, 0], delta[:, 1])
    on_road = []
    in_lane = []
    for q, i in enumerate(idx):
        te = rm.edges[int(d.edge[i])]
        if edge is not None:
            me = rm.edges[int(edge[i])]
            # OSM splits one street into many ways, so compare the street and the direction.
            right_road = me.name == te.name and abs(wrap(me.heading - te.heading)) < math.pi / 2
        else:
            # Without a map, a position counts as on the right road if it lies within that road's
            # paved width.
            lat, _, t = point_to_segment(xy[q], te.a, te.b)
            right_road = te.left_edge <= lat <= te.right_edge and 0.0 < t < 1.0
        on_road.append(right_road)
        if lane is not None:
            k = min(int(lane[i]), te.lanes - 1)
        else:
            lat, _, _ = point_to_segment(xy[q], te.a, te.b)
            k = int(round((te.lane0_offset - lat) / LANE_WIDTH_M))
        in_lane.append(right_road and k == int(d.lane[i]))
    return {
        "rms_m": round(float(np.sqrt(np.mean(err**2))), 2),
        "p95_m": round(float(np.percentile(err, 95)), 2),
        "max_m": round(float(err.max()), 2),
        "lateral_rms_m": round(float(np.sqrt(np.mean(lateral**2))), 2),
        "along_rms_m": round(float(np.sqrt(np.mean(along**2))), 2),
        "on_road_pct": round(100 * float(np.mean(on_road)), 1),
        "in_lane_pct": round(100 * float(np.mean(in_lane)), 1),
    }
