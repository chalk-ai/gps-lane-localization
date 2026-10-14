#!/usr/bin/env -S uv run --script
#
# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy<3", "scipy==1.16.*"]
# ///
"""Scores every stage over many simulated drives, without the server.

./evaluate.py --profile urban-canyon --drives 20
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent / "app"))

import estimators as est  # noqa: E402
import roadmap  # noqa: E402
import sim  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", default="urban-canyon", choices=sorted(sim.GPS_PROFILES))
    parser.add_argument("--drives", type=int, default=10)
    parser.add_argument("--fleet-drives", type=int, default=60)
    args = parser.parse_args()

    rm = roadmap.load_roadmap(Path(__file__).parent / "app" / "data" / "soma.json")
    graph = est.RoadGraph(rm)
    landmarks = sim.make_landmarks(rm, 0)
    fleet = [
        sim.simulate(rm, landmarks, 1000 + k, sim.GPS_PROFILES["urban-canyon"], False) for k in range(args.fleet_drives)
    ]
    lmap = est.build_landmark_map(fleet)
    err = np.linalg.norm(lmap.xy - landmarks[lmap.ids], axis=1)
    print(
        f"landmark map from {args.fleet_drives} drives: {len(lmap.ids)} landmarks, median error {np.median(err):.2f} m"
    )

    results: dict[str, list[dict[str, float]]] = {}
    for seed in range(args.drives):
        d = sim.simulate(rm, landmarks, seed, sim.GPS_PROFILES[args.profile], True)
        idx = np.arange(len(d.t))
        kf = est.kalman_filter(d)
        kfm = est.map_match(rm, graph, kf, 6.0)
        (slam,), _, _ = est.graph_slam([(d, kf)], (lmap.ids, lmap.xy, lmap.sigma))
        lane = est.lane_localize(rm, d, slam, est.map_match(rm, graph, slam, 3.0))
        valid = ~np.isnan(d.gps[:, 0])
        for name, s in (
            ("raw GPS", est.score(rm, d, d.gps[valid], d.gps_idx[valid], None, None)),
            ("Kalman filter", est.score(rm, d, kf.xy, idx, None, None)),
            ("+ map matching", est.score(rm, d, kfm.xy, idx, kfm.edge, None)),
            ("+ SLAM landmark map", est.score(rm, d, slam.xy, idx, None, None)),
            ("+ lane map", est.score(rm, d, lane.xy, idx, lane.edge, lane.lane)),
        ):
            results.setdefault(name, []).append(s)

    print(f"\nmedians over {args.drives} drives, {args.profile} GPS")
    print(f"{'stage':22s} {'RMS':>7s} {'lateral':>8s} {'along':>7s} {'road':>6s} {'lane':>6s}")
    for name, rows in results.items():
        med = {k: float(np.median([r[k] for r in rows])) for k in rows[0]}
        print(
            f"{name:22s} {med['rms_m']:6.2f}m {med['lateral_rms_m']:7.2f}m {med['along_rms_m']:6.2f}m "
            f"{med['on_road_pct']:5.1f}% {med['in_lane_pct']:5.1f}%"
        )


if __name__ == "__main__":
    main()
