"""Serves the GPS-to-lane localization demo.

GET  /            the viewer (index.html)
GET  /api/map     the lane map to draw, the I-80 overpass, and the fleet-built landmark map
POST /api/run     {"profile": "urban-canyon", "seed": 7} -> one simulated drive and every
                  stage's trajectory, per-sample error, and scores against ground truth

At startup the server simulates a mapping fleet and builds the landmark map with SLAM, so the
first request does not pay for it.
"""

from __future__ import annotations

import asyncio
import logging
import os
import random
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field

import estimators as est
import roadmap
import sim

HERE = Path(__file__).resolve().parent
PORT = int(os.environ.get("PORT", "8080"))
FLEET_DRIVES = int(os.environ.get("FLEET_DRIVES", "60"))
LANDMARK_SEED = 0
FLEET_SEED0 = 1000
# Tracks go to the browser at 5 Hz; the estimators run at 10 Hz.
OUT_EVERY = 2

STAGES = [
    ("gps", "Raw GPS"),
    ("kalman", "Kalman filter"),
    ("map", "+ Map matching"),
    ("slam", "+ SLAM landmark map"),
    ("lane", "+ Lane map"),
]

log = logging.getLogger("gps-demo")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
state: dict[str, Any] = {}


def _r(a: np.ndarray, digits: int) -> list[Any]:
    return np.round(a, digits).tolist()


def run_drive(profile: str, seed: int) -> dict[str, Any]:
    rm: roadmap.RoadMap = state["map"]
    lmap: est.LandmarkMap = state["landmark_map"]
    d = sim.simulate(rm, state["landmarks"], seed, sim.GPS_PROFILES[profile], True)
    n = len(d.t)
    timings: dict[str, float] = {}

    def timed(name: str, fn: Any) -> Any:
        t0 = time.perf_counter()
        out = fn()
        timings[name] = round((time.perf_counter() - t0) * 1000, 1)
        return out

    kf = timed("kalman", lambda: est.kalman_filter(d))
    kfm = timed("map", lambda: est.map_match(rm, state["graph"], kf, 6.0))
    slam_tracks = timed("slam", lambda: est.graph_slam([(d, kf)], (lmap.ids, lmap.xy, lmap.sigma)))
    slam = slam_tracks[0][0]
    slamm = est.map_match(rm, state["graph"], slam, 3.0)
    lane = timed("lane", lambda: est.lane_localize(rm, d, slam, slamm))

    all_idx = np.arange(n)
    valid = ~np.isnan(d.gps[:, 0])
    scores = {
        "gps": est.score(rm, d, d.gps[valid], d.gps_idx[valid], None, None),
        "kalman": est.score(rm, d, kf.xy, all_idx, None, None),
        "map": est.score(rm, d, kfm.xy, all_idx, kfm.edge, None),
        "slam": est.score(rm, d, slam.xy, all_idx, None, None),
        "lane": est.score(rm, d, lane.xy, all_idx, lane.edge, lane.lane),
    }
    truth = np.stack([d.x, d.y], axis=1)
    out_idx = np.arange(0, n, OUT_EVERY)
    tracks = {"kalman": kf.xy, "map": kfm.xy, "slam": slam.xy, "lane": lane.xy}
    gps_k = np.minimum(out_idx // sim.GPS_EVERY, len(d.gps) - 1)
    gps_err = np.linalg.norm(d.gps[gps_k] - truth[out_idx], axis=1)

    route_names: list[str] = []
    for e in d.route:
        name = rm.edges[e].name
        if not route_names or route_names[-1] != name:
            route_names.append(name)
    # The Kalman filter's 1-sigma position ellipse at each GPS fix time.
    ellipses = []
    for i in d.gps_idx:
        assert kf.cov is not None
        vals, vecs = np.linalg.eigh(kf.cov[i])
        ellipses.append(
            [
                round(float(np.sqrt(max(vals[1], 0))), 2),
                round(float(np.sqrt(max(vals[0], 0))), 2),
                round(float(np.arctan2(vecs[1, 1], vecs[0, 1])), 3),
            ]
        )
    seen = sorted({lid for _, lid, _, _ in d.obs})
    obs_by_frame: dict[int, list[int]] = {}
    for i, lid, _, _ in d.obs:
        obs_by_frame.setdefault(i // OUT_EVERY, []).append(int(lid))

    return {
        "profile": profile,
        "seed": seed,
        "duration_s": round(float(d.t[-1]), 1),
        "route_m": round(float(sum(rm.edges[e].length for e in d.route)), 0),
        "route_names": route_names,
        "dt": sim.DT * OUT_EVERY,
        "truth": _r(truth[out_idx], 2),
        "truth_lane": d.lane[out_idx].tolist(),
        "tracks": {k: _r(v[out_idx], 2) for k, v in tracks.items()},
        "lane_estimate": lane.lane[out_idx].tolist() if lane.lane is not None else [],
        "errors": {
            "gps": [None if np.isnan(e) else round(float(e), 2) for e in gps_err],
            **{k: _r(np.linalg.norm(v[out_idx] - truth[out_idx], axis=1), 2) for k, v in tracks.items()},
        },
        "gps": [None if np.isnan(p[0]) else [round(float(p[0]), 2), round(float(p[1]), 2)] for p in d.gps],
        "gps_t": _r(d.t[d.gps_idx], 1),
        "gps_shadow": d.gps_shadow.tolist(),
        "kalman_ellipse": ellipses,
        "landmarks_seen": seen,
        "observations": {str(k): v for k, v in obs_by_frame.items()},
        "scores": scores,
        "stages": [{"key": k, "label": label} for k, label in STAGES],
        "timings_ms": timings,
    }


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    t0 = time.perf_counter()
    rm = roadmap.load_roadmap(HERE / "data" / "soma.json")
    landmarks = sim.make_landmarks(rm, LANDMARK_SEED)

    def build() -> est.LandmarkMap:
        fleet = [
            sim.simulate(rm, landmarks, FLEET_SEED0 + k, sim.GPS_PROFILES["urban-canyon"], False)
            for k in range(FLEET_DRIVES)
        ]
        return est.build_landmark_map(fleet)

    landmark_map = await asyncio.to_thread(build)
    err = np.linalg.norm(landmark_map.xy - landmarks[landmark_map.ids], axis=1)
    state.update(
        map=rm, graph=est.RoadGraph(rm), landmarks=landmarks, landmark_map=landmark_map, lock=asyncio.Semaphore(4)
    )
    log.info(
        "fleet of %d drives mapped %d/%d landmarks (median error %.2f m) in %.1fs",
        FLEET_DRIVES,
        len(landmark_map.ids),
        len(landmarks),
        float(np.median(err)),
        time.perf_counter() - t0,
    )
    yield


app = FastAPI(lifespan=lifespan)


@app.get("/")
async def index() -> HTMLResponse:
    return HTMLResponse((HERE / "index.html").read_text())


@app.get("/api/map")
async def map_data() -> JSONResponse:
    rm: roadmap.RoadMap = state["map"]
    lmap: est.LandmarkMap = state["landmark_map"]
    xs = [p[0] for p in rm.nodes.values()]
    ys = [p[1] for p in rm.nodes.values()]
    return JSONResponse(
        {
            "bounds": [min(xs), min(ys), max(xs), max(ys)],
            "lane_width_m": roadmap.LANE_WIDTH_M,
            "render": rm.render,
            "landmark_map": {
                "ids": lmap.ids.tolist(),
                "xy": _r(lmap.xy, 2),
                "passes": lmap.passes.tolist(),
                "true_xy": _r(state["landmarks"][lmap.ids], 2),
            },
            "fleet_drives": FLEET_DRIVES,
            "profiles": list(sim.GPS_PROFILES),
        }
    )


@app.get("/api/status")
async def status() -> JSONResponse:
    return JSONResponse({"ok": True, "fleet_drives": FLEET_DRIVES})


class RunRequest(BaseModel):
    profile: str = Field(default="urban-canyon")
    seed: int | None = Field(default=None, ge=0, le=1_000_000)


@app.post("/api/run")
async def run(req: RunRequest) -> JSONResponse:
    if req.profile not in sim.GPS_PROFILES:
        raise HTTPException(400, f"profile must be one of {sorted(sim.GPS_PROFILES)}")
    seed = req.seed if req.seed is not None else random.randrange(1_000_000)
    async with state["lock"]:
        result = await asyncio.to_thread(run_drive, req.profile, seed)
    return JSONResponse(result)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=PORT)
