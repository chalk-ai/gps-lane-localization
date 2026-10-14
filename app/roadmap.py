"""A lane-level road map built from an OpenStreetMap extract.

Every OSM way becomes directed edges, one per consecutive node pair and drivable direction. An
edge knows how many lanes run in its direction and where each lane's centre sits relative to
the way's centreline, so the same structure serves road-level map matching and lane-level
localization. Coordinates are metres in a local east/north frame.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

LANE_WIDTH_M = 3.2
EARTH_RADIUS_M = 6_371_000.0
DRIVABLE = {
    "primary",
    "secondary",
    "tertiary",
    "residential",
    "unclassified",
    "primary_link",
    "secondary_link",
    "tertiary_link",
}
# Lane counts per direction when a way carries no lanes tag.
DEFAULT_LANES = {"primary": 2, "secondary": 2, "tertiary": 1}


@dataclass(frozen=True)
class Edge:
    id: int
    way_id: int
    name: str
    u: int
    v: int
    a: tuple[float, float]
    b: tuple[float, float]
    lanes: int
    # Lateral offset (metres, positive = right of travel) of the centre of lane 0, the curb lane.
    # Lane k's centre is at lane0_offset - k * LANE_WIDTH_M.
    lane0_offset: float
    # Lateral offsets of the paved road's right and left edges, relative to the centreline.
    right_edge: float
    left_edge: float
    two_way: bool

    @property
    def length(self) -> float:
        return math.hypot(self.b[0] - self.a[0], self.b[1] - self.a[1])

    @property
    def heading(self) -> float:
        return math.atan2(self.b[1] - self.a[1], self.b[0] - self.a[0])

    def lane_offset(self, lane: int) -> float:
        return self.lane0_offset - lane * LANE_WIDTH_M

    def carriageway_offset(self) -> float:
        """Offset of the middle of this direction's lanes: where road-level snapping lands."""
        return self.lane0_offset - (self.lanes - 1) * LANE_WIDTH_M / 2


@dataclass
class RoadMap:
    origin: tuple[float, float]
    nodes: dict[int, tuple[float, float]]
    edges: list[Edge]
    out_edges: dict[int, list[int]]
    # Polylines of I-80 and its ramps, where GPS degrades (overpass shadow and multipath).
    freeway: list[np.ndarray]
    render: dict[str, list[list[list[float]]]] = field(default_factory=dict)

    def project(self, lat: float, lon: float) -> tuple[float, float]:
        lat0, lon0 = self.origin
        x = math.radians(lon - lon0) * EARTH_RADIUS_M * math.cos(math.radians(lat0))
        y = math.radians(lat - lat0) * EARTH_RADIUS_M
        return (x, y)


def _lanes_per_direction(tags: dict[str, str], highway: str, two_way: bool) -> tuple[int, int]:
    def num(key: str) -> int | None:
        try:
            return max(1, int(tags[key].split(";")[0]))
        except (KeyError, ValueError):
            return None

    default = DEFAULT_LANES.get(highway, 1)
    if not two_way:
        return (num("lanes") or default, 0)
    fwd, bwd = num("lanes:forward"), num("lanes:backward")
    total = num("lanes")
    if fwd is None and bwd is None and total is not None:
        fwd = bwd = max(1, total // 2)
    return (fwd or default, bwd or default)


def _offset_polyline(pts: np.ndarray, offset: float) -> np.ndarray:
    """Offsets a polyline to its right (positive) with mitred joins, for drawing lane lines."""
    if len(pts) < 2:
        return pts
    d = np.diff(pts, axis=0)
    d /= np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-9)
    right = np.stack([d[:, 1], -d[:, 0]], axis=1)
    normals = np.vstack([right[:1], right[:-1] + right[1:], right[-1:]])
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-9)
    # Mitre: scale by 1/cos(half the turn angle), capped so sharp corners do not spike.
    cos_half = np.ones(len(pts))
    cos_half[1:-1] = np.clip(np.sum(normals[1:-1] * right[:-1], axis=1), 0.5, 1.0)
    return pts + normals * (offset / cos_half)[:, None]


def load_roadmap(path: Path) -> RoadMap:
    data = json.loads(path.read_text())
    ways = data["elements"]
    lats = [p["lat"] for w in ways for p in w["geometry"]]
    lons = [p["lon"] for w in ways for p in w["geometry"]]
    rm = RoadMap(
        origin=((min(lats) + max(lats)) / 2, (min(lons) + max(lons)) / 2), nodes={}, edges=[], out_edges={}, freeway=[]
    )
    render: dict[str, list[list[list[float]]]] = {"edge": [], "divider": [], "center": [], "freeway": []}

    for w in ways:
        tags = w.get("tags", {})
        highway = tags.get("highway", "")
        pts = np.array([rm.project(p["lat"], p["lon"]) for p in w["geometry"]])
        if highway.startswith("motorway"):
            rm.freeway.append(pts)
            render["freeway"].append(np.round(pts, 1).tolist())
            continue
        if highway not in DRIVABLE:
            continue
        # Ways that meet share an OSM node, so identical coordinates identify the junction.
        node_ids = [round(g["lat"] * 1e7) * 10**10 + round(g["lon"] * 1e7) for g in w["geometry"]]
        for nid, p in zip(node_ids, pts):
            rm.nodes[nid] = (float(p[0]), float(p[1]))
        oneway = tags.get("oneway", "no")
        two_way = oneway not in ("yes", "-1", "true", "1")
        fwd, bwd = _lanes_per_direction(tags, highway, two_way)
        directions: list[tuple[list[int], int]] = []
        if oneway == "-1":
            directions.append((list(reversed(node_ids)), fwd))
        else:
            directions.append((node_ids, fwd))
            if two_way:
                directions.append((list(reversed(node_ids)), bwd))
        name = tags.get("name", highway)
        for i, (node_seq, lanes) in enumerate(directions):
            # Two-way: this direction's lanes fill the right half. One-way: lanes straddle the centreline.
            other = (bwd if i == 0 else fwd) if two_way else 0
            if two_way:
                lane0, right, left = (lanes - 0.5) * LANE_WIDTH_M, lanes * LANE_WIDTH_M, -other * LANE_WIDTH_M
            else:
                lane0 = (lanes - 1) * LANE_WIDTH_M / 2
                right, left = lanes * LANE_WIDTH_M / 2, -lanes * LANE_WIDTH_M / 2
            for u, v in zip(node_seq[:-1], node_seq[1:]):
                e = Edge(
                    len(rm.edges), w["id"], name, u, v, rm.nodes[u], rm.nodes[v], lanes, lane0, right, left, two_way
                )
                if e.length < 0.5:
                    continue
                rm.edges.append(e)
                rm.out_edges.setdefault(u, []).append(e.id)

        # Lane-map drawing for the whole way: outer edges, dashed dividers, and a centre line.
        seq = pts if oneway != "-1" else pts[::-1]
        if two_way:
            for k in range(1, fwd):
                render["divider"].append(np.round(_offset_polyline(seq, k * LANE_WIDTH_M), 2).tolist())
            for k in range(1, bwd):
                render["divider"].append(np.round(_offset_polyline(seq, -k * LANE_WIDTH_M), 2).tolist())
            render["edge"].append(np.round(_offset_polyline(seq, fwd * LANE_WIDTH_M), 2).tolist())
            render["edge"].append(np.round(_offset_polyline(seq, -bwd * LANE_WIDTH_M), 2).tolist())
            render["center"].append(np.round(seq, 2).tolist())
        else:
            half = fwd * LANE_WIDTH_M / 2
            for k in range(1, fwd):
                render["divider"].append(np.round(_offset_polyline(seq, half - k * LANE_WIDTH_M), 2).tolist())
            render["edge"].append(np.round(_offset_polyline(seq, half), 2).tolist())
            render["edge"].append(np.round(_offset_polyline(seq, -half), 2).tolist())
    rm.render = render
    return rm


def point_to_segment(p: np.ndarray, a: tuple[float, float], b: tuple[float, float]) -> tuple[float, float, float]:
    """Projection of p onto segment ab: (signed lateral offset, positive right; along-distance; t)."""
    ax, ay = a
    dx, dy = b[0] - ax, b[1] - ay
    length2 = dx * dx + dy * dy
    t = 0.0 if length2 == 0 else ((p[0] - ax) * dx + (p[1] - ay) * dy) / length2
    t = min(1.0, max(0.0, t))
    qx, qy = ax + t * dx, ay + t * dy
    length = math.sqrt(length2)
    # Right of travel is (dy, -dx); the cross product sign picks the side.
    lateral = ((p[0] - qx) * dy - (p[1] - qy) * dx) / max(length, 1e-9)
    return lateral, t * length, t
