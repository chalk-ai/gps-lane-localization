#!/usr/bin/env -S uv run --script
#
# /// script
# requires-python = ">=3.12"
# dependencies = ["httpx"]
# ///
"""Refreshes app/data/soma.json: San Francisco's SoMa road network from OpenStreetMap.

Pulls every drivable way (with its lane and one-way tags) plus I-80 and its ramps, which only
mark where GPS degrades under the overpass. Map data © OpenStreetMap contributors, ODbL 1.0.

    ./fetch_map.py [--bbox south,west,north,east]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import httpx

OVERPASS = "https://overpass-api.de/api/interpreter"
HIGHWAYS = "primary|secondary|tertiary|residential|unclassified|primary_link|secondary_link|tertiary_link|motorway|motorway_link"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bbox", default="37.7745,-122.4065,37.7865,-122.3935")
    parser.add_argument("--out", default=str(Path(__file__).parent / "app" / "data" / "soma.json"))
    args = parser.parse_args()
    query = f'[out:json][timeout:90];(way["highway"~"^({HIGHWAYS})$"]({args.bbox}););out geom tags;'
    for attempt in range(5):
        res = httpx.post(OVERPASS, data={"data": query}, timeout=180, headers={"User-Agent": "chalk-gps-lane-demo"})
        try:
            data = res.json()
        except json.JSONDecodeError:
            # Overpass answers an overloaded request with an HTML error page; back off and retry.
            time.sleep(20 * (attempt + 1))
            continue
        Path(args.out).write_text(json.dumps(data))
        print(f"wrote {len(data['elements'])} ways to {args.out}")
        return
    raise SystemExit("Overpass did not answer; try again later")


if __name__ == "__main__":
    main()
