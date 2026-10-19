#!/usr/bin/env -S uv run --script
#
# /// script
# requires-python = ">=3.12,<3.14"
# dependencies = ["chalkcompute==2.13.4"]
# ///
"""Deploy the GPS-to-lane localization demo as a Chalk ScalingGroup (CPU only).

The server simulates a mapping fleet at startup and builds a landmark map with SLAM, then
serves the viewer and a `POST /api/run` endpoint that drives a car and runs every stage.

Credentials come from your existing `chalk login` (or the CHALK_* env vars):
    ./deploy.py
"""

from __future__ import annotations

import time
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main() -> None:
    from chalkcompute import Image, ReadinessProbe, ScalingGroup

    image = (
        Image.debian_slim(python_version="3.12")
        .pip_install(["numpy<3", "scipy==1.16.*", "fastapi==0.116.1", "uvicorn[standard]==0.35.0"])
        # The app plus its OpenStreetMap extract is larger than inline copies allow.
        .add_local_dir(str(HERE / "app"), "/app", strategy="volume")
        .workdir("/app")
    )
    sg = ScalingGroup(
        image=image,
        name='gps-lanes',
        cpu="2",
        memory="4Gi",
        env={"FLEET_DRIVES": str(60)},
        entrypoint=["python", "/app/server.py"],
        port=8080,
        # /api/status answers only after the fleet's landmark map is built.
        readiness_probe=ReadinessProbe.http("/api/status"),
        min_replicas=1,
        max_replicas=1,
    )
    t0 = time.monotonic()
    try:
        sg.deploy(build_timeout=1200.0, ready_timeout=900.0)
    finally:
        print(f"deploy returned after {time.monotonic() - t0:.0f}s")
    print("web_url:", sg.web_url)


if __name__ == "__main__":
    main()
