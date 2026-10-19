# From a fuzzy GPS fix to the right lane

A car drives a real San Francisco street grid (SoMa, from OpenStreetMap). Its GPS is noisy and
biased, jumps around in the urban canyon, and drops out under the I-80 overpass. Four classic
localization techniques are layered on top, one at a time, and every stage is scored against
ground truth so you can see what each one fixes.

| Stage | What it adds | What it fixes |
| --- | --- | --- |
| Raw GPS | 1 Hz fixes | – |
| **Kalman filter** | An EKF fusing GPS with 10 Hz wheel speed and gyro, estimating gyro bias, gating multipath outliers | Noise, and coasting through dropouts. It still drifts off the road with the GPS bias |
| **Map matching** | A hidden Markov model over nearby road segments (Newson & Krumm): emissions score distance to each road, transitions score whether the driving distance between candidates matches the straight-line distance. Viterbi picks the road sequence, and the car is snapped onto it | Puts the car on the right street. Along-track error from the GPS bias remains |
| **SLAM landmark map** | A mapping fleet drives the area, and graph SLAM over each drive estimates the positions of roadside poles and signs from range/bearing detections, odometry and GPS. Fusing many drives averages out their independent GPS biases. The live drive then runs SLAM with that landmark map as a prior | Removes the GPS bias, so along-track error drops below a metre |
| **Lane map** | The OSM lanes become a lane-level map. A second HMM picks the lane from a lane camera's solid/dashed boundaries and lane-change events, and the camera's in-lane offset places the car inside it | Picks the right lane, and lateral error drops to tens of centimetres |

Medians over 20 drives in the "urban canyon" GPS profile (`./evaluate.py --drives 20`):

| Stage | RMS error | Lateral | Along-track | On the right road | In the right lane |
| --- | --- | --- | --- | --- | --- |
| Raw GPS | 10.5 m | 7.8 m | 7.3 m | 47% | 17% |
| Kalman filter | 8.5 m | 6.2 m | 5.6 m | 48% | 20% |
| + Map matching | 6.4 m | 3.1 m | 5.6 m | 96% | 49% |
| + SLAM landmark map | 1.4 m | 1.2 m | 0.9 m | 92% | 77% |
| + Lane map | 1.2 m | 0.6 m | 1.0 m | 98% | 97% |

With open-sky GPS the stack finishes at 1.1 m (97% right lane), and with harsh GPS (17 m raw
error) at 1.3 m (96%). Map matching alone cannot remove along-track error; only the landmark
map does.

`./evaluate.py --profile open-sky|urban-canyon|harsh` reproduces these numbers offline.

## Files

| File | Role |
| --- | --- |
| `app/roadmap.py` | OSM ways become directed road edges with per-direction lane counts and lane-centre offsets. Also produces the lane-map drawing. |
| `app/sim.py` | Route planning, the lane plan and speed profile, and the sensors: GPS (three quality profiles), wheel odometry and gyro, lane camera, and landmark detector. |
| `app/estimators.py` | The Kalman filter, HMM map matching, sparse Levenberg–Marquardt graph SLAM, fleet map fusion, the lane HMM, and scoring. |
| `app/server.py`, `app/index.html` | FastAPI server and the canvas viewer: tracks, delta lines, the Kalman ellipse, SLAM landmark rays, and an error-over-time chart. |
| `deploy.py` | Deploys the demo as a `chalkcompute.ScalingGroup` (CPU only). |
| `evaluate.py` | Offline scoring across many drives. |
| `fetch_map.py` | Refreshes the OpenStreetMap extract. |

## Run it

```bash
./deploy.py                     # deploys to the environment in the SDK’s ambient credentials from `chalk login`
./evaluate.py --drives 20       # offline, no server
```

Or locally: `cd app && uv run --with numpy --with scipy --with fastapi --with uvicorn python server.py`.
The server builds the fleet landmark map at startup (about 25 s for 60 drives) and then answers
each drive in under a second.

## What is simulated, and what is simplified

- **The roads are real; the drive is simulated.** Routes, lanes and one-way rules come from
  OpenStreetMap. The car's motion, speed, lane changes and every sensor reading are simulated
  from ground truth, which is what makes scoring possible.
- **Landmark data association is given.** The detector reports which landmark it saw. Real
  systems have to work this out, and getting it wrong is a large part of why production SLAM is
  hard.
- **The fleet map is fused, not jointly optimized.** Each mapping drive is solved on its own and
  the landmark estimates are averaged. A joint solve over all drives is more accurate, but it
  needs a better sparse solver than SciPy's for 60 drives.
- **Lane geometry is derived.** OSM gives lane counts, not lane paint, so lanes are offsets of
  the way's centreline at a 3.2 m width.

## License and credits

The code is Apache-2.0. Map data © [OpenStreetMap](https://www.openstreetmap.org/copyright)
contributors, available under the Open Database License (see `app/data/README.md`).

## Local authentication

The SDK reads credentials, API host, and the selected environment from your existing
`chalk login` in `~/.chalk.yml`, including project-scoped and default logins.
For unattended runs, `CHALK_CLIENT_ID` and `CHALK_CLIENT_SECRET` remain supported. Where available, `--env-id`
overrides the target environment; otherwise the SDK uses your login or environment settings.
