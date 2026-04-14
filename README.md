# Real-time People Counting System

YOLOv8 + ByteTrack + Virtual Line Crossing — with a Streamlit UI.

---

## Project Structure

```
AI_DETECTION/
├── main.py          # Core pipeline (CLI + importable generator)
├── tracker.py       # YOLOv8 + ByteTrack wrapper → TrackedPerson objects
├── counter.py       # Line-crossing logic, double-count prevention
├── ui.py            # Streamlit web interface
├── utils.py         # Drawing helpers, video utilities, data structures
└── requirements.txt
```

---

## Quick Start

### 1. Install dependencies

```bash
# (recommended) create and activate a virtual environment
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

> The first run will automatically download `yolov8n.pt` (~6 MB) from
> Ultralytics if it is not already present.

---

### 2a. Streamlit UI (recommended)

```bash
streamlit run ui.py
```

Open `http://localhost:8501` in your browser, upload a video, and click
**▶ Start Processing**.

---

### 2b. Command-line

```bash
# Test video (side-view street — use vertical line)
python main.py --source test_video.mp4 --line-axis vertical --output output.mp4

# Webcam (device 0)
python main.py --source 0

# Top-down / entrance camera (use horizontal line)
python main.py --source entrance.mp4 --line-axis horizontal --output output.mp4

# More accurate model, higher confidence
python main.py --source test_video.mp4 --line-axis vertical --model yolov8m.pt --conf 0.5

# Headless (no GUI window) — useful on servers
python main.py --source test_video.mp4 --line-axis vertical --no-display --output result.mp4
```

> **Tip — choose the right line orientation:**
> - **vertical** line → for side-view cameras where people walk left/right (e.g. `test_video.mp4`)
> - **horizontal** line → for top-down or front-facing cameras where people walk toward/away

Full option list:

| Flag | Default | Description |
|------|---------|-------------|
| `--source` | `0` | Video path or webcam index |
| `--model` | `yolov8n.pt` | Model weights (n/s/m/l/x) |
| `--output` | _(none)_ | Save annotated video |
| `--line-ratio` | `0.50` | Line position (0–1 fraction) |
| `--line-axis` | `horizontal` | `horizontal` or `vertical` |
| `--in-dir` | `1` | `+1` or `-1` for IN direction |
| `--conf` | `0.35` | Detection confidence threshold |
| `--iou` | `0.50` | NMS IoU threshold |
| `--max-dim` | `1280` | Max frame dimension (px) |
| `--device` | _(auto)_ | `cuda` / `cpu` / `mps` |
| `--no-display` | _(off)_ | Headless mode |

---

## How it works

```
Video frame
    │
    ▼
PeopleTracker.track_frame()          tracker.py
  └─ YOLO.track(persist=True)        ← ByteTrack built into Ultralytics
  └─ Returns List[TrackedPerson]      ← (id, bbox, conf, centroid)
    │
    ▼
LineCrossingCounter.update()         counter.py
  └─ For each person:
       cross = (line_end−line_start) × (centroid−line_start)
       sign(cross) → current_side
       prev_side → current_side transition → IN / OUT
       Cooldown window prevents double-counts
    │
    ▼
Draw annotations                     utils.py
  └─ Bounding boxes + IDs
  └─ Counting line
  └─ HUD (IN / OUT / FPS)
    │
    ▼
Yield annotated frame → UI / CLI     main.py
```

### Double-count prevention

1. **Side tracking** — each track ID records which side of the line it was
   on in the *previous* frame.  Only a side *change* triggers a count.
2. **Cooldown** — after a crossing, the ID is locked for `cooldown_frames`
   (default 10) frames before another crossing can be registered.
3. **Stale-ID cleanup** — IDs absent for > `max_absent_frames` (default 30)
   frames are evicted, freeing memory.

---

## Performance tips

| Goal | Tip |
|------|-----|
| Faster inference | Use `yolov8n.pt` (default) + lower `--max-dim 640` |
| Better accuracy | Use `yolov8m.pt` or `yolov8l.pt` |
| GPU acceleration | Install `torch` with CUDA, set `--device cuda` |
| Reduce false positives | Raise `--conf` to 0.5–0.6 |
| Crowded scenes | Lower `--iou` to 0.3–0.4 to keep overlapping boxes |
