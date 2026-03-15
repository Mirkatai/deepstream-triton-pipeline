# DeepStream Multi-Stream Triton Pipeline

A production-ready pipeline for GPU-accelerated video analysis:
- Multi-stream RTSP/MP4 input
- Frame-ordered Triton inference
- Synchronized overlay with hold (no flicker on slow models)
- Graceful fallback to passthrough
- Per-channel black-screen + recovery on stream loss
- Full resource monitoring API

---

## Recommended Stack

| Component        | Version         | Why                                                  |
|------------------|-----------------|------------------------------------------------------|
| **DeepStream**   | **6.4**         | Stable Python bindings, nvv4l2, GStreamer 1.20       |
| **Triton**       | **23.10-py3**   | CUDA 12.2 match, Python backend, sequence batcher    |
| **CUDA**         | 12.2            | Matches both above                                   |
| **CuPy**         | cupy-cuda12x    | GPU array operations inside Triton                   |
| **Base image**   | `nvcr.io/nvidia/deepstream:6.4-gc-triton-devel` | Includes Triton client + DS |

> **Alternative**: Use `deepstream:7.0-gc-triton-devel` if you need RTSP 2.0, NvDRM, or newer GStreamer 1.22 features.

---

## Architecture

```
[RTSP/MP4 sources]
       │
  [nvstreammux]          ← batches N streams
       │
  [nvvideoconvert]
       │
  [nvdsosd]  ← GStreamer probe here:
       │          1. Extract frame → submit to Triton (async, non-blocking)
       │          2. Get current overlay from OverlayStore (or hold last)
       │          3. Draw overlay via NvDsDisplayMeta
       │
  [nvmultistreamtiler]   ← grid layout
       │
  [nvvideoconvert]
       │
  [nvv4l2h264enc]        ← GPU encoder
       │
  [rtph264pay]
       │
  [udpsink / GstRtspServer]
       │
  rtsp://host:8554/live


[Background threads]
  TritonWorker × 2       ← async inference, sequence_id per stream
  OverlayStore           ← thread-safe, hold last overlay N frames
  ResourceMonitor        ← GPU/RAM/CPU polling via pynvml
  StreamHealthMonitor    ← watchdog, triggers black screen + recovery
```

---

## Frame Ordering in Triton

Triton's **Sequence Batcher** (`sequence_batching` in config.pbtxt) guarantees
that all requests with the same `sequence_id` are processed **in order**.

We map:  `sequence_id = stream_id + 1`

So stream 0 → sequence 1, stream 1 → sequence 2, etc.
Frames within a stream always arrive in order at the model.

---

## Overlay Synchronization

The key insight: **Triton runs asynchronously**. The probe never blocks.

```
Frame arrives at probe:
  → submit to Triton queue (non-blocking, drop if full)
  → call overlay_store.get(stream_id)
      ├── if new overlay ready → draw it
      └── if no new overlay (Triton still processing)
              → reuse last overlay (hold for up to overlay_hold_frames)
```

With `time.sleep(0.1)` in the model and 25fps video:
- Triton takes 100ms = ~2.5 frames
- `overlay_hold_frames=60` means overlay persists 2.4 seconds
- Video never drops, overlay stays stable until next result

---

## Quick Start

### 1. Build and run

```bash
# Clone / copy this directory
cd deepstream_solution/docker

# Set your sources in docker-compose.yml, then:
docker-compose up --build

# Watch output:
ffplay rtsp://localhost:8554/live
# or
vlc rtsp://localhost:8554/live
```

### 2. Run directly (if DeepStream is installed)

```bash
pip install -r requirements.txt

# With RTSP sources:
python3 main.py --sources rtsp://cam1/stream rtsp://cam2/stream

# With local files:
python3 main.py --sources /data/video1.mp4 /data/video2.mp4

# Mixed:
python3 main.py --sources rtsp://cam1/stream /data/local.mp4
```

### 3. Check resources anytime

```bash
# One-shot report:
python3 resource_check.py

# Live monitor:
python3 resource_check.py --watch 2

# JSON output (for integration):
python3 resource_check.py --json

# From code:
from resource_check import get_resources
r = get_resources(gpu_id=0)
print(r["gpu"]["mem_used_mb"])
print(r["gpu"]["encoder_util_pct"])
```

---

## Adding Your Own Analysis

1. Edit `triton_model/frame_analyzer/1/model.py`
2. In `_analyze_gpu()`, replace the stub with your logic
3. Input: `frame_gpu` — CuPy array `(H, W, 3)` uint8 on GPU
4. Output: numpy array `(N, 6)` float32 — `[x1, y1, x2, y2, class_id, conf]`

See `example_custom_analysis.py` for:
- Motion detection
- Scene change detection
- ONNX YOLOv8 integration

---

## Failure Modes

| Failure                   | Behavior                                           |
|---------------------------|----------------------------------------------------|
| Triton unreachable        | Pipeline runs as clean passthrough (no overlay)    |
| Triton model error        | Same — passthrough, error logged                   |
| Stream drops              | Black screen shown, recovery thread retries every 5s |
| GStreamer error            | Warning logged, pipeline continues if passthrough=True |
| GPU OOM                   | Triton drops excess frames (queue backpressure)    |

---

## Resource Management Tips

| Setting                   | Recommended                                        |
|---------------------------|----------------------------------------------------|
| `overlay_buffer_size`     | 8 (drop excess Triton requests when busy)          |
| `overlay_hold_frames`     | 60 at 25fps = 2.4s hold                           |
| Triton `instance_group`   | 2 per GPU for your stream count                    |
| `batched-push-timeout`    | 33333 µs (~30fps)                                  |
| `nvv4l2h264enc preset`    | 1 = ultra-fast (low latency)                       |
| Docker `shm_size`         | 2GB (Triton Python backend shared memory)          |

---

## Files

```
deepstream_solution/
├── main.py                          ← entry point (argparse)
├── pipeline.py                      ← full GStreamer pipeline
├── resource_check.py                ← standalone resource monitor
├── example_custom_analysis.py       ← analysis examples (CuPy, ONNX, etc.)
├── requirements.txt
├── triton_model/
│   └── frame_analyzer/
│       ├── config.pbtxt             ← Triton model config (sequence batcher)
│       └── 1/
│           └── model.py             ← Python backend (CuPy, time.sleep demo)
└── docker/
    ├── Dockerfile                   ← DeepStream 6.4 image
    └── docker-compose.yml           ← Full stack (Triton + DeepStream)
```
