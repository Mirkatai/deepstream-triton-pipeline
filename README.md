# DeepStream Multi-Stream Triton Pipeline

A production-ready pipeline for GPU-accelerated video analysis:
- Multi-stream RTSP/MP4 input
- Frame-ordered Triton inference (Sequence Batcher)
- Synchronized overlay with hold (no flicker on slow models)
- Graceful fallback to passthrough on Triton failure
- Per-channel black-screen + recovery on stream loss
- Full GPU / RAM / CPU resource monitoring API

---

## Recommended Stack

| Component        | Version                      | Why                                                         |
|------------------|------------------------------|-------------------------------------------------------------|
| **DeepStream**   | **8.0**                      | First release with official **Blackwell** GPU support       |
| **Triton**       | **25.03-py3**                | Officially paired with DS 8.0 on x86; CUDA 12.8 + TRT 10.9 |
| **CUDA**         | 12.8                         | Full Blackwell SM_100 support                               |
| **TensorRT**     | 10.9                         | Blackwell-optimized inference kernels                       |
| **CuPy**         | cupy-cuda12x                 | GPU array ops inside Triton model                           |
| **NVIDIA Driver**| **≥ 570.133.20**             | Minimum required for DS 8.0                                 |
| **Base image**   | `nvcr.io/nvidia/deepstream:8.0-gc-triton-devel` | Includes Triton client + full DS SDK |

> **Supported GPUs**: Blackwell (RTX 50xx, GB200), Hopper (H100), Ada (RTX 40xx), Ampere (A100, RTX 30xx), Turing (T4, RTX 20xx)

> **Important DS 8.0 note**: The Docker container no longer bundles some multimedia libraries (audio parsing, CPU decode/encode). Run `/opt/nvidia/deepstream/deepstream/user_additional_install.sh` inside the container — this is handled automatically in our Dockerfile.

---

## Architecture

```
[RTSP/MP4 sources]
       │
  [nvstreammux]          ← batches N streams
       │
  [nvvideoconvert]
       │
  [nvdsosd]  ← GStreamer probe:
       │          1. Extract frame → submit to Triton (async, non-blocking)
       │          2. Get current overlay from OverlayStore (or hold last)
       │          3. Draw overlay via NvDsDisplayMeta
       │
  [nvmultistreamtiler]   ← grid layout
       │
  [nvvideoconvert]
       │
  [nvv4l2h264enc]        ← GPU encoder (Blackwell NVENC)
       │
  [rtph264pay]
       │
  [GstRtspServer]
       │
  rtsp://host:8554/live


[Background threads]
  TritonWorker × 2       ← async inference, sequence_id per stream
  OverlayStore           ← thread-safe, holds last overlay N frames
  ResourceMonitor        ← GPU/RAM/CPU polling via pynvml
  StreamHealthMonitor    ← watchdog → black screen + recovery on stream loss
```

---

## Frame Ordering in Triton

Triton's **Sequence Batcher** (`sequence_batching` in `config.pbtxt`) guarantees
that all requests with the same `sequence_id` are processed **in order**.

We map: `sequence_id = stream_id + 1`

So stream 0 → sequence 1, stream 1 → sequence 2, etc.
Frames within a stream always arrive in strict order at the model.

---

## Overlay Synchronization + Triton Latency Hiding

The probe **never blocks**. Triton runs fully asynchronously.

```
Frame arrives at GStreamer probe:
  → submit to Triton queue (non-blocking, drop if full — video has priority)
  → call overlay_store.get(stream_id)
      ├── new overlay ready → draw it
      └── no new overlay yet (Triton still processing)
              → reuse last overlay (hold for up to overlay_hold_frames)
```

With `time.sleep(0.1)` in the model and 25fps:
- Triton takes 100ms ≈ 2.5 frames
- `overlay_hold_frames=60` → overlay persists up to 2.4s
- Video never drops a frame, overlay stays stable

---

## Quick Start

### 1. Docker (recommended)

```bash
cd deepstream_solution/docker
docker-compose up --build

# Watch output:
ffplay rtsp://localhost:8554/live
# or
vlc rtsp://localhost:8554/live
```

### 2. Direct (if DeepStream 8.0 is installed on host)

```bash
pip install -r requirements.txt

# RTSP sources:
python3 main.py --sources rtsp://cam1/stream rtsp://cam2/stream

# Local files:
python3 main.py --sources /data/video1.mp4 /data/video2.mp4

# Mixed:
python3 main.py --sources rtsp://cam1/stream /data/local.mp4
```

### 3. Check resources anytime

```bash
# One-shot:
python3 resource_check.py

# Live watch (every 2s):
python3 resource_check.py --watch 2

# JSON output:
python3 resource_check.py --json

# From code:
from resource_check import get_resources
r = get_resources(gpu_id=0)
print(r["gpu"]["mem_used_mb"])
print(r["gpu"]["encoder_util_pct"])
print(r["gpu"]["temperature_c"])
```

---

## Adding Your Own Analysis

1. Edit `triton_model/frame_analyzer/1/model.py`
2. Replace `_analyze_gpu()` with your module's logic
3. **Input**: `frame_gpu` — CuPy array `(H, W, 3)` uint8 on GPU
4. **Output**: numpy array `(N, 6)` float32 — `[x1, y1, x2, y2, class_id, conf]`

See `example_custom_analysis.py` for:
- Motion detection (CuPy frame differencing)
- Scene change detection (color histogram)
- ONNX YOLOv8 integration example

---

## Failure Modes

| Failure                   | Behavior                                           |
|---------------------------|----------------------------------------------------|
| Triton unreachable        | Pipeline runs as clean passthrough (no overlay)    |
| Triton model error        | Same — passthrough, error logged                   |
| Stream drops              | Black screen shown, recovery thread retries every 5s |
| GStreamer error            | Warning logged, pipeline continues (passthrough=True) |
| GPU OOM                   | Triton drops excess frames via queue backpressure  |

---

## Resource Management Tips

| Setting                   | Recommended                                         |
|---------------------------|-----------------------------------------------------|
| `overlay_buffer_size`     | 8 — drop excess Triton requests when GPU is busy    |
| `overlay_hold_frames`     | 60 at 25fps = 2.4s hold                            |
| Triton `instance_group`   | 2 per GPU, increase for more parallel streams       |
| `batched-push-timeout`    | 33333 µs (~30fps)                                   |
| `nvv4l2h264enc preset`    | 1 = ultra-fast / low latency                        |
| Docker `shm_size`         | 2GB (Triton Python backend shared memory)           |
| NVIDIA Driver             | ≥ 570.133.20 (required for DS 8.0 + Blackwell)     |

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
│       ├── config.pbtxt             ← Triton config (sequence batcher)
│       └── 1/
│           └── model.py             ← Python backend (CuPy arrays, time.sleep demo)
└── docker/
    ├── Dockerfile                   ← DeepStream 8.0 + Blackwell ready
    └── docker-compose.yml           ← Triton 25.03 + DeepStream 8.0 full stack
```
