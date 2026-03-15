"""
Example: How to write your own analysis in the Triton model.py
================================================================
Demonstrates:
  1. time.sleep(0.1) — pipeline stays smooth (overlay held from last frame)
  2. Motion detection using CuPy
  3. Simple color histogram
  4. How to integrate a real ONNX model

Drop the relevant _analyze_gpu() method into your model.py.
"""

import numpy as np
import time

try:
    import cupy as cp
    CUPY_OK = True
except ImportError:
    CUPY_OK = False


# ============================================================
# EXAMPLE 1:  Simulated heavy analysis with time.sleep(0.1)
# Shows that even 100ms processing doesn't stutter the video.
# The last valid overlay persists on screen during the delay.
# ============================================================

class Example1_SleepDemo:
    def analyze(self, frame_gpu):
        """Simulate 100ms analysis — pipeline stays smooth."""
        time.sleep(0.1)   # This is the key line from your requirement

        H, W = frame_gpu.shape[:2]
        # Just return a static box in the center
        cx, cy = W // 2, H // 2
        return np.array([
            [cx - 100, cy - 50, cx + 100, cy + 50, 99, 0.99]
        ], dtype=np.float32)


# ============================================================
# EXAMPLE 2:  Motion Detection (frame differencing on GPU)
# ============================================================

class Example2_MotionDetector:
    def __init__(self):
        self._prev_frame = None

    def analyze(self, frame_gpu):
        """
        Returns bounding box around motion region.
        frame_gpu: cupy array (H, W, 3) uint8
        """
        gray = cp.mean(frame_gpu.astype(cp.float32), axis=2)  # (H, W)

        detections = np.zeros((0, 6), dtype=np.float32)

        if self._prev_frame is not None and self._prev_frame.shape == gray.shape:
            diff = cp.abs(gray - self._prev_frame)
            motion_mask = diff > 30.0   # threshold

            motion_pct = float(cp.mean(motion_mask))

            if motion_pct > 0.01:  # at least 1% of pixels moved
                ys, xs = cp.where(motion_mask)
                if len(xs) > 0:
                    x1 = int(cp.min(xs))
                    y1 = int(cp.min(ys))
                    x2 = int(cp.max(xs))
                    y2 = int(cp.max(ys))
                    # class_id=1 (motion), conf = motion percentage
                    detections = np.array([[x1, y1, x2, y2, 1, motion_pct]], dtype=np.float32)

        self._prev_frame = gray.copy()
        return detections


# ============================================================
# EXAMPLE 3:  Color histogram (simple scene change detection)
# ============================================================

class Example3_SceneChange:
    def __init__(self, threshold: float = 0.4):
        self._prev_hist = None
        self.threshold = threshold

    def analyze(self, frame_gpu):
        """
        Flags a scene change with a full-frame box (class_id=2).
        """
        H, W = frame_gpu.shape[:2]

        # Compute normalized color histogram on GPU (fast)
        r = frame_gpu[:, :, 0].ravel().astype(cp.float32) / 255.0
        hist = cp.histogram(r, bins=64, range=(0.0, 1.0))[0].astype(cp.float32)
        hist = hist / (cp.sum(hist) + 1e-6)

        change = 0.0
        if self._prev_hist is not None:
            change = float(cp.sum(cp.abs(hist - self._prev_hist)))

        self._prev_hist = hist.copy()

        if change > self.threshold:
            return np.array([[0, 0, W, H, 2, change]], dtype=np.float32)
        return np.zeros((0, 6), dtype=np.float32)


# ============================================================
# EXAMPLE 4:  Real ONNX model (e.g. YOLOv8 nano)
# ============================================================

class Example4_OnnxYolo:
    def __init__(self, model_path: str = "/models/yolov8n.onnx"):
        import onnxruntime as ort
        self.session = ort.InferenceSession(
            model_path,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = (640, 640)

    def preprocess(self, frame_gpu):
        """Resize + normalize on GPU using CuPy."""
        import cupy as cp
        from cupyx.scipy.ndimage import zoom as cp_zoom

        H, W = frame_gpu.shape[:2]
        th, tw = self.input_shape
        scale_h = th / H
        scale_w = tw / W

        frame_small = cp_zoom(frame_gpu, (scale_h, scale_w, 1), order=1)
        frame_float = frame_small.astype(cp.float32) / 255.0
        frame_chw = cp.transpose(frame_float, (2, 0, 1))           # (3, H, W)
        frame_bchw = cp.expand_dims(frame_chw, 0)                  # (1, 3, H, W)
        return cp.asnumpy(frame_bchw), scale_h, scale_w            # back to CPU for ONNX

    def analyze(self, frame_gpu):
        inp_np, sh, sw = self.preprocess(frame_gpu)
        outputs = self.session.run(None, {self.input_name: inp_np})

        # YOLOv8 output: (1, 84, 8400) — 4 box + 80 classes
        preds = outputs[0][0].T  # (8400, 84)
        boxes_xywh = preds[:, :4]
        scores = preds[:, 4:]
        class_ids = np.argmax(scores, axis=1)
        confidences = scores[np.arange(len(scores)), class_ids]

        mask = confidences > 0.5
        results = []
        for (x, y, w, h), cls, conf in zip(
            boxes_xywh[mask], class_ids[mask], confidences[mask]
        ):
            x1 = (x - w / 2) / sw
            y1 = (y - h / 2) / sh
            x2 = (x + w / 2) / sw
            y2 = (y + h / 2) / sh
            results.append([x1, y1, x2, y2, float(cls), float(conf)])

        if results:
            return np.array(results, dtype=np.float32)
        return np.zeros((0, 6), dtype=np.float32)


# ============================================================
# HOW TO USE in model.py
# ============================================================
#
# In TritonPythonModel.initialize():
#     from example_custom_analysis import Example2_MotionDetector
#     self.detector = Example2_MotionDetector()
#
# In TritonPythonModel._analyze_gpu():
#     return self.detector.analyze(frame_gpu)
#
# The time.sleep(0.1) from Example1 is already in the stub model.
# With overlay_hold_frames=60 (default), the last valid detection
# box stays on screen for up to 60 frames while the new one processes.
# ============================================================
