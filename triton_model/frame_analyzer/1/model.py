"""
Triton Python Backend Model: frame_analyzer
============================================
- Receives raw RGB frame as UINT8 numpy array
- Converts to CuPy array on GPU — zero-copy via DLPack
- Runs analysis (replace stub with real model)
- Returns overlay_data: shape (N, 6) float32 [x1,y1,x2,y2,class_id,conf]
- Demonstrates:
    * CuPy GPU arrays
    * Proper sequence_id ordering (Triton handles this via ensemble or stateful)
    * time.sleep(0.1) to show overlay hold still works perfectly
"""

import numpy as np
import json
import triton_python_backend_utils as pb_utils
import time
import logging

log = logging.getLogger("triton.frame_analyzer")

# Lazy imports — loaded once at model init
_cupy_available = False
try:
    import cupy as cp
    _cupy_available = True
except ImportError:
    log.warning("CuPy not available — falling back to NumPy on CPU.")


class TritonPythonModel:

    def initialize(self, args):
        """
        Called once when the model loads.
        Load your weights, warm up GPU, etc.
        """
        self.model_config = json.loads(args["model_config"])
        self.device_id = int(args.get("model_instance_device_id", 0))

        if _cupy_available:
            cp.cuda.Device(self.device_id).use()
            # Pre-allocate a small workspace to warm up CuPy
            _ = cp.zeros((1, 3, 64, 64), dtype=cp.float32)
            log.info(f"CuPy ready on device {self.device_id}")

        # === Load your real model here ===
        # Example: TensorRT engine, ONNX, PyTorch traced model...
        # self.engine = load_tensorrt_engine("/models/yolov8.engine")
        log.info("frame_analyzer model initialized.")

    def execute(self, requests):
        """
        Called for each batch of requests.
        Each request contains one frame from one stream.
        """
        responses = []

        for request in requests:
            # --- Get input ---
            input_tensor = pb_utils.get_input_tensor_by_name(request, "input_frame")
            frame_np = input_tensor.as_numpy()   # (H, W, 3) uint8, CPU

            # --- Move to GPU as CuPy array ---
            if _cupy_available:
                frame_gpu = cp.asarray(frame_np)  # zero-copy if pinned, else H2D
                detections = self._analyze_gpu(frame_gpu)
            else:
                detections = self._analyze_cpu(frame_np)

            # --- Return as float32 array (N, 6) ---
            out_tensor = pb_utils.Tensor(
                "overlay_data",
                detections.astype(np.float32)
            )
            responses.append(pb_utils.InferenceResponse(output_tensors=[out_tensor]))

        return responses

    def _analyze_gpu(self, frame_gpu):
        """
        GPU analysis using CuPy arrays.

        frame_gpu: cupy.ndarray shape (H, W, 3) uint8

        This is where you plug in your real detector.
        Demonstrates: time.sleep(0.1) → pipeline still smooth because
        overlay is held from last frame, not blocked.
        """

        # ==== EXAMPLE: Simulate processing delay (0.1s) ====
        # This would cause dropped frames if inline — but here it's fine
        # because Triton runs async and the pipeline holds the last overlay.
        time.sleep(0.1)

        H, W = frame_gpu.shape[:2]

        # ==== Stub detection: fake a single centered box ====
        # Replace with your real model inference:
        # e.g.:  output = self.engine.infer(frame_gpu)
        #        detections = postprocess(output, H, W)

        # Normalize to float on GPU
        frame_float = frame_gpu.astype(cp.float32) / 255.0   # (H,W,3)

        # Compute simple brightness metric as a "detection score" (demo)
        mean_brightness = float(cp.mean(frame_float))

        if mean_brightness > 0.05:  # scene is not black
            cx, cy = W // 2, H // 2
            bw, bh = W // 4, H // 4
            x1, y1 = cx - bw // 2, cy - bh // 2
            x2, y2 = cx + bw // 2, cy + bh // 2
            class_id = 0
            confidence = round(mean_brightness, 3)
            detections_np = np.array([[x1, y1, x2, y2, class_id, confidence]], dtype=np.float32)
        else:
            detections_np = np.zeros((0, 6), dtype=np.float32)

        # Free GPU temp arrays explicitly (good practice)
        del frame_float
        cp.get_default_memory_pool().free_all_blocks()

        return detections_np

    def _analyze_cpu(self, frame_np):
        """CPU fallback when CuPy is unavailable."""
        time.sleep(0.1)
        H, W = frame_np.shape[:2]
        mean_b = np.mean(frame_np) / 255.0
        if mean_b > 0.05:
            cx, cy = W // 2, H // 2
            bw, bh = W // 4, H // 4
            return np.array([[cx-bw//2, cy-bh//2, cx+bw//2, cy+bh//2, 0, mean_b]], dtype=np.float32)
        return np.zeros((0, 6), dtype=np.float32)

    def finalize(self):
        """Called when the model is unloaded."""
        if _cupy_available:
            cp.get_default_memory_pool().free_all_blocks()
        log.info("frame_analyzer finalized.")
