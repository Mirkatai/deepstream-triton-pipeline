# ============================================================
# Recommended base: DeepStream 6.4 on NGC
# Why 6.4?
#   - Python bindings (pyds) are stable and well-documented
#   - nvv4l2decoder / nvv4l2h264enc are mature
#   - Compatible with Triton 23.10 (same CUDA 12.2)
#   - GStreamer 1.20 — nvmultistreamtiler, nvdsosd work reliably
#   - CUDA 12.2 + cuDNN 8.9 (matches CuPy 12.x)
#
# Alternative: DeepStream 7.0 (if you need NvDRM / RTSP 2.0 support)
#   nvcr.io/nvidia/deepstream:7.0-gc-triton-devel
# ============================================================

FROM nvcr.io/nvidia/deepstream:6.4-gc-triton-devel

# --- System deps ---
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip \
    python3-dev \
    python3-gi \
    python3-gi-cairo \
    gir1.2-gst-rtsp-server-1.0 \
    libgstrtspserver-1.0-0 \
    libgirepository1.0-dev \
    gcc \
    curl \
    && rm -rf /var/lib/apt/lists/*

# --- Python packages ---
COPY requirements.txt /tmp/requirements.txt
RUN pip3 install --no-cache-dir -r /tmp/requirements.txt

# --- CuPy for CUDA 12 ---
RUN pip3 install --no-cache-dir cupy-cuda12x

# --- DeepStream Python bindings ---
# Already included in the gc-triton-devel image at:
# /opt/nvidia/deepstream/deepstream/lib/pyds.so
RUN pip3 install \
    /opt/nvidia/deepstream/deepstream/lib/pyds*.whl 2>/dev/null || \
    echo "pyds wheel not found, using .so directly"

# --- Set Python path for pyds ---
ENV PYTHONPATH="${PYTHONPATH}:/opt/nvidia/deepstream/deepstream/lib"
ENV GST_PLUGIN_PATH="/opt/nvidia/deepstream/deepstream/lib/gst-plugins"
ENV LD_LIBRARY_PATH="/opt/nvidia/deepstream/deepstream/lib:${LD_LIBRARY_PATH}"
ENV DISPLAY=:0

# --- Copy application ---
WORKDIR /app
COPY . /app/

# --- Triton model repository ---
ENV TRITON_MODEL_REPO=/models
RUN mkdir -p /models/frame_analyzer
COPY triton_model/frame_analyzer /models/frame_analyzer/

CMD ["python3", "main.py"]
