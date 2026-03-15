# ============================================================
# DeepStream 8.0 + Triton 25.03 — Blackwell / x86 dGPU
# ============================================================
# Why DeepStream 8.0?
#   - First release with official Blackwell GPU support
#   - Paired with Triton 25.03 (CUDA 12.8, TensorRT 10.9)
#   - GStreamer 1.20+ with nvdsdynamicsrcbin (dynamic stream add)
#   - Python bindings (pyds) stable, pyservicemaker available
#   - Supports GB200, RTX 50xx, H100, A100, T4 all in one image
#
# NVIDIA Driver requirement: 570.133.20 or later
# ============================================================

FROM nvcr.io/nvidia/deepstream:8.0-gc-triton-devel

# --- DS 8.0 requires extra GStreamer plugins for MP4/audio ---
# Run the official script that installs gstreamer1.0-libav,
# gstreamer1.0-plugins-good/bad/ugly etc.
RUN /opt/nvidia/deepstream/deepstream/user_additional_install.sh || true

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
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# --- Python packages ---
COPY requirements.txt /tmp/requirements.txt
RUN pip3 install --no-cache-dir -r /tmp/requirements.txt

# --- CuPy for CUDA 12.8 (Blackwell / DS 8.0 stack) ---
RUN pip3 install --no-cache-dir cupy-cuda12x

# --- DeepStream Python bindings ---
# In DS 8.0 the .whl is at /opt/nvidia/deepstream/deepstream/lib/
RUN pip3 install \
    /opt/nvidia/deepstream/deepstream/lib/pyds*.whl 2>/dev/null || \
    echo "pyds wheel not found, using .so path directly"

# --- Environment ---
ENV PYTHONPATH="${PYTHONPATH}:/opt/nvidia/deepstream/deepstream/lib"
ENV GST_PLUGIN_PATH="/opt/nvidia/deepstream/deepstream/lib/gst-plugins"
ENV LD_LIBRARY_PATH="/opt/nvidia/deepstream/deepstream/lib:${LD_LIBRARY_PATH}"
# Required for video driver capabilities (encoder/decoder) in DS 8.0
ENV NVIDIA_DRIVER_CAPABILITIES="${NVIDIA_DRIVER_CAPABILITIES},video"
ENV DISPLAY=:0

# --- Copy application ---
WORKDIR /app
COPY . /app/

# --- Triton model repository ---
ENV TRITON_MODEL_REPO=/models
RUN mkdir -p /models/frame_analyzer
COPY triton_model/frame_analyzer /models/frame_analyzer/

CMD ["python3", "main.py"]
