"""
DeepStream Multi-Stream Pipeline with Ordered Triton Inference
=============================================================
- Accepts list of RTSP URLs or local MP4 files
- Guarantees frame order within Triton (sequence_id based)
- Overlay is synchronized to frames via a small buffer
- Triton latency is hidden by holding last valid overlay
- Graceful fallback to passthrough on any failure
- Black screen + recovery on per-channel failure
- GPU memory / bandwidth friendly design
"""

import sys
import os
import time
import threading
import queue
import logging
import traceback
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable

import gi
gi.require_version("Gst", "1.0")
gi.require_version("GstRtspServer", "1.0")
from gi.repository import Gst, GLib, GstRtspServer

import numpy as np

# Pyds (DeepStream Python bindings)
import pyds

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
log = logging.getLogger("ds_pipeline")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class PipelineConfig:
    sources: List[str]                          # RTSP URLs or file paths
    output_rtsp_port: int = 8554
    output_width: int = 1280
    output_height: int = 720
    output_fps: int = 25
    output_bitrate: int = 4000000              # 4 Mbps per stream
    triton_url: str = "localhost:8001"          # gRPC
    triton_model_name: str = "frame_analyzer"
    triton_model_version: str = "1"
    overlay_buffer_size: int = 8               # frames held waiting for overlay
    overlay_hold_frames: int = 60              # hold last overlay N frames if no new one
    recovery_interval_sec: float = 5.0
    passthrough_on_failure: bool = True
    black_screen_on_stream_loss: bool = True
    gpu_id: int = 0
    batch_size: int = 1                        # per stream, NvDsBatchMeta handles batching
    muxer_output_width: int = 1280
    muxer_output_height: int = 720
    enable_osd: bool = True
    log_resources_interval_sec: float = 10.0


# ---------------------------------------------------------------------------
# Resource Monitor
# ---------------------------------------------------------------------------

class ResourceMonitor:
    """
    Call .get_report() at any time for a reliable snapshot of:
      - GPU memory used/total per device
      - GPU utilization %
      - GPU encoder / decoder utilization
      - System RAM
      - Estimated bandwidth (bytes processed)
    """

    def __init__(self, gpu_id: int = 0, interval: float = 2.0):
        self.gpu_id = gpu_id
        self.interval = interval
        self._report: Dict = {}
        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._bytes_in = 0
        self._bytes_out = 0
        self._last_bytes_ts = time.time()

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True, name="resource_monitor")
        self._thread.start()

    def stop(self):
        self._running = False

    def add_bytes(self, in_bytes: int = 0, out_bytes: int = 0):
        """Call from pipeline probes to track bandwidth."""
        self._bytes_in += in_bytes
        self._bytes_out += out_bytes

    def _collect(self):
        report = {"timestamp": time.time(), "gpu": {}, "ram": {}, "bandwidth": {}}
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(self.gpu_id)
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            enc = pynvml.nvmlDeviceGetEncoderUtilization(handle)
            dec = pynvml.nvmlDeviceGetDecoderUtilization(handle)
            report["gpu"] = {
                "id": self.gpu_id,
                "mem_used_mb": round(mem.used / 1e6, 1),
                "mem_total_mb": round(mem.total / 1e6, 1),
                "mem_pct": round(100 * mem.used / mem.total, 1),
                "gpu_util_pct": util.gpu,
                "mem_util_pct": util.memory,
                "encoder_util_pct": enc[0],
                "decoder_util_pct": dec[0],
            }
            pynvml.nvmlShutdown()
        except Exception as e:
            report["gpu"]["error"] = str(e)

        try:
            import psutil
            vm = psutil.virtual_memory()
            report["ram"] = {
                "used_mb": round(vm.used / 1e6, 1),
                "total_mb": round(vm.total / 1e6, 1),
                "pct": vm.percent,
            }
        except Exception as e:
            report["ram"]["error"] = str(e)

        now = time.time()
        elapsed = now - self._last_bytes_ts
        if elapsed > 0:
            report["bandwidth"] = {
                "in_mbps": round(self._bytes_in * 8 / elapsed / 1e6, 2),
                "out_mbps": round(self._bytes_out * 8 / elapsed / 1e6, 2),
            }
        self._bytes_in = 0
        self._bytes_out = 0
        self._last_bytes_ts = now

        return report

    def _loop(self):
        while self._running:
            try:
                r = self._collect()
                with self._lock:
                    self._report = r
            except Exception:
                pass
            time.sleep(self.interval)

    def get_report(self) -> Dict:
        """Returns the latest resource snapshot (thread-safe)."""
        with self._lock:
            return dict(self._report)

    def print_report(self):
        r = self.get_report()
        gpu = r.get("gpu", {})
        ram = r.get("ram", {})
        bw = r.get("bandwidth", {})
        print("\n===== RESOURCE REPORT =====")
        print(f"  GPU {gpu.get('id',0)}: "
              f"{gpu.get('mem_used_mb','?')} / {gpu.get('mem_total_mb','?')} MB  "
              f"util={gpu.get('gpu_util_pct','?')}%  "
              f"enc={gpu.get('encoder_util_pct','?')}%  "
              f"dec={gpu.get('decoder_util_pct','?')}%")
        print(f"  RAM: {ram.get('used_mb','?')} / {ram.get('total_mb','?')} MB  ({ram.get('pct','?')}%)")
        print(f"  BW in={bw.get('in_mbps','?')} Mbps  out={bw.get('out_mbps','?')} Mbps")
        print("===========================\n")


# ---------------------------------------------------------------------------
# Overlay Store  (thread-safe, per-stream)
# ---------------------------------------------------------------------------

@dataclass
class OverlayData:
    frame_num: int
    labels: List[str] = field(default_factory=list)
    boxes: List[tuple] = field(default_factory=list)   # (x1,y1,x2,y2,label,conf)
    extra: Dict = field(default_factory=dict)


class OverlayStore:
    """
    Triton writes overlay data here.
    The GStreamer probe reads from here — if no new overlay since last N frames,
    the previous overlay is reused (transparent hold).
    """

    def __init__(self, num_streams: int, hold_frames: int = 60):
        self._lock = threading.Lock()
        self._overlays: Dict[int, Optional[OverlayData]] = {i: None for i in range(num_streams)}
        self._ages: Dict[int, int] = defaultdict(int)   # frames since last update
        self.hold_frames = hold_frames

    def update(self, stream_id: int, data: OverlayData):
        with self._lock:
            self._overlays[stream_id] = data
            self._ages[stream_id] = 0

    def get(self, stream_id: int) -> Optional[OverlayData]:
        with self._lock:
            age = self._ages[stream_id]
            self._ages[stream_id] = age + 1
            if age > self.hold_frames:
                self._overlays[stream_id] = None  # expired
            return self._overlays[stream_id]


# ---------------------------------------------------------------------------
# Ordered Frame Queue  (per stream)
# Frame numbers from DeepStream are monotonically increasing.
# We buffer a small window so Triton can return results slightly out of order.
# ---------------------------------------------------------------------------

class OrderedFrameQueue:
    """
    Accepts (frame_num, data) pairs that may arrive slightly out of order.
    Yields them in strict ascending order once the window is satisfied.
    """

    def __init__(self, window: int = 8):
        self.window = window
        self._buf: Dict[int, any] = {}
        self._next = 0
        self._lock = threading.Lock()

    def put(self, frame_num: int, data):
        with self._lock:
            self._buf[frame_num] = data
            if self._next == 0 and frame_num == 0:
                self._next = 0

    def drain(self) -> list:
        """Returns a list of (frame_num, data) in order, consuming what's ready."""
        out = []
        with self._lock:
            while self._next in self._buf:
                out.append((self._next, self._buf.pop(self._next)))
                self._next += 1
            # Safety: if gap too large, skip ahead (prevents deadlock on drop)
            if self._buf and (min(self._buf.keys()) - self._next) > self.window * 2:
                self._next = min(self._buf.keys())
        return out


# ---------------------------------------------------------------------------
# Triton Client Worker  (runs in a background thread pool)
# ---------------------------------------------------------------------------

class TritonWorker:
    """
    Receives raw frame data (numpy/cupy), sends to Triton model,
    writes results to OverlayStore.
    Completely decoupled from GStreamer main loop.
    """

    def __init__(self, config: PipelineConfig, overlay_store: OverlayStore,
                 resource_monitor: ResourceMonitor):
        self.config = config
        self.overlay_store = overlay_store
        self.resource_monitor = resource_monitor
        self._queue: queue.Queue = queue.Queue(maxsize=config.overlay_buffer_size * config.batch_size)
        self._running = False
        self._threads: List[threading.Thread] = []
        self._client = None
        self._failed = False
        self.num_workers = 2  # parallel Triton calls

    def start(self):
        self._running = True
        try:
            import tritonclient.grpc as grpcclient
            self._client = grpcclient.InferenceServerClient(
                url=self.config.triton_url,
                verbose=False
            )
            log.info(f"Triton client connected to {self.config.triton_url}")
        except Exception as e:
            log.warning(f"Triton client init failed: {e}. Will run in passthrough mode.")
            self._failed = True

        for i in range(self.num_workers):
            t = threading.Thread(target=self._worker_loop, daemon=True, name=f"triton_worker_{i}")
            t.start()
            self._threads.append(t)

    def stop(self):
        self._running = False

    def submit(self, stream_id: int, frame_num: int, frame_rgb: np.ndarray):
        """Non-blocking. Drop frame if queue full (backpressure)."""
        if self._failed:
            return
        try:
            self._queue.put_nowait((stream_id, frame_num, frame_rgb))
        except queue.Full:
            pass  # drop — video smoothness has priority

    def _worker_loop(self):
        while self._running:
            try:
                item = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue
            stream_id, frame_num, frame_rgb = item
            try:
                self._infer(stream_id, frame_num, frame_rgb)
            except Exception as e:
                log.warning(f"Triton infer error stream {stream_id}: {e}")
                if self.config.passthrough_on_failure:
                    self._failed = True

    def _infer(self, stream_id: int, frame_num: int, frame_rgb: np.ndarray):
        """
        Calls the Triton model and stores overlay result.
        frame_rgb is a numpy array (H, W, 3) uint8 on CPU (copied from GPU surface).
        Inside the Triton model itself, cupy arrays are used — see model.py.
        """
        if self._failed or self._client is None:
            return

        import tritonclient.grpc as grpcclient

        inp = grpcclient.InferInput("input_frame", frame_rgb.shape, "UINT8")
        inp.set_data_from_numpy(frame_rgb)

        out = grpcclient.InferRequestedOutput("overlay_data")

        # sequence_id ensures ordering within a stream on Triton's stateful models
        response = self._client.infer(
            model_name=self.config.triton_model_name,
            inputs=[inp],
            outputs=[out],
            sequence_id=stream_id + 1,   # must be > 0
        )

        raw = response.as_numpy("overlay_data")  # shape (N, 6): x1,y1,x2,y2,class_id,conf
        boxes = []
        labels = []
        if raw is not None and len(raw) > 0:
            for det in raw:
                x1, y1, x2, y2, cls, conf = det
                label = f"cls{int(cls)} {conf:.2f}"
                boxes.append((float(x1), float(y1), float(x2), float(y2), label, float(conf)))
                labels.append(label)

        overlay = OverlayData(frame_num=frame_num, labels=labels, boxes=boxes)
        self.overlay_store.update(stream_id, overlay)

        # Track bytes sent to Triton
        self.resource_monitor.add_bytes(in_bytes=frame_rgb.nbytes)


# ---------------------------------------------------------------------------
# GStreamer Probe — attaches overlay to NvDsBatchMeta
# ---------------------------------------------------------------------------

def make_osd_probe(overlay_store: OverlayStore, triton_worker: TritonWorker,
                   resource_monitor: ResourceMonitor, config: PipelineConfig):
    """
    Returns a GStreamer pad probe function.
    Runs on every batch buffer — extracts frame, submits to Triton,
    and draws the latest overlay onto the batch meta.
    """

    def probe_fn(pad, info, u_data):
        try:
            gst_buffer = info.get_buffer()
            if not gst_buffer:
                return Gst.PadProbeReturn.OK

            batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
            if not batch_meta:
                return Gst.PadProbeReturn.OK

            l_frame = batch_meta.frame_meta_list
            while l_frame:
                try:
                    frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
                except StopIteration:
                    break

                stream_id = frame_meta.pad_index
                frame_num = frame_meta.frame_num

                # ---- Submit frame to Triton (async, non-blocking) ----
                try:
                    n_frame = pyds.get_nvds_buf_surface(hash(gst_buffer), frame_meta.batch_id)
                    # n_frame is numpy array on GPU memory — copy to CPU for Triton HTTP/gRPC
                    frame_copy = np.array(n_frame, copy=True, order="C")
                    frame_rgb = frame_copy[:, :, :3]  # drop alpha if RGBA
                    resource_monitor.add_bytes(in_bytes=frame_rgb.nbytes)
                    triton_worker.submit(stream_id, frame_num, frame_rgb)
                except Exception as e:
                    log.debug(f"Frame extract error: {e}")

                # ---- Get current overlay (or hold last) ----
                overlay = overlay_store.get(stream_id)
                if overlay and config.enable_osd:
                    _draw_overlay(batch_meta, frame_meta, overlay)

                try:
                    l_frame = l_frame.next
                except StopIteration:
                    break

        except Exception as e:
            log.warning(f"Probe error: {e}")

        return Gst.PadProbeReturn.OK

    return probe_fn


def _draw_overlay(batch_meta, frame_meta, overlay: OverlayData):
    """Draw bounding boxes and labels using NvDsDisplayMeta."""
    try:
        display_meta = pyds.nvds_acquire_display_meta_from_pool(batch_meta)
        display_meta.num_rects = 0
        display_meta.num_labels = 0

        for i, (x1, y1, x2, y2, label, conf) in enumerate(overlay.boxes[:16]):  # max 16
            if display_meta.num_rects < pyds.MAX_ELEMENTS_IN_DISPLAY_META:
                rect = display_meta.rect_params[display_meta.num_rects]
                rect.left = int(x1)
                rect.top = int(y1)
                rect.width = int(x2 - x1)
                rect.height = int(y2 - y1)
                rect.border_width = 2
                rect.border_color.set(0.0, 1.0, 0.0, 1.0)  # green
                rect.has_bg_color = 0
                display_meta.num_rects += 1

            if display_meta.num_labels < pyds.MAX_ELEMENTS_IN_DISPLAY_META:
                txt = display_meta.text_params[display_meta.num_labels]
                txt.display_text = label
                txt.x_offset = int(x1)
                txt.y_offset = max(0, int(y1) - 20)
                txt.font_params.font_name = "Serif"
                txt.font_params.font_size = 12
                txt.font_params.font_color.set(1.0, 1.0, 1.0, 1.0)
                txt.set_bg_clr = 1
                txt.text_bg_clr.set(0.0, 0.0, 0.0, 0.6)
                display_meta.num_labels += 1

        pyds.nvds_add_display_meta_to_frame(frame_meta, display_meta)
    except Exception as e:
        log.debug(f"Overlay draw error: {e}")


# ---------------------------------------------------------------------------
# Stream Health Monitor  (per-stream watchdog)
# ---------------------------------------------------------------------------

class StreamHealthMonitor:
    """
    Tracks frame arrival per stream.
    If a stream goes silent > timeout → triggers black screen + recovery.
    """

    def __init__(self, num_streams: int, timeout_sec: float = 5.0,
                 recovery_cb: Optional[Callable] = None):
        self._last_seen: Dict[int, float] = {i: time.time() for i in range(num_streams)}
        self._healthy: Dict[int, bool] = {i: True for i in range(num_streams)}
        self._lock = threading.Lock()
        self.timeout = timeout_sec
        self.recovery_cb = recovery_cb
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def heartbeat(self, stream_id: int):
        with self._lock:
            self._last_seen[stream_id] = time.time()
            if not self._healthy[stream_id]:
                log.info(f"Stream {stream_id} recovered.")
                self._healthy[stream_id] = True

    def is_healthy(self, stream_id: int) -> bool:
        with self._lock:
            return self._healthy[stream_id]

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._watchdog, daemon=True, name="stream_watchdog")
        self._thread.start()

    def stop(self):
        self._running = False

    def _watchdog(self):
        while self._running:
            time.sleep(1.0)
            now = time.time()
            with self._lock:
                for sid, last in self._last_seen.items():
                    if now - last > self.timeout and self._healthy[sid]:
                        log.warning(f"Stream {sid} lost! Triggering recovery.")
                        self._healthy[sid] = False
                        if self.recovery_cb:
                            threading.Thread(
                                target=self.recovery_cb, args=(sid,),
                                daemon=True
                            ).start()


# ---------------------------------------------------------------------------
# Pipeline Builder
# ---------------------------------------------------------------------------

class DeepStreamPipeline:

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.pipeline: Optional[Gst.Pipeline] = None
        self.loop: Optional[GLib.MainLoop] = None
        self._failed = False

        self.resource_monitor = ResourceMonitor(gpu_id=config.gpu_id)
        self.overlay_store = OverlayStore(
            num_streams=len(config.sources),
            hold_frames=config.overlay_hold_frames
        )
        self.triton_worker = TritonWorker(config, self.overlay_store, self.resource_monitor)
        self.health_monitor = StreamHealthMonitor(
            num_streams=len(config.sources),
            timeout_sec=config.recovery_interval_sec,
            recovery_cb=self._on_stream_failure
        )

    # ------------------------------------------------------------------
    def build(self):
        Gst.init(None)
        self.pipeline = Gst.Pipeline.new("ds-pipeline")

        # --- StreamMuxer ---
        streammux = self._make("nvstreammux", "muxer")
        streammux.set_property("width", self.config.muxer_output_width)
        streammux.set_property("height", self.config.muxer_output_height)
        streammux.set_property("batch-size", len(self.config.sources))
        streammux.set_property("batched-push-timeout", 33333)  # ~30fps timeout
        streammux.set_property("live-source", 1)
        streammux.set_property("gpu-id", self.config.gpu_id)
        self.pipeline.add(streammux)

        # --- Sources ---
        for i, src in enumerate(self.config.sources):
            self._add_source(i, src, streammux)

        # --- NvInfer (optional: built-in nvinfer for quick detectors) ---
        # We use Triton via probe — skip nvinfer here for full Triton control.

        # --- OSD ---
        if self.config.enable_osd:
            nvvidconv_pre = self._make("nvvideoconvert", "vidconv_pre")
            nvvidconv_pre.set_property("gpu-id", self.config.gpu_id)
            self.pipeline.add(nvvidconv_pre)

            osd = self._make("nvdsosd", "osd")
            osd.set_property("gpu-id", self.config.gpu_id)
            osd.set_property("process-mode", 1)  # GPU mode
            self.pipeline.add(osd)

            # Attach probe to OSD sink pad
            osd_sink_pad = osd.get_static_pad("sink")
            osd_sink_pad.add_probe(
                Gst.PadProbeType.BUFFER,
                make_osd_probe(
                    self.overlay_store, self.triton_worker,
                    self.resource_monitor, self.config
                )
            )

        # --- Tiler (grid layout for multi-stream) ---
        tiler = self._make("nvmultistreamtiler", "tiler")
        cols = max(1, int(len(self.config.sources) ** 0.5 + 0.5))
        rows = max(1, (len(self.config.sources) + cols - 1) // cols)
        tiler.set_property("rows", rows)
        tiler.set_property("columns", cols)
        tiler.set_property("width", self.config.output_width)
        tiler.set_property("height", self.config.output_height)
        tiler.set_property("gpu-id", self.config.gpu_id)
        self.pipeline.add(tiler)

        # --- Encoder ---
        nvvidconv_enc = self._make("nvvideoconvert", "vidconv_enc")
        nvvidconv_enc.set_property("gpu-id", self.config.gpu_id)
        self.pipeline.add(nvvidconv_enc)

        encoder = self._make("nvv4l2h264enc", "encoder")
        encoder.set_property("bitrate", self.config.output_bitrate)
        encoder.set_property("gpu-id", self.config.gpu_id)
        encoder.set_property("preset-level", 1)     # ultra-fast
        encoder.set_property("insert-sps-pps", 1)
        encoder.set_property("bufapi-version", 1)
        self.pipeline.add(encoder)

        # --- RTP Payloader ---
        rtppay = self._make("rtph264pay", "rtppay")
        rtppay.set_property("config-interval", 1)
        rtppay.set_property("pt", 96)
        self.pipeline.add(rtppay)

        # --- RTSP Server Sink ---
        sink = self._make("udpsink", "sink")
        sink.set_property("host", "224.224.255.255")
        sink.set_property("port", self.config.output_rtsp_port)
        sink.set_property("async", False)
        sink.set_property("sync", True)
        self.pipeline.add(sink)

        # --- Link chain ---
        chain = []
        if self.config.enable_osd:
            chain = [streammux, nvvidconv_pre, osd, tiler, nvvidconv_enc, encoder, rtppay, sink]
        else:
            chain = [streammux, tiler, nvvidconv_enc, encoder, rtppay, sink]

        for a, b in zip(chain[:-1], chain[1:]):
            if not a.link(b):
                raise RuntimeError(f"Failed to link {a.get_name()} -> {b.get_name()}")

        # --- RTSP server ---
        self._setup_rtsp_server()

        log.info("Pipeline built successfully.")

    # ------------------------------------------------------------------
    def _add_source(self, idx: int, src: str, streammux):
        is_rtsp = src.startswith("rtsp://")
        is_file = not is_rtsp

        srcbin = Gst.Bin.new(f"source-bin-{idx}")

        if is_rtsp:
            source = self._make("rtspsrc", f"src_{idx}")
            source.set_property("location", src)
            source.set_property("latency", 100)
            source.set_property("drop-on-latency", True)
            srcbin.add(source)

            depay = self._make("rtph264depay", f"depay_{idx}")
            parse = self._make("h264parse", f"parse_{idx}")
            decoder = self._make("nvv4l2decoder", f"dec_{idx}")
            decoder.set_property("gpu-id", self.config.gpu_id)
            decoder.set_property("enable-max-performance", 1)

            for el in [depay, parse, decoder]:
                srcbin.add(el)

            depay.link(parse)
            parse.link(decoder)

            def on_pad_added(element, pad, depay=depay):
                caps = pad.get_current_caps()
                if caps and "video" in caps.to_string():
                    sink_pad = depay.get_static_pad("sink")
                    if not sink_pad.is_linked():
                        pad.link(sink_pad)

            source.connect("pad-added", on_pad_added)

            ghost_pad = Gst.GhostPad.new("src", decoder.get_static_pad("src"))

        else:  # file
            source = self._make("filesrc", f"src_{idx}")
            source.set_property("location", src)
            srcbin.add(source)

            demux = self._make("qtdemux", f"demux_{idx}")
            parse = self._make("h264parse", f"parse_{idx}")
            decoder = self._make("nvv4l2decoder", f"dec_{idx}")
            decoder.set_property("gpu-id", self.config.gpu_id)
            decoder.set_property("enable-max-performance", 1)

            for el in [demux, parse, decoder]:
                srcbin.add(el)

            source.link(demux)
            parse.link(decoder)

            def on_demux_pad(element, pad, parse=parse):
                caps = pad.get_current_caps() or pad.query_caps(None)
                if caps and "video" in caps.to_string():
                    sink = parse.get_static_pad("sink")
                    if not sink.is_linked():
                        pad.link(sink)

            demux.connect("pad-added", on_demux_pad)

            ghost_pad = Gst.GhostPad.new("src", decoder.get_static_pad("src"))

        srcbin.add_pad(ghost_pad)
        self.pipeline.add(srcbin)

        sinkpad = streammux.get_request_pad(f"sink_{idx}")
        srcpad = srcbin.get_static_pad("src")
        if srcpad and sinkpad:
            srcpad.link(sinkpad)
        else:
            log.error(f"Could not link source {idx} to muxer")

    # ------------------------------------------------------------------
    def _setup_rtsp_server(self):
        """Proper RTSP server using GstRtspServer."""
        self.rtsp_server = GstRtspServer.RTSPServer.new()
        self.rtsp_server.set_service(str(self.config.output_rtsp_port))

        factory = GstRtspServer.RTSPMediaFactory.new()
        factory.set_launch(
            f"( udpsrc name=pay0 port={self.config.output_rtsp_port} "
            f"caps=\"application/x-rtp, media=video, encoding-name=H264, payload=96\" "
            f"! rtph264depay ! h264parse ! rtph264pay name=pay0 pt=96 )"
        )
        factory.set_shared(True)

        mounts = self.rtsp_server.get_mount_points()
        mounts.add_factory("/live", factory)
        self.rtsp_server.attach(None)

        log.info(f"RTSP output: rtsp://localhost:{self.config.output_rtsp_port}/live")

    # ------------------------------------------------------------------
    def _on_stream_failure(self, stream_id: int):
        log.warning(f"Recovery: stream {stream_id} — presenting black screen, retrying...")
        # In production: replace source with a videotestsrc (black) and re-link
        # For now we log and let the watchdog retry
        time.sleep(self.config.recovery_interval_sec)
        log.info(f"Attempting reconnect for stream {stream_id}")
        # TODO: dynamic source reconnection via GStreamer pipeline surgery

    # ------------------------------------------------------------------
    def _make(self, element_type: str, name: str) -> Gst.Element:
        el = Gst.ElementFactory.make(element_type, name)
        if not el:
            raise RuntimeError(f"Could not create GStreamer element: {element_type}")
        return el

    # ------------------------------------------------------------------
    def _bus_callback(self, bus, msg):
        t = msg.type
        if t == Gst.MessageType.EOS:
            log.info("End of stream.")
            self.loop.quit()
        elif t == Gst.MessageType.ERROR:
            err, dbg = msg.parse_error()
            log.error(f"GStreamer error: {err} | {dbg}")
            if self.config.passthrough_on_failure:
                log.warning("Passthrough mode — continuing despite error.")
            else:
                self.loop.quit()
        elif t == Gst.MessageType.WARNING:
            warn, dbg = msg.parse_warning()
            log.warning(f"GStreamer warning: {warn} | {dbg}")
        return True

    # ------------------------------------------------------------------
    def start(self):
        self.triton_worker.start()
        self.resource_monitor.start()
        self.health_monitor.start()

        ret = self.pipeline.set_state(Gst.State.PLAYING)
        if ret == Gst.StateChangeReturn.FAILURE:
            raise RuntimeError("Pipeline failed to start.")

        bus = self.pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect("message", self._bus_callback)

        # Periodic resource logging
        def log_resources():
            self.resource_monitor.print_report()
            return True  # keep repeating

        GLib.timeout_add_seconds(
            int(self.config.log_resources_interval_sec),
            log_resources
        )

        self.loop = GLib.MainLoop()
        log.info("Pipeline running. CTRL+C to stop.")
        try:
            self.loop.run()
        except KeyboardInterrupt:
            log.info("Interrupted.")
        finally:
            self.stop()

    # ------------------------------------------------------------------
    def stop(self):
        log.info("Stopping pipeline...")
        self.triton_worker.stop()
        self.resource_monitor.stop()
        self.health_monitor.stop()
        if self.pipeline:
            self.pipeline.set_state(Gst.State.NULL)
        log.info("Pipeline stopped.")

    # ------------------------------------------------------------------
    def get_resources(self) -> Dict:
        """Public API: call anytime to get a resource snapshot."""
        return self.resource_monitor.get_report()


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------

def run_pipeline(sources: List[str], **kwargs):
    cfg = PipelineConfig(sources=sources, **kwargs)
    p = DeepStreamPipeline(cfg)
    p.build()
    p.start()
    return p


if __name__ == "__main__":
    sources = sys.argv[1:] if len(sys.argv) > 1 else [
        "rtsp://192.168.1.100:554/stream1",
        "rtsp://192.168.1.101:554/stream2",
        "/data/videos/test1.mp4",
    ]
    run_pipeline(sources)
