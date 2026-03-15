"""
Entry point — parse args and launch pipeline.
"""

import argparse
import sys
import signal
from pipeline import run_pipeline, PipelineConfig, DeepStreamPipeline

def main():
    parser = argparse.ArgumentParser(description="DeepStream Multi-Stream Triton Pipeline")
    parser.add_argument(
        "--sources", nargs="+", required=True,
        help="List of RTSP URLs or local MP4 file paths"
    )
    parser.add_argument("--triton-url", default="localhost:8001", help="Triton gRPC endpoint")
    parser.add_argument("--triton-model", default="frame_analyzer")
    parser.add_argument("--output-port", type=int, default=8554)
    parser.add_argument("--output-width", type=int, default=1280)
    parser.add_argument("--output-height", type=int, default=720)
    parser.add_argument("--output-fps", type=int, default=25)
    parser.add_argument("--bitrate", type=int, default=4000000)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--overlay-buffer", type=int, default=8,
                        help="Max frames buffered waiting for Triton overlay")
    parser.add_argument("--overlay-hold", type=int, default=60,
                        help="Hold last overlay for this many frames if no new result")
    parser.add_argument("--passthrough", action="store_true", default=True,
                        help="Pass-through video if Triton fails")
    parser.add_argument("--log-resources", type=float, default=10.0,
                        help="Log resource usage every N seconds")
    args = parser.parse_args()

    cfg = PipelineConfig(
        sources=args.sources,
        triton_url=args.triton_url,
        triton_model_name=args.triton_model,
        output_rtsp_port=args.output_port,
        output_width=args.output_width,
        output_height=args.output_height,
        output_fps=args.output_fps,
        output_bitrate=args.bitrate,
        gpu_id=args.gpu_id,
        overlay_buffer_size=args.overlay_buffer,
        overlay_hold_frames=args.overlay_hold,
        passthrough_on_failure=args.passthrough,
        log_resources_interval_sec=args.log_resources,
    )

    pipeline = DeepStreamPipeline(cfg)
    pipeline.build()

    # Graceful shutdown on SIGINT/SIGTERM
    def _shutdown(sig, frame):
        print("\nShutting down...")
        pipeline.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    pipeline.start()


if __name__ == "__main__":
    main()
