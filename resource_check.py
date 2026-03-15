"""
Standalone resource checker — can be called from anywhere.

Usage:
    python3 resource_check.py                  # one-shot report
    python3 resource_check.py --watch 2        # refresh every 2 seconds
    python3 resource_check.py --json           # output JSON for programmatic use

Or import and use in your code:
    from resource_check import get_resources
    report = get_resources(gpu_id=0)
    print(report["gpu"]["mem_used_mb"])
"""

import argparse
import json
import time
from typing import Dict


def get_resources(gpu_id: int = 0) -> Dict:
    """
    Returns a reliable resource snapshot dict:
    {
        "gpu": {
            "id": 0,
            "mem_used_mb": 3200.5,
            "mem_total_mb": 24000.0,
            "mem_pct": 13.3,
            "gpu_util_pct": 45,
            "mem_util_pct": 12,
            "encoder_util_pct": 60,
            "decoder_util_pct": 70,
            "temperature_c": 72,
            "power_w": 180,
            "power_limit_w": 250,
        },
        "ram": {
            "used_mb": 8000.0,
            "total_mb": 32000.0,
            "pct": 25.0
        },
        "cpu": {
            "pct": 35.0,
            "cores": 16
        },
        "processes": [...]   # top GPU processes
    }
    """
    report = {}

    # --- GPU via pynvml ---
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)

        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        enc = pynvml.nvmlDeviceGetEncoderUtilization(handle)
        dec = pynvml.nvmlDeviceGetDecoderUtilization(handle)

        try:
            temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
        except Exception:
            temp = -1

        try:
            power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
            power_limit = pynvml.nvmlDeviceGetEnforcedPowerLimit(handle) / 1000.0
        except Exception:
            power = power_limit = -1

        # GPU processes
        procs = []
        try:
            for p in pynvml.nvmlDeviceGetComputeRunningProcesses(handle):
                procs.append({
                    "pid": p.pid,
                    "mem_mb": round(p.usedGpuMemory / 1e6, 1) if p.usedGpuMemory else 0
                })
        except Exception:
            pass

        report["gpu"] = {
            "id": gpu_id,
            "name": pynvml.nvmlDeviceGetName(handle).decode() if isinstance(
                pynvml.nvmlDeviceGetName(handle), bytes
            ) else pynvml.nvmlDeviceGetName(handle),
            "mem_used_mb": round(mem.used / 1e6, 1),
            "mem_free_mb": round(mem.free / 1e6, 1),
            "mem_total_mb": round(mem.total / 1e6, 1),
            "mem_pct": round(100 * mem.used / mem.total, 1),
            "gpu_util_pct": util.gpu,
            "mem_util_pct": util.memory,
            "encoder_util_pct": enc[0],
            "decoder_util_pct": dec[0],
            "temperature_c": temp,
            "power_w": round(power, 1),
            "power_limit_w": round(power_limit, 1),
            "processes": procs,
        }
        pynvml.nvmlShutdown()
    except Exception as e:
        report["gpu"] = {"error": str(e)}

    # --- System RAM ---
    try:
        import psutil
        vm = psutil.virtual_memory()
        report["ram"] = {
            "used_mb": round(vm.used / 1e6, 1),
            "available_mb": round(vm.available / 1e6, 1),
            "total_mb": round(vm.total / 1e6, 1),
            "pct": vm.percent,
        }
        report["cpu"] = {
            "pct": psutil.cpu_percent(interval=0.1),
            "cores": psutil.cpu_count(logical=True),
        }
    except Exception as e:
        report["ram"] = {"error": str(e)}

    return report


def print_report(report: Dict):
    gpu = report.get("gpu", {})
    ram = report.get("ram", {})
    cpu = report.get("cpu", {})

    sep = "=" * 52
    print(f"\n{sep}")
    print("  RESOURCE REPORT")
    print(sep)

    if "error" not in gpu:
        print(f"  GPU [{gpu.get('id')}] {gpu.get('name','')}")
        print(f"    Memory : {gpu.get('mem_used_mb')} / {gpu.get('mem_total_mb')} MB  ({gpu.get('mem_pct')}%)")
        print(f"    Compute: {gpu.get('gpu_util_pct')}%  |  Mem BW: {gpu.get('mem_util_pct')}%")
        print(f"    Encoder: {gpu.get('encoder_util_pct')}%  |  Decoder: {gpu.get('decoder_util_pct')}%")
        print(f"    Temp   : {gpu.get('temperature_c')}°C  |  Power: {gpu.get('power_w')}W / {gpu.get('power_limit_w')}W")
        procs = gpu.get("processes", [])
        if procs:
            print(f"    GPU Procs: " + ", ".join(f"pid={p['pid']} {p['mem_mb']}MB" for p in procs[:5]))
    else:
        print(f"  GPU error: {gpu.get('error')}")

    print(f"  RAM  : {ram.get('used_mb')} / {ram.get('total_mb')} MB  ({ram.get('pct')}%)")
    print(f"  CPU  : {cpu.get('pct')}%  ({cpu.get('cores')} cores)")
    print(sep + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--watch", type=float, default=0, help="Refresh interval in seconds (0=once)")
    parser.add_argument("--json", action="store_true", help="Output JSON")
    args = parser.parse_args()

    while True:
        r = get_resources(gpu_id=args.gpu_id)
        if args.json:
            print(json.dumps(r, indent=2))
        else:
            print_report(r)

        if args.watch <= 0:
            break
        time.sleep(args.watch)
