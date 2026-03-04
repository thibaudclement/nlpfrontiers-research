import csv
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional
import numpy as np
from pynvml import (
    nvmlInit,
    nvmlShutdown,
    nvmlDeviceGetHandleByIndex,
    nvmlDeviceGetPowerUsage,
)

# Store raw power samples for integration and debugging
@dataclass
class PowerSample:
    timestamp_s: float
    power_w: int

# Sample GPU power in background threat with NVML
class GPUPowerSampler:
    def __init__(self, gpu_index: int, sample_interval_s: float):
        self.gpu_index = gpu_index
        self.sample_interval_s = sample_interval_s
        self._samples: List[PowerSample] = []
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    # Start sampling power at fixed intervals
    def start(self) -> None:
        nvmlInit()
        self._stop_event.clear()
        self._samples.clear()
        self._thread = threading.Thread(target = self._run, daemon = True)
        self._thread.start()

    # Stop sampling power and wait for thread to finish
    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join()
        nvmlShutdown()

    # Return list of raw power samples
    def get_samples(self) -> List[PowerSample]:
        return list(self._samples)
    
    # Collect samples
    def _run(self) -> None:
        handle = nvmlDeviceGetHandleByIndex(self.gpu_index)
        start_time = time.perf_counter()
        while not self._stop_event.is_set():
            elapsed_time_s = time.perf_counter() - start_time
            # Important: NVML report power in milliwatts, which requires conversion to watts
            power_mw = nvmlDeviceGetPowerUsage(handle)
            power_w = power_mw / 1000.0
            self._samples.append(PowerSample(timestamp_s = elapsed_time_s, power_w = power_w))
            time.sleep(self.sample_interval_s)
                                 
# Compute energy in joules by integrating power samples over time
def integrate_energy(samples: List[PowerSample]) -> float:
    if len(samples) < 2:
        return float("nan")
    times = np.array([sample.timestamp_s for sample in samples], dtype = np.float64)
    powers = np.array([sample.power_w for sample in samples], dtype = np.float64)
    return float(np.trapz(y = powers, x = times))

# Save raw power trace to CSV
def save_power_trace_csv(samples: List[PowerSample], output_path: Path) -> None:
    with open(output_path, "w", newline = "", encoding = "utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp_s", "power_w"])
        for sample in samples:
            writer.writerow([f"{sample.timestamp_s:.6f}", f"{sample.power_w}"])