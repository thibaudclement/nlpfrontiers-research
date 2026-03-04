from __future__ import annotations
import subprocess
import threading
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional
from src.utils.io import write_json_file

@dataclass
class PowerSample:
    timestamp_seconds_since_start: float
    power_watts: float

# Read instantaneous GPU power draw (watts) using nvidia-smi
def read_gpu_power_watts_from_nvidia_smi() -> Optional[float]:
    try:
        # Query power draw without units (in multi-GPU setups, take GPU 0)
        command = [
            "nvidia-smi",
            "--query-gpu=power.draw",
            "--format=csv,noheader,nounits",
        ]
        output = subprocess.check_output(command, stderr=subprocess.STDOUT, text=True).strip()
        first_line = output.splitlines()[0].strip()
        return float(first_line)
    except Exception:
        return None

class EnergyMeter:
    # Initialize power sampler that approximates energy via trapezoidal integration
    def __init__(self, sampling_interval_seconds: float = 0.5) -> None:
        self.sampling_interval_seconds = float(sampling_interval_seconds)
        self.samples: List[PowerSample] = []
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._start_wall_time_seconds: Optional[float] = None
        self._stop_wall_time_seconds: Optional[float] = None

    # Start sampling GPU power
    def start(self) -> None:
        self._stop_event.clear()
        self.samples = []
        self._start_wall_time_seconds = time.time()
        self._stop_wall_time_seconds = None

        # Run the sampling loop in a daemon thread
        def sampling_loop() -> None:
            assert self._start_wall_time_seconds is not None
            while not self._stop_event.is_set():
                power_watts = read_gpu_power_watts_from_nvidia_smi()
                if power_watts is not None:
                    elapsed = time.time() - self._start_wall_time_seconds
                    self.samples.append(
                        PowerSample(timestamp_seconds_since_start=float(elapsed), power_watts=float(power_watts))
                    )
                time.sleep(self.sampling_interval_seconds)

        self._thread = threading.Thread(target=sampling_loop, daemon=True)
        self._thread.start()

    # Stop sampling GPU power
    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        self._thread = None
        self._stop_wall_time_seconds = time.time()

    # Compute measured wall-clock duration in seconds
    def get_duration_seconds(self) -> float:
        if self._start_wall_time_seconds is None:
            return 0.0
        end_time = self._stop_wall_time_seconds if self._stop_wall_time_seconds is not None else time.time()
        return float(end_time - self._start_wall_time_seconds)

    # Compute energy (joules) using trapezoidal integration over sampled power
    def get_energy_joules(self) -> float:
        if len(self.samples) < 2:
            return 0.0

        energy_joules = 0.0
        for previous_sample, current_sample in zip(self.samples[:-1], self.samples[1:]):
            delta_time = current_sample.timestamp_seconds_since_start - previous_sample.timestamp_seconds_since_start
            average_power = 0.5 * (previous_sample.power_watts + current_sample.power_watts)
            energy_joules += average_power * delta_time

        return float(energy_joules)

    # Produce JSON-serializable report for metering interval
    def build_report(self, additional_fields: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        report: Dict[str, Any] = {
            "sampling_interval_seconds": self.sampling_interval_seconds,
            "duration_seconds": self.get_duration_seconds(),
            "number_of_samples": len(self.samples),
            "energy_joules": self.get_energy_joules(),
            "samples": [asdict(sample) for sample in self.samples],
        }
        if additional_fields:
            report.update(additional_fields)
        return report

    # Save metering report as JSON
    def save_report(self, path: str | Path, additional_fields: Optional[Dict[str, Any]] = None) -> None:
        report = self.build_report(additional_fields=additional_fields)
        write_json_file(report, path)