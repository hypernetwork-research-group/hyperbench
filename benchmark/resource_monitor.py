import csv
import pynvml
import psutil
import threading
import time

from lightning.pytorch.callbacks import Callback
from pathlib import Path


class ResourceMonitor(Callback):
    """
    A callback to monitor resource usage during training.
    Usable only on NVIDIA GPUs. It records CPU, RAM, GPU usage, and GPU memory usage at a specified
        interval and saves the results to a CSV file at the end of training or upon an exception.
    """

    def __init__(
        self,
        csv_path: str | Path,
        interval: float = 1.0,
        gpu_id: int = 0,
    ):
        super().__init__()

        self.csv_path = Path(csv_path)
        self.interval = interval
        self.gpu_id = gpu_id

        self.running = False
        self.thread = None
        self.handle = None

        self.cpu_usage = []
        self.ram_usage = []
        self.gpu_usage = []
        self.gpu_memory = []

    def on_fit_start(self, trainer, pl_module):
        if not trainer.is_global_zero:
            return

        self.csv_path.parent.mkdir(parents=True, exist_ok=True)

        pynvml.nvmlInit()
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(self.gpu_id)

        self.running = True
        self.thread = threading.Thread(
            target=self._monitor,
            daemon=True,
        )
        self.thread.start()

    def on_fit_end(self, trainer, pl_module):
        if not trainer.is_global_zero:
            return

        self._stop_and_save()

    def on_exception(self, trainer, pl_module, exception):
        if trainer.is_global_zero:
            self._stop_and_save()

    def _stop_and_save(self):
        if not self.running:
            return

        self.running = False

        if self.thread is not None:
            self.thread.join()

        pynvml.nvmlShutdown()

        if not self.cpu_usage:
            return

        cpu_average = sum(self.cpu_usage) / len(self.cpu_usage)
        cpu_peak = max(self.cpu_usage)

        ram_average = sum(self.ram_usage) / len(self.ram_usage)
        ram_peak = max(self.ram_usage)

        gpu_average = sum(self.gpu_usage) / len(self.gpu_usage)
        gpu_peak = max(self.gpu_usage)

        gpu_memory_average_mb = sum(self.gpu_memory) / len(self.gpu_memory) / 1024**2
        gpu_memory_peak_mb = max(self.gpu_memory) / 1024**2

        with self.csv_path.open(
            mode="w",
            newline="",
            encoding="utf-8",
        ) as file:
            writer = csv.writer(file)

            writer.writerow(
                [
                    "metric",
                    "average",
                    "peak",
                    "unit",
                ]
            )

            writer.writerow(
                [
                    "cpu_usage",
                    cpu_average,
                    cpu_peak,
                    "%",
                ]
            )

            writer.writerow(
                [
                    "ram_usage",
                    ram_average,
                    ram_peak,
                    "%",
                ]
            )

            writer.writerow(
                [
                    "gpu_usage",
                    gpu_average,
                    gpu_peak,
                    "%",
                ]
            )

            writer.writerow(
                [
                    "gpu_memory",
                    gpu_memory_average_mb,
                    gpu_memory_peak_mb,
                    "MB",
                ]
            )

        print(f"Resource usage saved to {self.csv_path}")

    def _monitor(self):
        while self.running:
            self.cpu_usage.append(psutil.cpu_percent(interval=None))

            self.ram_usage.append(psutil.virtual_memory().percent)

            utilization = pynvml.nvmlDeviceGetUtilizationRates(self.handle)
            memory = pynvml.nvmlDeviceGetMemoryInfo(self.handle)

            self.gpu_usage.append(float(utilization.gpu))
            self.gpu_memory.append(float(memory.used))

            time.sleep(self.interval)
