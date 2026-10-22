"""Background sampler for CPU, memory and disk, written alongside the slide stats.

This exists to answer one question the 48 h corpus run could not: about a quarter
of slides embedded at under 800 tiles/s while the rest ran at the GPU's ceiling,
and nothing about those slides -- codec, width, tile count, extract_px, segment
cache fit -- predicts which. Replaying the reader on them single-process gives
the same decode cost as the fast ones, so the loss is not in the slide.

The discriminator is `cpu_busy` against `cpu_ours`. Both are cores, not percent:

  * `cpu_busy` comes from /proc/stat, which is NOT namespaced, so inside the
    container it still reports the whole node.
  * `cpu_ours` sums utime+stime over the visible /proc/<pid>, and the container's
    PID namespace makes exactly our own process tree visible.

If a slow slide coincides with `cpu_busy` near `nproc` while `cpu_ours` stays
low, another tenant is eating the node -- slurm books GPUs here but does not
isolate CPUs. If `cpu_busy` tracks `cpu_ours` and both are low, the readers are
blocked on something of our own making.
"""

import os
import re
import threading
import time
from pathlib import Path

from ._logging import StatsLog

CLK_TCK = os.sysconf("SC_CLK_TCK")
PID_DIR = re.compile(r"^\d+$")
# partitions and loopbacks double-count; the corpus lives on an ext4 volume
# striped over the nvme0n1..nvme5n1 set
DISK = re.compile(r"^(nvme\d+n\d+|sd[a-z]+|md\d+)$")
SECTOR_BYTES = 512


def _cpu_total() -> tuple[float, float]:
    """(busy, total) jiffies for the whole node, from the un-namespaced /proc/stat."""
    with open("/proc/stat") as fh:
        fields = [float(v) for v in fh.readline().split()[1:]]
    total = sum(fields)
    idle = fields[3] + (fields[4] if len(fields) > 4 else 0.0)  # idle + iowait
    return total - idle, total


def _cpu_ours() -> float:
    """utime+stime jiffies summed over every process in our PID namespace."""
    total = 0.0
    for entry in os.listdir("/proc"):
        if not PID_DIR.match(entry):
            continue
        try:
            with open(f"/proc/{entry}/stat") as fh:
                data = fh.read()
            # comm can contain spaces and parentheses, so index past the last ')'
            fields = data[data.rindex(")") + 2 :].split()
            total += float(fields[11]) + float(fields[12])  # utime, stime
        except (OSError, ValueError, IndexError):
            continue  # process exited mid-scan; readers churn constantly
    return total


def _disk_read_bytes() -> float:
    total = 0.0
    try:
        with open("/proc/diskstats") as fh:
            for line in fh:
                parts = line.split()
                if len(parts) > 5 and DISK.match(parts[2]):
                    total += float(parts[5]) * SECTOR_BYTES  # sectors read
    except OSError:
        pass
    return total


def _ours_footprint() -> dict[str, float]:
    """Resident memory, process count, fds and shm segments for our own tree.

    The corpus run started at ~1095 tiles/s -- the same rate a fresh 40-slide run
    gets -- and decayed ~2.2x over its first four hours with slide properties
    held constant, so whatever causes it accumulates with elapsed run time. These
    are the process-internal candidates: RSS growth or fragmentation across the
    64 persistent readers, leaked descriptors, and the per-batch shared-memory
    segments torch allocates (128 in flight at ~77 MB each) failing to be reaped.

    `rss_sum_gb` double-counts pages shared between forked workers -- Virchow's
    weights and the coords buffer among them -- so read it as a trend, not a
    total. `rss_max_gb` is one process and is exact.
    """
    rss_pages = 0
    rss_max = 0
    procs = 0
    fds = 0
    page = os.sysconf("SC_PAGE_SIZE")
    for entry in os.listdir("/proc"):
        if not PID_DIR.match(entry):
            continue
        try:
            with open(f"/proc/{entry}/statm") as fh:
                resident = int(fh.read().split()[1])
            rss_pages += resident
            rss_max = max(rss_max, resident)
            procs += 1
            fds += len(os.listdir(f"/proc/{entry}/fd"))
        except (OSError, ValueError, IndexError):
            continue
    out = {
        "rss_sum_gb": round(rss_pages * page / 1e9, 2),
        "rss_max_gb": round(rss_max * page / 1e9, 2),
        "procs": procs,
        "fds": fds,
    }
    try:
        out["shm_files"] = len(os.listdir("/dev/shm"))
    except OSError:
        pass
    return out


def _meminfo() -> dict[str, float]:
    out = {}
    try:
        with open("/proc/meminfo") as fh:
            for line in fh:
                key, _, rest = line.partition(":")
                if key in ("MemAvailable", "Cached", "Shmem"):
                    out[key.lower()] = float(rest.split()[0]) * 1024
    except OSError:
        pass
    return out


class SysMonitor:
    """Samples into a `StatsLog` as `kind="sys"` records until stopped."""

    def __init__(self, *, stats: StatsLog, interval: float = 5.0):
        self._stats = stats
        self._interval = interval
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.nproc = os.cpu_count() or 0

    def start(self):
        self._thread = threading.Thread(
            target=self._run, name="sysmon", daemon=True  # never hold up a shutdown
        )
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2 * self._interval)

    def _run(self):
        busy0, total0 = _cpu_total()
        ours0 = _cpu_ours()
        read0 = _disk_read_bytes()
        t0 = time.monotonic()

        while not self._stop.wait(self._interval):
            try:
                busy1, total1 = _cpu_total()
                ours1 = _cpu_ours()
                read1 = _disk_read_bytes()
                t1 = time.monotonic()

                span, jiffies = t1 - t0, total1 - total0
                if span <= 0 or jiffies <= 0:
                    continue
                self._stats.write(
                    kind="sys",
                    t=time.time(),
                    span_s=round(span, 2),
                    nproc=self.nproc,
                    # cores, so they compare directly against nproc
                    cpu_busy=round((busy1 - busy0) / jiffies * self.nproc, 2),
                    cpu_ours=round((ours1 - ours0) / CLK_TCK / span, 2),
                    disk_read_mbs=round((read1 - read0) / span / 1e6, 1),
                    load1=round(os.getloadavg()[0], 2),
                    **_ours_footprint(),
                    **{k: round(v / 1e9, 2) for k, v in _meminfo().items()},
                )
                busy0, total0, ours0, read0, t0 = busy1, total1, ours1, read1, t1
            except Exception:
                # a sampler must never take the run down with it
                continue

    def __enter__(self) -> "SysMonitor":
        self.start()
        return self

    def __exit__(self, *exc):
        self.stop()
