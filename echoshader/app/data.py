"""Session-local, bounded acoustic reads shared by the app's rendering views."""

import hashlib
import logging
import threading
import time
from collections import OrderedDict
from pathlib import Path

logger = logging.getLogger(__name__)


class SourceChangedError(RuntimeError):
    """A mutable source was republished while an app session was using it."""


def source_revision(path):
    if path is None:
        return None
    path = Path(path)
    if not path.is_dir():
        stat = path.stat()
        return (stat.st_mtime_ns, stat.st_size)
    stat = path.stat()
    result = [("directory", stat.st_mtime_ns)]
    for name in (".zmetadata", ".zattrs", ".zgroup", "zarr.json", ".revision"):
        file = path / name
        if file.exists():
            result.append(
                (
                    name,
                    file.stat().st_mtime_ns,
                    hashlib.blake2b(file.read_bytes(), digest_size=16).digest(),
                )
            )
    return tuple(result)


class AcousticReadCache:
    """Cache concrete selected arrays by variable/channel/window, with a byte cap.

    This is not a snapshot or a replacement for atomic publication by the writer.
    A changed source is refused rather than mixing metadata from different versions.
    """

    def __init__(self, ds, path=None, max_bytes=256 * 1024**2):
        self.ds = ds
        self.path = path
        self.revision = source_revision(path)
        self.max_bytes = int(max_bytes)
        self._items = OrderedDict()
        self._bytes = 0
        self._lock = threading.RLock()

    def _check(self):
        try:
            revision = source_revision(self.path)
        except OSError as exc:
            self.clear()
            raise SourceChangedError(
                "The acoustic source is temporarily unavailable. Reload after its writer finishes."
            ) from exc
        if revision != self.revision:
            self.clear()
            raise SourceChangedError(
                "The acoustic source was updated. Waiting for the viewer to refresh."
            )

    def clear(self):
        with self._lock:
            self._items.clear()
            self._bytes = 0

    def read(self, variable, channel, start, stop, vertical, sample_step=1):
        key = (variable, str(channel), int(start), int(stop), vertical, int(sample_step))
        with self._lock:
            self._check()
            if key in self._items:
                self._items.move_to_end(key)
                return self._items[key]
            names = list(dict.fromkeys([variable, vertical]))
            data = self.ds[names].sel(channel=channel, drop=True).isel(ping_time=slice(start, stop))
            sample_dim = next(d for d in data[variable].dims if d != "ping_time")
            data = data.isel({sample_dim: slice(None, None, max(1, int(sample_step)))})
            started = time.perf_counter()
            try:
                data = data.compute()
                self._check()
            except SourceChangedError:
                raise
            except (ValueError, OSError) as exc:
                self._check()
                raise SourceChangedError(
                    "Could not read the acoustic Zarr consistently. It may be updating "
                    "or incomplete. Wait for its writer to finish, then reload. "
                    f"Underlying error: {exc}"
                ) from exc
            logger.info(
                "Acoustic read %s / %s: %.2fs, %.1f MiB",
                variable,
                channel,
                time.perf_counter() - started,
                data.nbytes / 1024**2,
            )
            size = data.nbytes
            if size <= self.max_bytes:
                while self._items and self._bytes + size > self.max_bytes:
                    _, old = self._items.popitem(last=False)
                    self._bytes -= old.nbytes
                self._items[key] = data
                self._bytes += size
            return data


def open_source(settings, path):
    """Open source metadata lazily; acoustic array reads remain windowed."""
    import xarray as xr

    kind = settings.get("type", "netcdf").lower()
    if kind == "zarr":
        return xr.open_zarr(path)
    if kind in ("netcdf", "nc"):
        return xr.open_dataset(path, chunks="auto")
    raise ValueError(f"Unsupported acoustic source: {kind!r}")


def bounded_preview(data, variable, max_pings, max_range_samples):
    """Select a bounded display sample lazily, preserving physical coordinates."""
    import math

    sample = next(dim for dim in data[variable].dims if dim not in ("channel", "ping_time"))
    ping_step = max(1, math.ceil(data.sizes["ping_time"] / max(1, int(max_pings))))
    sample_step = (
        max(1, math.ceil(data.sizes[sample] / max_range_samples)) if max_range_samples > 0 else 1
    )
    return data.isel(
        ping_time=slice(None, None, ping_step), **{sample: slice(None, None, sample_step)}
    )
