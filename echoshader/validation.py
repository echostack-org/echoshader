"""Validate Echopype-style acoustic datasets for Echoshader.

Structural validation is global. Tricolor readiness is an optional capability
check, not a requirement for normal single-channel visualizations.
No data are edited, sorted, filtered, or resampled.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import xarray as xr


@dataclass(frozen=True)
class DatasetValidation:
    vertical_dim: str
    channels: tuple[str, ...]
    processing_function: str | None
    resampling_mode: str | None
    tricolor_ready: bool
    tricolor_reason: str


def resolve_rgb_channels(ds, frequencies=None, *, tolerance_hz=1000):
    """Resolve RGB frequencies in Hz; explicit requests never use substitutes.

    Without configuration, prefer 120/38/70 kHz then remaining distinct
    channels. Fewer than three channels cannot produce RGB.
    """
    names = tuple(map(str, ds.channel.values))
    nominal = ds.get("frequency_nominal")
    if nominal is None:
        if frequencies is not None:
            raise ValueError("RGB frequencies require frequency_nominal metadata.")
        if len(names) < 3:
            raise ValueError("RGB requires three distinct channels.")
        return names[:3]
    if nominal.dims != ("channel",):
        raise ValueError("frequency_nominal must have one value per channel.")
    values = np.asarray(nominal.values, dtype=float)
    if frequencies is None:
        selected = []
        for target in (120000, 38000, 70000):
            matches = np.flatnonzero(np.isfinite(values) & (abs(values - target) <= tolerance_hz))
            if len(matches) == 1 and names[matches[0]] not in selected:
                selected.append(names[matches[0]])
        selected.extend(name for name in names if name not in selected)
        if len(selected) < 3:
            raise ValueError("RGB requires three distinct channels.")
        return tuple(selected[:3])
    if not isinstance(frequencies, (list, tuple)) or len(frequencies) != 3:
        raise ValueError("rgb_frequencies must contain three frequencies in Hz (red, green, blue).")
    if any(isinstance(f, (bool, str)) for f in frequencies):
        raise ValueError("RGB frequencies must be positive finite numbers in Hz.")
    try:
        requested = np.asarray(frequencies, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("RGB frequencies must be positive finite numbers in Hz.") from exc
    if requested.shape != (3,) or not np.isfinite(requested).all() or (requested <= 0).any():
        raise ValueError("RGB frequencies must be positive finite numbers in Hz.")
    selected = []
    for target in requested:
        matches = np.flatnonzero(np.isfinite(values) & (abs(values - target) <= tolerance_hz))
        if len(matches) != 1:
            available = ", ".join(f"{f / 1000:g}" for f in values if np.isfinite(f))
            raise ValueError(
                f"RGB requires one channel near {target / 1000:g} kHz; found {len(matches)}. "
                f"Available frequencies (kHz): {available}."
            )
        selected.append(names[matches[0]])
    if len(set(selected)) != 3:
        raise ValueError("RGB frequencies must resolve to three distinct channels.")
    return tuple(selected)


def _equal_geometry(a: xr.DataArray, b: xr.DataArray, atol: float) -> bool:
    """Compare full physical geometry, including matching missing-value masks."""
    a, b = xr.align(a, b, join="exact")
    close = xr.apply_ufunc(
        np.isclose,
        a,
        b,
        kwargs={"rtol": 0.0, "atol": atol, "equal_nan": True},
        dask="allowed",
    )
    result = close.all()
    return bool(result.compute().item() if hasattr(result.data, "compute") else result.item())


def _tricolor_status(
    ds: xr.Dataset,
    channels: tuple[str, ...],
    vertical_dim: str,
    atol: float,
    vert_dim: str | None,
    allow_variable_geometry: bool = False,
) -> tuple[bool, str]:
    if len(channels) != 3 or len(set(channels)) != 3:
        return False, "Tricolor requires exactly three distinct channels."
    missing = [c for c in channels if c not in ds.channel.values]
    if missing:
        return False, f"Unknown tricolor channels: {missing}."

    # A one-dimensional physical depth/range dimension is a verified shared grid.
    if vertical_dim in ("depth", "echo_range"):
        axis = ds[vertical_dim]
        if axis.dims == (vertical_dim,):
            values = np.asarray(axis.values)
            if not np.isfinite(values).all():
                return False, f"Physical {vertical_dim} axis has invalid values."
            if not np.all(np.diff(values) > 0):
                return False, f"Physical {vertical_dim} axis must increase."
            return True, "Shared one-dimensional physical grid."

    # A range_sample axis alone has no physical meaning.
    if vert_dim is not None and vert_dim not in (vertical_dim, "depth", "echo_range"):
        return False, f"Unsupported tricolor vertical axis {vert_dim!r}."
    candidates = (vert_dim,) if vert_dim in ("depth", "echo_range") else ("depth", "echo_range")
    physical = next(
        (name for name in candidates if name in ds and vertical_dim in ds[name].dims), None
    )
    if physical is None:
        return (
            False,
            "No physical depth/echo_range geometry; range_sample indices are insufficient.",
        )

    geometry = ds[physical]
    if "channel" in geometry.dims:
        first = geometry.sel(channel=channels[0], drop=True)
        for ch in channels[1:]:
            try:
                matched = _equal_geometry(first, geometry.sel(channel=ch, drop=True), atol)
            except (ValueError, KeyError) as exc:
                return False, f"Could not compare physical geometry: {exc}"
            if not matched:
                return (
                    False,
                    f"Physical {physical} geometry differs between {channels[0]!r} and {ch!r}.",
                )
        geometry = first

    if allow_variable_geometry and "ping_time" in geometry.dims:
        values = np.asarray(geometry.transpose(vertical_dim, "ping_time").compute().values)
        if not np.isfinite(values).any():
            return False, "No finite physical geometry."
        for column in values.T:
            finite = column[np.isfinite(column)]
            if finite.size and (finite.size < 2 or np.any(np.diff(finite) <= 0)):
                return False, "Physical geometry must increase within each ping."
        return True, f"Channels share physical {physical} cell geometry."

    if "ping_time" in geometry.dims:
        ref = geometry.isel(ping_time=0, drop=True)
        if not _equal_geometry(geometry, ref, atol):
            return (
                False,
                "Channels align, but physical geometry changes between pings; current hv.RGB needs a constant vertical axis.",
            )
        geometry = ref

    if geometry.ndim != 1 or geometry.dims != (vertical_dim,):
        return False, f"Cannot render {physical} geometry with the current 1D-axis hv.RGB renderer."

    values = np.asarray(geometry.compute().values)
    finite = values[np.isfinite(values)]
    if finite.size < 2 or not np.all(np.diff(finite) > 0):
        return False, f"Physical {physical} axis has too few valid or increasing coordinates."
    return True, f"Channels share constant physical {physical} geometry."


def validate_dataset(
    ds: xr.Dataset,
    *,
    tricolor_channels: Sequence[str] | None = None,
    vert_dim: str | None = None,
    var_name: str = "Sv",
    atol: float = 1e-6,
    allow_variable_geometry: bool = False,
) -> DatasetValidation:
    """Validate dataset schema and report readiness for tricolor rendering.

    Raises ValueError for invalid acoustic input. An incompatible tricolor
    configuration instead returns tricolor_ready=False so normal echograms work.
    Full physical-coordinate checks may trigger Dask computation when tricolor
    channels are supplied.
    """
    if not isinstance(ds, xr.Dataset):
        raise TypeError("Echoshader expects an xarray.Dataset.")
    if var_name not in ds:
        raise ValueError(f"Dataset must contain a variable named {var_name!r}.")

    sv = ds[var_name]
    required = {"channel", "ping_time"}
    if sv.ndim != 3 or sv.dims[:2] != ("channel", "ping_time"):
        raise ValueError(
            f"{var_name!r} requires exactly (channel, ping_time, vertical) dimensions; got {sv.dims}."
        )
    vertical_dim = next(d for d in sv.dims if d not in required)
    if vertical_dim not in {"range_sample", "depth", "echo_range"}:
        raise ValueError(f"Unsupported vertical dimension: {vertical_dim!r}.")
    for name in ("channel", "ping_time", vertical_dim):
        if name not in ds.coords or ds[name].dims != (name,):
            raise ValueError(f"Missing one-dimensional coordinate {name!r}.")
        if ds.sizes[name] == 0:
            raise ValueError(f"Empty coordinate {name!r}.")

    channels = tuple(map(str, ds.channel.values.tolist()))
    if len(set(channels)) != len(channels):
        raise ValueError("Duplicate channel identifiers.")
    times = np.asarray(ds.ping_time.values)
    if not np.issubdtype(times.dtype, np.datetime64):
        raise ValueError("ping_time must have datetime64 dtype.")
    times = times.astype("datetime64[ns]")
    if np.isnat(times).any() or np.any(np.diff(times.astype("int64")) <= 0):
        raise ValueError("ping_time must be valid, unique, and strictly increasing.")

    if tricolor_channels is None:
        ready, reason = False, "Not checked (pass tricolor_channels to validate physical geometry)."
    else:
        ready, reason = _tricolor_status(
            ds,
            tuple(map(str, tricolor_channels)),
            vertical_dim,
            atol,
            vert_dim,
            allow_variable_geometry,
        )
    return DatasetValidation(
        vertical_dim=vertical_dim,
        channels=channels,
        processing_function=ds.attrs.get("processing_function"),
        resampling_mode=sv.attrs.get("resampling_mode"),
        tricolor_ready=ready,
        tricolor_reason=reason,
    )
