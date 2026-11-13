import asyncio

import dask.array as da
import numpy as np
import panel as pn
import pytest
import xarray as xr
from dask import delayed

from echoshader.app.data import AcousticReadCache, SourceChangedError
from echoshader.app.single_viewer import AcousticSource, build_acoustic_app


def dataset(calls):
    @delayed
    def read():
        calls.append("acoustic read")
        return np.full((3, 5, 8), -60.0)

    return xr.Dataset(
        {
            "Sv": (
                ("channel", "ping_time", "echo_range"),
                da.from_delayed(read(), shape=(3, 5, 8), dtype=float),
            )
        },
        coords={
            "channel": ["a", "b", "c"],
            "echo_range": np.arange(8, dtype=float),
            "ping_time": np.datetime64("2024-01-01") + np.arange(5) * np.timedelta64(1, "s"),
        },
    )


def test_shared_cache_reuses_materialized_selection():
    calls = []
    ds = dataset(calls)
    cache = AcousticReadCache(ds, max_bytes=100000)
    first = cache.read("Sv", "a", 0, 4, "echo_range")
    assert len(calls) == 1
    assert cache.read("Sv", "a", 0, 4, "echo_range") is first
    assert len(calls) == 1
    cache.read("Sv", "a", 1, 5, "echo_range")
    assert len(calls) == 2


def test_bounded_cache_and_changed_source(tmp_path):
    calls = []
    ds = dataset(calls)
    metadata = tmp_path / ".zattrs"
    metadata.write_text("{}")
    cache = AcousticReadCache(ds, tmp_path, max_bytes=1)
    first = cache.read("Sv", "a", 0, 4, "echo_range")
    second = cache.read("Sv", "a", 0, 4, "echo_range")
    assert len(calls) == 2  # above the cache budget: each read computes independently
    assert first.Sv.chunks is None
    assert second.Sv.chunks is None
    assert first.sizes == {"ping_time": 4, "echo_range": 8}
    xr.testing.assert_equal(first.compute(), second.compute())
    assert len(calls) == 2
    metadata.write_text('{"transect_number":"010"}')
    with pytest.raises(SourceChangedError, match="updated"):
        cache.read("Sv", "a", 0, 4, "echo_range")


def test_build_does_not_evaluate_hidden_views():
    calls = []
    extension_calls = []

    def extra(*_):
        extension_calls.append(1)
        return pn.pane.Markdown("Transects")

    app = build_acoustic_app(
        AcousticSource(dataset(calls)), extra_tabs=[("Recent transects", extra)]
    )
    assert not calls
    assert not extension_calls
    tabs = app.main[0]
    assert tabs.dynamic
    assert len(tabs) == 3
    extra_pane = tabs[2]
    extra_pane.eval(extra_pane.object)
    assert extension_calls == [1]
    assert not calls


def test_main_and_rgb_reuse_reads_and_leave_extension_unopened():
    calls = []
    extension_calls = []

    def extra(*_):
        extension_calls.append(1)
        return pn.pane.Markdown("Transects")

    app = build_acoustic_app(
        AcousticSource(dataset(calls), rasterized=False), extra_tabs=[("Recent transects", extra)]
    )
    tabs = app.main[0]
    multi = tabs[0][1]
    asyncio.run(multi.eval(multi.object))
    assert len(calls) == 3
    asyncio.run(multi.eval(multi.object))
    assert len(calls) == 3
    rgb = tabs[1][1]
    asyncio.run(rgb.eval(rgb.object))
    assert len(calls) == 3
    assert not extension_calls
