import sqlite3
import holoviews as hv
import numpy as np
import xarray as xr
import panel as pn


hv.extension("bokeh")


def test_previews_are_reused_and_refresh_reads_catalogue(tmp_path, monkeypatch):
    from echoshader.extensions import cps as adapter

    (tmp_path / "CPS_Sv").mkdir()
    t = np.datetime64("2024-01-01") + np.arange(6) * np.timedelta64(1, "s")
    ds = xr.Dataset(
        {
            "Sv": (("channel", "ping_time", "range_sample"), np.full((3, 6, 8), -60.0)),
            "depth": (
                ("ping_time", "range_sample"),
                np.tile(np.arange(8, dtype=float), (6, 1)),
            ),
            "longitude": ("ping_time", np.linspace(-124.5, -124.4, 6)),
            "latitude": ("ping_time", np.linspace(41.0, 41.01, 6)),
            "frequency_nominal": ("channel", [18000.0, 38000.0, 70000.0]),
        },
        coords={
            "channel": ["a", "b", "c"],
            "ping_time": t,
            "range_sample": np.arange(8),
        },
        attrs={"transect_number": "001"},
    )
    ds.to_zarr(tmp_path / "CPS_Sv" / "one.zarr", mode="w", zarr_format=2)
    con = sqlite3.connect(tmp_path / "processing.db")
    con.executescript("""CREATE TABLE transects (transect_part TEXT,transect_number TEXT,
      start_time TEXT,end_time TEXT,status TEXT,coverage_start TEXT,coverage_end TEXT);
      CREATE TABLE sv_cps (cps_filename TEXT,first_ping_time TEXT,last_ping_time TEXT,status TEXT);
      INSERT INTO transects VALUES ('001','001','2024-01-01T00:00:00','2024-01-01T00:00:05','completed','','');
      INSERT INTO sv_cps VALUES ('one.zarr','2024-01-01T00:00:00','2024-01-01T00:00:05','completed');""")
    con.commit()
    con.close()
    opens = []
    loads = []
    original_load = xr.Dataset.load

    def loading(data, *args, **kwargs):
        if "Sv" in data and data.Sv.ndim == 3:
            loads.append(data.Sv.shape)
        return original_load(data, *args, **kwargs)

    monkeypatch.setattr(xr.Dataset, "load", loading)
    original = xr.open_zarr

    def opening(*a, **kw):
        opens.append(a[0])
        return original(*a, **kw)

    monkeypatch.setattr(xr, "open_zarr", opening)
    _, tabs = adapter.build_tabs(
        ds=ds, root=tmp_path, config={}, max_pings=6, max_range_samples=8
    )
    assert not opens
    color = pn.widgets.RangeSlider(start=-120, end=-20, value=(-100, -30))
    tab = tabs[0][1](None, color.param.value)
    initial_opens = len(opens)
    assert initial_opens > 0  # Metadata discovery and preview are separate operations.
    assert len(loads) == 1
    color.value = (-90, -30)
    assert len(opens) == initial_opens
    assert len(loads) == 1
    refresh = tab.select(pn.widgets.Button)[0]
    refresh.clicks += 1
    assert len(opens) == initial_opens
    assert len(loads) == 1

    before = len(opens)
    catalogue = adapter.build_files(root=tmp_path)
    assert len(opens) == before  # catalogue is metadata only
    with catalogue["Latest completed transect"]() as latest:
        assert latest.sizes["ping_time"] == 6
        assert latest.attrs["transect_number"] == "001"

    from echoshader.extensions.transects import build_tabs

    _, generic = build_tabs(
        ds=ds,
        root=tmp_path / "CPS_Sv",
        config={"vert_dim": "depth", "channels": ["a", "b", "c"]},
        max_pings=6,
        max_range_samples=8,
    )
    content = generic[0][1]()
    assert isinstance(content, pn.Column)
    assert content.select(
        pn.pane.HoloViews
    )  # The generic wrapper still contains a rendered preview.
