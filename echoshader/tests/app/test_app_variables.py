"""Generic acoustic-variable selection, independent of workflow names."""

import numpy as np
import panel as pn
import pytest
import xarray as xr

from echoshader.app.builder import build_app
from echoshader.app.holoviz import AcousticSource, build_acoustic_app


@pytest.fixture
def dataset():
    shape = (3, 5, 10)
    ds = xr.Dataset(
        {
            "Sv": (("channel", "ping_time", "echo_range"), np.full(shape, -60.0)),
            "frequency_nominal": ("channel", [18000.0, 38000.0, 120000.0]),
        },
        coords={
            "channel": ["18", "38", "120"],
            "ping_time": np.datetime64("2024-08-05T13:20:00", "ns")
            + np.arange(5) * np.timedelta64(1, "s"),
            "echo_range": np.arange(10, dtype=float),
        },
    )
    ds["processed_backscatter"] = ds.Sv - 5
    return ds


def test_default_source_keeps_sv():
    assert AcousticSource(data=xr.Dataset()).variables == ("Sv",)


def test_app_exposes_configured_variable_choices(dataset):
    app = build_acoustic_app(
        AcousticSource(dataset, variables=("Sv", "processed_backscatter"))
    )
    selectors = {
        widget.label: widget
        for content in app.main
        for widget in content.select(pn.widgets.Select)
    }
    assert selectors["Acoustic variable"].options == ["Sv", "processed_backscatter"]
    assert selectors["RGB variable"].options == ["Sv", "processed_backscatter"]
    # Changing each selector must rebuild its bound view without an exception.
    selectors["Acoustic variable"].value = "processed_backscatter"
    selectors["RGB variable"].value = "processed_backscatter"


def test_yaml_builder_passes_variables(dataset, monkeypatch):
    import echoshader.app.builder as builder

    captured = {}

    def capture(source, **kwargs):
        captured["source"] = source
        return source

    monkeypatch.setattr(builder, "build_acoustic_app", capture)
    builder._build_file_app(
        {"display": {"variables": ["Sv", "processed_backscatter"]}},
        datasets={"survey": dataset},
    )
    assert captured["source"].variables == ("Sv", "processed_backscatter")


@pytest.mark.parametrize("variables", [[], "Sv", [None]])
def test_builder_rejects_invalid_variable_configuration(dataset, variables):
    with pytest.raises(ValueError, match="display.variables"):
        build_app({"display": {"variables": variables}}, datasets={"survey": dataset})


def test_app_rejects_missing_variable(dataset):
    with pytest.raises(ValueError, match="missing_backscatter"):
        build_acoustic_app(AcousticSource(dataset, variables=("missing_backscatter",)))


def test_app_rejects_duplicate_variable(dataset):
    with pytest.raises(ValueError, match="unique"):
        build_acoustic_app(AcousticSource(dataset, variables=("Sv", "Sv")))
