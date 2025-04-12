from pathlib import Path

import panel
import pytest
import xarray as xr

import echoshader

DATA_DIR = Path("./echoshader/test_data/concatenated_MVBS.nc")


@pytest.fixture
def get_data():
    # When the bug is fixed, set this to True
    echoshader.utils.gram_opts["Image"]["invert_yaxis"] = False

    # Load sample data for testing
    MVBS_ds = xr.open_mfdataset(
        paths=DATA_DIR,
        data_vars="minimal",
        coords="minimal",
        combine="by_coords",
    )

    return MVBS_ds


def test_echogram(get_data):
    # Load sample data for testing
    MVBS_ds = get_data

    # Create the echogram panel
    echogram_panel = panel.Row(
        panel.Column(
            MVBS_ds.eshader.Sv_range_slider,
            MVBS_ds.eshader.colormap,
        ),
        MVBS_ds.eshader.echogram(),
    )

    # Check if the panel is created without raising an exception
    assert isinstance(echogram_panel, panel.Row)


def test_tricolor_echogram(get_data):
    # Load sample data for testing
    MVBS_ds = get_data

    # Create the tricolor echogram panel
    tricolor_echogram_panel = panel.Row(
        panel.Column(MVBS_ds.eshader.Sv_range_slider),
        MVBS_ds.eshader.echogram(
            channel=[
                "GPT 120 kHz 00907205a6d0 4-1 ES120-7C",
                "GPT  38 kHz 009072058146 2-1 ES38B",
                "GPT  18 kHz 009072058c8d 1-1 ES18-11",
            ],
            rgb_composite=True,
        ),
    )

    # Check if the panel is created without raising an exception
    assert isinstance(tricolor_echogram_panel, panel.Row)


def test_track(get_data):
    # Load sample data for testing
    MVBS_ds = get_data

    track_panel = panel.Row(
        panel.Column(MVBS_ds.eshader.tile_select),
        MVBS_ds.eshader.track(),
    )

    # Check if the panel is created without raising an exception
    assert isinstance(track_panel, panel.Row)


def test_track_echogram_integration(get_data):
    # Load sample data for testing
    MVBS_ds = get_data

    # Create the panels
    MVBS_ds.eshader.control_mode_select.value = False

    track_panel = panel.Row(
        panel.Column(MVBS_ds.eshader.tile_select),
        MVBS_ds.eshader.track(),
    )

    echogram_panel = panel.Row(
        panel.Column(MVBS_ds.eshader.Sv_range_slider, MVBS_ds.eshader.colormap),
        MVBS_ds.eshader.echogram(),
    )

    integration_panel = panel.Column(
        track_panel,
        echogram_panel,
    )

    # Check if the panel is created without raising an exception
    assert isinstance(integration_panel, panel.Column)

    MVBS_ds.eshader.control_mode_select.value = True

    assert isinstance(integration_panel, panel.Column)


def test_track_tricolor_echogram_integration(get_data):
    # Load sample data for testing
    MVBS_ds = get_data

    # Create the panels
    MVBS_ds.eshader.control_mode_select.value = False

    track_panel = panel.Row(
        panel.Column(MVBS_ds.eshader.tile_select),
        MVBS_ds.eshader.track(),
    )

    tricolor_echogram_panel = panel.Row(
        panel.Column(MVBS_ds.eshader.Sv_range_slider),
        MVBS_ds.eshader.echogram(
            channel=[
                "GPT 120 kHz 00907205a6d0 4-1 ES120-7C",
                "GPT  38 kHz 009072058146 2-1 ES38B",
                "GPT  18 kHz 009072058c8d 1-1 ES18-11",
            ],
            rgb_composite=True,
        ),
    )

    integration_panel = panel.Column(
        track_panel,
        tricolor_echogram_panel,
    )

    # Check if the panel is created without raising an exception
    assert isinstance(integration_panel, panel.Column)

    MVBS_ds.eshader.control_mode_select.value = True

    assert isinstance(integration_panel, panel.Column)


def test_track_integration(get_data):
    # Load sample data for testing
    MVBS_ds = get_data

    # Create the panels
    MVBS_ds.eshader.control_mode_select.value = False

    track_panel = panel.Row(
        panel.Column(MVBS_ds.eshader.tile_select),
        MVBS_ds.eshader.track(),
    )

    echogram_panel = panel.Row(
        panel.Column(MVBS_ds.eshader.Sv_range_slider, MVBS_ds.eshader.colormap),
        MVBS_ds.eshader.echogram(),
    )

    tricolor_echogram_panel = panel.Row(
        panel.Column(MVBS_ds.eshader.Sv_range_slider),
        MVBS_ds.eshader.echogram(
            channel=[
                "GPT 120 kHz 00907205a6d0 4-1 ES120-7C",
                "GPT  38 kHz 009072058146 2-1 ES38B",
                "GPT  18 kHz 009072058c8d 1-1 ES18-11",
            ],
            rgb_composite=True,
        ),
    )

    integration_panel = panel.Column(
        MVBS_ds.eshader.control_mode_select,
        track_panel,
        echogram_panel,
        tricolor_echogram_panel,
    )

    # Check if the panel is created without raising an exception
    assert isinstance(integration_panel, panel.Column)

    MVBS_ds.eshader.control_mode_select.value = True

    assert isinstance(integration_panel, panel.Column)


def test_curtain(get_data):
    # Load sample data for testing
    MVBS_ds = get_data

    curtain_panel = panel.Row(
        panel.Column(MVBS_ds.eshader.channel_select, MVBS_ds.eshader.curtain_ratio),
        MVBS_ds.eshader.curtain(),
    )

    # Remember to set panel extension to "pyvista" when showing 2.5D curtain
    panel.extension("pyvista")

    # Check if the panel is created without raising an exception
    assert isinstance(curtain_panel, panel.Row)


def test_curtain_echogram_integration(get_data):
    # Load sample data for testing
    MVBS_ds = get_data

    # Create the panels
    MVBS_ds.eshader.control_mode_select.value = True

    echogram_panel = panel.Row(
        panel.Column(MVBS_ds.eshader.Sv_range_slider, MVBS_ds.eshader.colormap),
        MVBS_ds.eshader.echogram(),
    )

    curtain_panel = panel.Row(
        panel.Column(
            MVBS_ds.eshader.channel_select,
            MVBS_ds.eshader.curtain_ratio,
        ),
        MVBS_ds.eshader.curtain(),
    )

    # Remember to set panel extension to "pyvista" when showing 2.5D curtain
    panel.extension("pyvista")

    integration_panel = panel.Column(
        echogram_panel,
        curtain_panel,
    )

    # Check if the panel is created without raising an exception
    assert isinstance(integration_panel, panel.Column)


def test_curtain_tricolor_echogram_integration(get_data):
    # Load sample data for testing
    MVBS_ds = get_data

    # Create the panels
    MVBS_ds.eshader.control_mode_select.value = True

    tricolor_echogram_panel = panel.Row(
        panel.Column(MVBS_ds.eshader.Sv_range_slider),
        MVBS_ds.eshader.echogram(
            channel=[
                "GPT 120 kHz 00907205a6d0 4-1 ES120-7C",
                "GPT  38 kHz 009072058146 2-1 ES38B",
                "GPT  18 kHz 009072058c8d 1-1 ES18-11",
            ],
            rgb_composite=True,
        ),
    )

    curtain_panel = panel.Row(
        panel.Column(
            MVBS_ds.eshader.channel_select,
            MVBS_ds.eshader.curtain_ratio,
        ),
        MVBS_ds.eshader.curtain(),
    )

    # Remember to set panel extension to "pyvista" when showing 2.5D curtain
    panel.extension("pyvista")

    integration_panel = panel.Column(
        tricolor_echogram_panel,
        curtain_panel,
    )

    # Check if the panel is created without raising an exception
    assert isinstance(integration_panel, panel.Column)


def test_curtain_track_integration(get_data):
    # Load sample data for testing
    MVBS_ds = get_data

    # Create the panels
    MVBS_ds.eshader.control_mode_select.value = False

    track_panel = panel.Row(
        panel.Column(MVBS_ds.eshader.tile_select),
        MVBS_ds.eshader.track(),
    )

    curtain_panel = panel.Row(
        panel.Column(
            MVBS_ds.eshader.channel_select,
            MVBS_ds.eshader.curtain_ratio,
        ),
        MVBS_ds.eshader.curtain(),
    )

    # Remember to set panel extension to "pyvista" when showing 2.5D curtain
    panel.extension("pyvista")

    integration_panel = panel.Column(
        track_panel,
        curtain_panel,
    )

    # Check if the panel is created without raising an exception
    assert isinstance(integration_panel, panel.Column)


def test_curtain_integration(get_data):
    # Load sample data for testing
    MVBS_ds = get_data

    # Create the panels
    MVBS_ds.eshader.control_mode_select.value = False

    track_panel = panel.Row(
        panel.Column(MVBS_ds.eshader.tile_select),
        MVBS_ds.eshader.track(),
    )

    echogram_panel = panel.Row(
        panel.Column(MVBS_ds.eshader.Sv_range_slider, MVBS_ds.eshader.colormap),
        MVBS_ds.eshader.echogram(),
    )

    tricolor_echogram_panel = panel.Row(
        panel.Column(MVBS_ds.eshader.Sv_range_slider),
        MVBS_ds.eshader.echogram(
            channel=[
                "GPT 120 kHz 00907205a6d0 4-1 ES120-7C",
                "GPT  38 kHz 009072058146 2-1 ES38B",
                "GPT  18 kHz 009072058c8d 1-1 ES18-11",
            ],
            rgb_composite=True,
        ),
    )

    curtain_panel = panel.Row(
        panel.Column(
            MVBS_ds.eshader.channel_select,
            MVBS_ds.eshader.curtain_ratio,
        ),
        MVBS_ds.eshader.curtain(),
    )

    # Remember to set panel extension to "pyvista" when showing 2.5D curtain
    panel.extension("pyvista")

    integration_panel = panel.Column(
        MVBS_ds.eshader.control_mode_select,
        track_panel,
        echogram_panel,
        tricolor_echogram_panel,
        curtain_panel,
    )

    # Check if the panel is created without raising an exception
    assert isinstance(integration_panel, panel.Column)

    MVBS_ds.eshader.control_mode_select.value = True

    assert isinstance(integration_panel, panel.Column)


def test_hist(get_data):
    # Load sample data for testing
    MVBS_ds = get_data

    stats_panel = panel.Row(
        panel.Column(
            MVBS_ds.eshader.bin_size_input,
            MVBS_ds.eshader.overlay_layout_toggle,
        ),
        panel.Column(
            MVBS_ds.eshader.hist(),
            MVBS_ds.eshader.table(),
        ),
    )

    # Check if the panel is created without raising an exception
    assert isinstance(stats_panel, panel.Row)


def test_echogram_integration(get_data):
    # Load sample data for testing
    MVBS_ds = get_data

    # Create the panels
    MVBS_ds.eshader.control_mode_select.value = True

    echogram_panel = panel.Row(
        panel.Column(
            MVBS_ds.eshader.Sv_range_slider,
            MVBS_ds.eshader.colormap,
        ),
        MVBS_ds.eshader.echogram(),
    )

    stats_panel = panel.Row(
        panel.Column(
            MVBS_ds.eshader.bin_size_input,
            MVBS_ds.eshader.overlay_layout_toggle,
        ),
        panel.Column(
            MVBS_ds.eshader.hist(),
            MVBS_ds.eshader.table(),
        ),
    )

    integration_panel = panel.Column(
        echogram_panel,
        stats_panel,
    )

    # Check if the panel is created without raising an exception
    assert isinstance(integration_panel, panel.Column)


def test_hist_tricolor_echogram_integration(get_data):
    # Load sample data for testing
    MVBS_ds = get_data

    # Create the panels
    MVBS_ds.eshader.control_mode_select.value = True

    tricolor_echogram_panel = panel.Row(
        panel.Column(MVBS_ds.eshader.Sv_range_slider),
        MVBS_ds.eshader.echogram(
            channel=[
                "GPT 120 kHz 00907205a6d0 4-1 ES120-7C",
                "GPT  38 kHz 009072058146 2-1 ES38B",
                "GPT  18 kHz 009072058c8d 1-1 ES18-11",
            ],
            rgb_composite=True,
        ),
    )

    stats_panel = panel.Row(
        panel.Column(
            MVBS_ds.eshader.bin_size_input,
            MVBS_ds.eshader.overlay_layout_toggle,
        ),
        panel.Column(
            MVBS_ds.eshader.hist(),
            MVBS_ds.eshader.table(),
        ),
    )

    integration_panel = panel.Column(
        tricolor_echogram_panel,
        stats_panel,
    )

    # Check if the panel is created without raising an exception
    assert isinstance(integration_panel, panel.Column)


def test_hist_track_integration(get_data):
    # Load sample data for testing
    MVBS_ds = get_data

    # Create the panels
    MVBS_ds.eshader.control_mode_select.value = False

    track_panel = panel.Row(
        panel.Column(MVBS_ds.eshader.tile_select),
        MVBS_ds.eshader.track(),
    )

    stats_panel = panel.Row(
        panel.Column(
            MVBS_ds.eshader.bin_size_input,
            MVBS_ds.eshader.overlay_layout_toggle,
        ),
        panel.Column(
            MVBS_ds.eshader.hist(),
            MVBS_ds.eshader.table(),
        ),
    )

    integration_panel = panel.Column(
        track_panel,
        stats_panel,
    )

    # Check if the panel is created without raising an exception
    assert isinstance(integration_panel, panel.Column)


def test_hist_integration(get_data):
    # Load sample data for testing
    MVBS_ds = get_data

    # Create the panels
    MVBS_ds.eshader.control_mode_select.value = False

    track_panel = panel.Row(
        panel.Column(MVBS_ds.eshader.tile_select),
        MVBS_ds.eshader.track(),
    )

    echogram_panel = panel.Row(
        panel.Column(
            MVBS_ds.eshader.Sv_range_slider,
            MVBS_ds.eshader.colormap,
        ),
        MVBS_ds.eshader.echogram(),
    )

    tricolor_echogram_panel = panel.Row(
        panel.Column(MVBS_ds.eshader.Sv_range_slider),
        MVBS_ds.eshader.echogram(
            channel=[
                "GPT 120 kHz 00907205a6d0 4-1 ES120-7C",
                "GPT  38 kHz 009072058146 2-1 ES38B",
                "GPT  18 kHz 009072058c8d 1-1 ES18-11",
            ],
            rgb_composite=True,
        ),
    )

    stats_panel = panel.Row(
        panel.Column(
            MVBS_ds.eshader.bin_size_input,
            MVBS_ds.eshader.overlay_layout_toggle,
        ),
        panel.Column(
            MVBS_ds.eshader.hist(),
            MVBS_ds.eshader.table(),
        ),
    )

    integration_panel = panel.Column(
        MVBS_ds.eshader.control_mode_select,
        track_panel,
        echogram_panel,
        tricolor_echogram_panel,
        stats_panel,
    )

    # Check if the panel is created without raising an exception
    assert isinstance(integration_panel, panel.Column)

    MVBS_ds.eshader.control_mode_select.value = True

    assert isinstance(integration_panel, panel.Column)


def test_integration(get_data):
    # Load sample data for testing
    MVBS_ds = get_data

    # Create the panels
    MVBS_ds.eshader.control_mode_select.value = False

    track_panel = panel.Row(
        panel.Column(MVBS_ds.eshader.tile_select),
        MVBS_ds.eshader.track(),
    )

    echogram_panel = panel.Row(
        panel.Column(
            MVBS_ds.eshader.Sv_range_slider,
            MVBS_ds.eshader.colormap,
        ),
        MVBS_ds.eshader.echogram(),
    )

    tricolor_echogram_panel = panel.Row(
        panel.Column(MVBS_ds.eshader.Sv_range_slider),
        MVBS_ds.eshader.echogram(
            channel=[
                "GPT 120 kHz 00907205a6d0 4-1 ES120-7C",
                "GPT  38 kHz 009072058146 2-1 ES38B",
                "GPT  18 kHz 009072058c8d 1-1 ES18-11",
            ],
            rgb_composite=True,
        ),
    )

    curtain_panel = panel.Row(
        panel.Column(
            MVBS_ds.eshader.channel_select,
            MVBS_ds.eshader.curtain_ratio,
        ),
        MVBS_ds.eshader.curtain(),
    )

    # Remember to set panel extension to "pyvista" when showing 2.5D curtain
    panel.extension("pyvista")

    stats_panel = panel.Row(
        panel.Column(
            MVBS_ds.eshader.bin_size_input,
            MVBS_ds.eshader.overlay_layout_toggle,
        ),
        panel.Column(
            MVBS_ds.eshader.hist(),
            MVBS_ds.eshader.table(),
        ),
    )

    integration_panel = panel.Column(
        MVBS_ds.eshader.control_mode_select,
        track_panel,
        echogram_panel,
        tricolor_echogram_panel,
        curtain_panel,
        stats_panel,
    )

    # Check if the panel is created without raising an exception
    assert isinstance(integration_panel, panel.Column)

    MVBS_ds.eshader.control_mode_select.value = True

    assert isinstance(integration_panel, panel.Column)
