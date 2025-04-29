import logging
import warnings
from typing import List, Optional, Union

import holoviews
import numpy
import panel
import param
import xarray
from bokeh.util.warnings import BokehUserWarning

from .box import get_box_plot, get_box_stream
from .curtain import curtain_plot
from .echogram import single_echogram, tricolor_echogram
from .hist import hist_plot, table_plot
from .map import convert_EPSG, get_track_corners, tile_plot, track_plot
from .utils import curtain_opts, tiles

warnings.simplefilter(action="ignore", category=BokehUserWarning)
warnings.simplefilter("ignore", category=RuntimeWarning)
logging.getLogger("param").setLevel(logging.CRITICAL)

panel.extension("pyvista")
holoviews.extension("bokeh", logo=False)


@xarray.register_dataset_accessor("eshader")
class Echoshader(param.Parameterized):
    """
    Echoshader - A visualization tool for acoustic data analysis.

    This class provides a comprehensive visualization toolset for analyzing acoustic data using
    various visualizations such as echograms, tracks, curtains, histograms, and tables.

    Attributes
    ----------
        colormap : panel.widgets.LiteralInput
            A widget to control the colormap for echograms.
            https://holoviews.org/user_guide/Colormaps.html

        Sv_range_slider : panel.widgets.EditableRangeSlider
            A slider widget to control the Sv range.

        tile_select : panel.widgets.Select
            A dropdown widget to select the map tile.
            https://holoviews.org/reference/elements/bokeh/Tiles.html

        channel_select : panel.widgets.Select
            A dropdown widget to select the frequency channel.

        curtain_ratio : panel.widgets.FloatInput
            A numeric input widget for controlling curtain ratio.

        bin_size_input : panel.widgets.IntInput
            An input widget for controlling histogram bin size.

        overlay_layout_toggle : panel.widgets.Toggle
            A toggle widget for overlay and layout options.

        control_mode_select : panel.widgets.Select
            A dropdown widget to switch between control modes.

    Methods
    -------
        echogram(channel, cmap, vmin, vmax, rgb_composite, vert_dim, opts):
            Display echogram plots based on channel and options.

        track(tile, control, opts):
            Display track plots with specified tile and options.

        curtain(channel, ratio, **opts):
            Display curtain plots based on channel and curtain ratio.

        hist(bins, overlay, opts):
            Display histogram plots with specified bin size and overlay option.

        table(opts):
            Display data summary table.

        get_data_from_box():
            Get the data from the selected box (gram or track).

    Note:
        The Echoshader class allows users to interactively explore acoustic data using
        different visualization techniques.

        Users can control various parameters such as colormap, Sv range, tile selection,
        channel selection, curtain ratio, bin size, and overlay options.

        The class provides plots for echograms, tracks, curtains, histograms, and tables.

        The control mode can be switched between "Echograms Control" and "Tracks Control".
    """

    def __init__(self, MVBS_ds: xarray.Dataset):
        super().__init__()

        self.MVBS_ds = MVBS_ds

        self._check_input()

        self._init_widget()

        self._init_param()

    def _check_input(self):
        # Check if 'Sv' exists
        if "Sv" not in self.MVBS_ds.variables:
            raise ValueError("Dataset must contain a variable named 'Sv'.")

        # Check if 'Sv' has correct dimensions
        expected_dims = ("channel", "ping_time", "echo_range")
        actual_dims = self.MVBS_ds["Sv"].dims
        if actual_dims != expected_dims:
            raise ValueError(
                f"'Sv' must have dimensions {expected_dims}, but got {actual_dims}."
            )

    def _init_widget(self):
        self.colormap = panel.widgets.LiteralInput(
            name="Colormap", value="jet", type=(str, list)
        )

        self.Sv_range_slider = panel.widgets.EditableRangeSlider(
            name="Sv Range Slider",
            start=self.MVBS_ds.Sv.actual_range[0],
            end=self.MVBS_ds.Sv.actual_range[-1],
            value=(self.MVBS_ds.Sv.actual_range[0], self.MVBS_ds.Sv.actual_range[-1]),
            step=0.01,
        )

        self.tile_select = panel.widgets.Select(
            name="Map Tile Select", value="OSM", options=tiles
        )

        self.channel_select = panel.widgets.Select(
            name="Channel Select", options=self.MVBS_ds.channel.values.tolist()
        )

        self.curtain_ratio = panel.widgets.FloatInput(
            name="Ratio Input (Z Spacing)", value=0.001, step=1e-5, start=0
        )

        self.bin_size_input = panel.widgets.IntInput(
            name="Bin Size Input", value=24, step=10, start=0
        )

        self.overlay_layout_toggle = panel.widgets.Toggle(
            name="Overlay & Layout Toggle", value=True
        )

        self.control_mode_select = panel.widgets.Select(
            name="Control Mode Select",
            options={
                "Echograms Control": True,
                "Tracks Control": False,
            },
        )

    def _init_param(self):
        self.gram_box_stream = holoviews.streams.BoundsXY()

        self.gram_bounds = get_box_plot(self.gram_box_stream)

        self.update_gram_flag = holoviews.streams.Counter()

        self.update_track_flag = holoviews.streams.Counter()

        self.MVBS_ds_in_gram_box = self.MVBS_ds

        self.MVBS_ds_in_track_box = self.MVBS_ds

    def echogram(
        self,
        channel: List[str] = None,
        cmap: Union[str, List[str]] = None,
        vmin: float = None,
        vmax: float = None,
        rgb_composite: bool = False,
        vert_dim: Optional[str] = "echo_range",
        opts=[],
    ):
        """
        Display echogram plots based on specified parameters.

        Attached Widgets
        ----------------
        colormap : panel.widgets.LiteralInput
            A widget to control the colormap for echograms.
            https://holoviews.org/user_guide/Colormaps.html

        Sv_range_slider : panel.widgets.EditableRangeSlider
            A slider widget to control the Sv range.

        Parameters
        ----------
        channel : List[str], optional
            List of frequency channels. Default is None.
        cmap : Union[str, List[str]], optional
            Colormap for the echogram plot. Default is None.
            https://holoviews.org/user_guide/Colormaps.html
        vmin : float, optional
            Minimum value for Sv range. Default is None.
        vmax : float, optional
            Maximum value for Sv range. Default is None.
        rgb_composite : bool, optional
            Enable RGB tricolor echogram. Default is False.
        vert_dim : str, optional
            Name of the vertical dimension. Default is echo_range.
        opts : list[holoviews.opts], optional
            Additional options for plotting. Default is an empty list.
            https://holoviews.org/user_guide/Applying_Customizations.html#option-list-syntax

        Returns
        -------
        holoviews.Overlay
            Echogram plot.

        Examples
        --------
        echogram = MVBS_ds.eshader.echogram(vmin = -80, vmax = -30)

        panel.Row(echogram)
        """

        if cmap is not None:
            self.colormap.value = cmap

        data_vmin = self.Sv_range_slider.value[0]
        data_vmax = self.Sv_range_slider.value[1]

        self.Sv_range_slider.value = (
            vmin if vmin is not None else data_vmin,
            vmax if vmax is not None else data_vmax,
        )

        self.gram_opts = opts

        self.vert_dim = vert_dim

        if rgb_composite is True:
            if channel is None or len(channel) != 3:
                raise ValueError(
                    "Must have exactly 3 frequency channels for tricolor echogram."
                )

            self.tri_channel = channel

            return self._tricolor_echogram_plot

        else:
            if channel is None:
                self.channel = self.MVBS_ds.channel.values.tolist()
            else:
                self.channel = channel

            self.channel_select.options = self.channel

            return self._echogram_plot

    def _update_gram_box(self, bounds):
        """
        Update the gram box based on given bounds.

        Parameters
        ----------
        bounds : tuple
            Bounds of the gram box in the format (left, bottom, right, top).
        """
        self.gram_box_stream.update(bounds=bounds)

        self.MVBS_ds_in_gram_box = self._extract_data_from_gram_box(bounds)

        if self.control_mode_select.value is True:
            self.update_track_flag.event()

    def _update_gram_reset(self, resetting):
        """
        Event handler for resetting the gram box.

        Parameters
        ----------
        resetting : bool
            The value indicating a reset event.
        """
        self.update_gram_flag.event()

    def _extract_data_from_gram_box(self, bounds):
        """
        Extract data from the gram box based on given bounds.

        Parameters
        ----------
        bounds : tuple
            Bounds of the gram box in the format (left, bottom, right, top).

        Returns
        -------
        xarray.Dataset
            Extracted dataset within the specified bounds.
        """
        if bounds is None:
            MVBS_ds_in_gram_box = self.MVBS_ds

        else:
            MVBS_ds_in_gram_box = self.MVBS_ds.sel(
                {
                    "ping_time": slice(bounds[0], bounds[2]),
                    self.vert_dim: (
                        slice(bounds[1], bounds[3])
                        if bounds[3] > bounds[1]
                        else slice(bounds[3], bounds[1])
                    ),
                },
            )

        return MVBS_ds_in_gram_box

    @param.depends(
        "Sv_range_slider.value",
        "update_gram_flag.counter",
    )
    def _tricolor_echogram_plot(self):
        """
        Generate a tricolor echogram plot based on current parameters.

        Returns
        -------
        holoviews.Overlay
            Tricolor echogram plot.
        """

        if self.control_mode_select.value is True:
            MVBS_ds = self.MVBS_ds
        else:
            MVBS_ds = self.MVBS_ds_in_track_box

        rgb_map = {}
        rgb_map[self.tri_channel[0]] = "R"
        rgb_map[self.tri_channel[1]] = "G"
        rgb_map[self.tri_channel[2]] = "B"

        echogram = tricolor_echogram(
            MVBS_ds,
            self.Sv_range_slider.value[0],
            self.Sv_range_slider.value[1],
            rgb_map,
            self.vert_dim,
        )

        if self.control_mode_select.value is False:
            MVBS_ds_with_time_range = MVBS_ds.dropna(dim="ping_time", how="all")

            one_hour = numpy.timedelta64(1, "h")

            echogram.opts(
                xlim=(
                    MVBS_ds_with_time_range.ping_time.values[0] - one_hour,
                    MVBS_ds_with_time_range.ping_time.values[-1] + one_hour,
                )
            )

        # get box stream from echogram
        box_stream = get_box_stream(echogram)

        # add subscriber to update unified box select
        box_stream.add_subscriber(self._update_gram_box)

        # set inital value of box stream
        self._update_gram_box(tuple(echogram.lbrt))

        reset_stream = holoviews.streams.PlotReset(source=echogram)

        reset_stream.add_subscriber(self._update_gram_reset)

        bounds = self.gram_bounds

        return (echogram * bounds).opts(self.gram_opts)

    @param.depends(
        "Sv_range_slider.value",
        "colormap.value",
        "update_gram_flag.counter",
    )
    def _echogram_plot(self):
        """
        Generate an echogram plot based on current parameters.

        Returns
        -------
        holoviews.Layout
            Layout of echogram plots.
        """

        if self.control_mode_select.value is True:
            MVBS_ds = self.MVBS_ds
        else:
            MVBS_ds = self.MVBS_ds_in_track_box

        echograms_list = []

        for channel in self.channel:
            echogram = single_echogram(
                MVBS_ds,
                channel,
                self.colormap.value,
                self.Sv_range_slider.value,
                self.vert_dim,
            )

            if self.control_mode_select.value is False:
                MVBS_ds_with_time_range = MVBS_ds.dropna(dim="ping_time", how="all")

                one_hour = numpy.timedelta64(1, "h")

                echogram.opts(
                    xlim=(
                        MVBS_ds_with_time_range.ping_time.values[0] - one_hour,
                        MVBS_ds_with_time_range.ping_time.values[-1] + one_hour,
                    )
                )

            # get box stream from echogram
            box_stream = get_box_stream(echogram)

            # add subscriber to update unified box select
            box_stream.add_subscriber(self._update_gram_box)

            echograms_list.append(echogram)

        # set inital value of box stream
        self._update_gram_box(tuple(echograms_list[0].lbrt))

        reset_stream = holoviews.streams.PlotReset(source=echograms_list[0])

        reset_stream.add_subscriber(self._update_gram_reset)

        # get echograms stack
        echograms = holoviews.Layout(echograms_list).cols(1)

        bounds = self.gram_bounds

        return (echograms * bounds).opts(self.gram_opts)

    def track(
        self,
        tile: str = None,
        control: bool = False,
        opts=[],
    ):
        """
        Display track plots based on specified parameters.

        Attached Widgets
        ----------------
        tile_select : panel.widgets.Select
            A dropdown widget to select the map tile.
            https://holoviews.org/reference/elements/bokeh/Tiles.html

        Parameters
        ----------
        tile : str, optional
            Map tile for the track plot. Default is None.
        control : bool, optional
            Control mode for unified selection. Default is False.
        opts : list, optional
            Additional options for plotting. Default is an empty list.

        Returns
        -------
        holoviews.Overlay
            Track plot.

        Examples
        --------
        track = MVBS_ds.eshader.track(tile = "OSM", control = True)

        panel.Row(track)
        """
        if tile is not None:
            self.tile_select.value = tile

        self.control_mode_select.value = control

        self.track_opts = opts

        return self._track_tile_plot

    def _extract_data_from_track_box(self, bounds):
        """
        Extract data from the track box based on given bounds.

        Parameters
        ----------
        bounds : tuple
            Bounds of the track box in the format (left, bottom, right, top).

        Returns
        -------
        xarray.Dataset
            Extracted dataset within the specified bounds.
        """
        if bounds is None or (bounds[0] == bounds[2] or bounds[1] == bounds[3]):
            MVBS_ds_in_track_box = self.MVBS_ds
        else:
            MVBS_ds_in_track_box = self.MVBS_ds.where(
                (self.MVBS_ds.longitude > bounds[0])
                & (self.MVBS_ds.latitude > bounds[1])
                & (self.MVBS_ds.longitude < bounds[2])
                & (self.MVBS_ds.latitude < bounds[3])
            )

        return MVBS_ds_in_track_box

    def _update_track_reset(self, resetting):
        """
        Event handler for resetting the track plot.

        Parameters
        ----------
        resetting : boolean
            The value indicating a reset event.
        """
        self.update_track_flag.event()

    def _update_track_box(self, bounds):
        """
        Update the dataset within the track box based on given bounds.

        Parameters
        ----------
        bounds : tuple
            Bounds of the track box in the format (left, bottom, right, top).
        """
        self.MVBS_ds_in_track_box = self._extract_data_from_track_box(bounds)

        if self.control_mode_select.value is False:
            self.update_gram_flag.event()

    @param.depends(
        "tile_select.value",
        "update_track_flag.counter",
    )
    def _track_tile_plot(self):
        """
        Generate a track plot with tile based on current parameters.

        Returns
        -------
        holoviews.Overlay
            Track plot with tile.
        """
        track = self._track_plot() * self._tile_plot()

        return track.opts(self.track_opts)

    def _tile_plot(self):
        """
        Generate a map tile plot based on current parameters.

        Returns
        -------
        holoviews.Overlay
            Map tile plot.
        """
        if self.control_mode_select.value is True:
            MVBS_ds = self.MVBS_ds_in_gram_box
        else:
            MVBS_ds = self.MVBS_ds

        tile = tile_plot(self.tile_select.value)

        left, bottom, right, top = get_track_corners(MVBS_ds)

        bottom, left = convert_EPSG(lat=bottom, lon=left, mercator_to_coord=False)
        top, right = convert_EPSG(lat=top, lon=right, mercator_to_coord=False)

        center_lon = (left + right) / 2
        center_lat = (bottom + top) / 2
        center = (center_lon, center_lat, center_lon, center_lat)

        self.tile_box_stream = get_box_stream(tile, center)

        reset_stream = holoviews.streams.PlotReset(source=tile)

        reset_stream.add_subscriber(self._update_track_reset)

        tile_bounds = get_box_plot(self.tile_box_stream)

        return tile * tile_bounds

    def _track_plot(self):
        """
        Generate a track plot based on current parameters.

        Returns
        -------
        holoviews.Overlay
            Track plot.
        """
        if self.control_mode_select.value is True:
            MVBS_ds = self.MVBS_ds_in_gram_box
        else:
            MVBS_ds = self.MVBS_ds

        track = track_plot(MVBS_ds)

        left, bottom, right, top = get_track_corners(MVBS_ds)

        self.track_box_stream = get_box_stream(track, (left, bottom, right, top))

        self.track_box_stream.add_subscriber(self._update_track_box)

        self._update_track_box((left, bottom, right, top))

        return track

    def curtain(
        self,
        channel: str = None,
        ratio: float = None,
        **opts,
    ):
        """
        Display curtain plots based on specified parameters.

        Attached Widgets
        ----------------
        colormap : panel.widgets.LiteralInput
            A widget to control the colormap for echograms.
            https://holoviews.org/user_guide/Colormaps.html

        Sv_range_slider : panel.widgets.EditableRangeSlider
            A slider widget to control the Sv range.

        channel_select : panel.widgets.Select
            A dropdown widget to select the frequency channel.

        curtain_ratio : panel.widgets.FloatInput
            A numeric input widget for controlling curtain ratio.


        Parameters
        ----------
        channel : str, optional
            Frequency channel for curtain plot. Default is None.
        ratio : float, optional
            Curtain ratio for spacing. Default is None.
        opts : dict, optional
            Additional options for plotting.

        Returns
        -------
        panel.panel
            Curtain plot panel.

        Examples
        --------
        curtain = MVBS_ds.eshader.curtain(channel = "GPT 38 kHz 00907208dd13 5-1 OOI.38|200")

        panel.Row(curtain)
        """
        if channel is not None:
            self.channel_select.value = channel

        if ratio is not None:
            self.curtain_ratio.value = ratio

        self.curtain_opts = opts

        return self._curtain_plot

    @param.depends(
        "colormap.value",
        "Sv_range_slider.value",
        "channel_select.value",
        "curtain_ratio.value",
        "update_gram_flag.counter",
        "update_track_flag.counter",
    )
    def _curtain_plot(self):
        """
        Generate a curtain plot based on current parameters.

        Returns
        -------
        panel.panel
            Curtain plot panel.
        """
        if self.control_mode_select.value is True:
            MVBS_ds = self.MVBS_ds_in_gram_box
        else:
            MVBS_ds = self.MVBS_ds_in_track_box

        curtain = curtain_plot(
            MVBS_ds=MVBS_ds.sel(channel=self.channel_select.value),
            cmap=self.colormap.value,
            clim=self.Sv_range_slider.value,
            ratio=self.curtain_ratio.value,
        )

        if "width" not in self.curtain_opts:
            self.curtain_opts["width"] = curtain_opts["width"]

        if "height" not in self.curtain_opts:
            self.curtain_opts["height"] = curtain_opts["height"]

        if "orientation_widget" not in self.curtain_opts:
            self.curtain_opts["orientation_widget"] = True

        curtain_panel = panel.panel(
            curtain.ren_win,
            **self.curtain_opts,
        )

        return curtain_panel

    def hist(
        self,
        bins: int = None,
        overlay: bool = None,
        opts=[],
    ):
        """
        Display histogram plots based on specified parameters.

        Attached Widgets
        ----------------
        bin_size_input : panel.widgets.IntInput
            An input widget for controlling histogram bin size.

        overlay_layout_toggle : panel.widgets.Toggle
            A toggle widget for overlay and layout options.

        Parameters
        ----------
        bins : int, optional
            Number of bins for the histogram. Default is None.
        overlay : bool, optional
            Overlay multiple histograms. Default is None.
        opts : list, optional
            Additional options for plotting. Default is an empty list.

        Returns
        -------
        holoviews.Overlay
            Histogram plot.

        Examples
        --------
        histogram = MVBS_ds.eshader.hist(bins = 50)

        panel.Row(histogram)
        """
        if bins is not None:
            self.bin_size_input.value = bins

        if overlay is not None:
            self.overlay_layout_toggle.value = overlay

        self.hist_opts = opts

        return self._hist_plot

    @param.depends(
        "bin_size_input.value",
        "overlay_layout_toggle.value",
        "update_gram_flag.counter",
        "update_track_flag.counter",
    )
    def _hist_plot(self):
        """
        Generate a histogram plot based on current parameters.

        Returns
        -------
        holoviews.Overlay
            Histogram plot.
        """
        if self.control_mode_select.value is True:
            MVBS_ds = self.MVBS_ds_in_gram_box
        else:
            MVBS_ds = self.MVBS_ds_in_track_box

        hist = hist_plot(
            MVBS_ds,
            bins=self.bin_size_input.value,
            overlay=self.overlay_layout_toggle.value,
        )

        return hist.opts(self.hist_opts)

    def table(
        self,
        opts=[],
    ):
        """
        Display data summary table.

        Parameters
        ----------
        opts : list, optional
            Additional options for plotting. Default is an empty list.

        Returns
        -------
        holoviews.Table
            Data summary table.

        Examples
        --------
        table = MVBS_ds.eshader.table()

        panel.Row(table)
        """
        self.table_opts = opts

        return self._table_plot

    @param.depends(
        "update_gram_flag.counter",
        "update_track_flag.counter",
    )
    def _table_plot(self):
        """
        Generate a data summary table.

        Returns
        -------
        holoviews.Table
            Data summary table.
        """
        if self.control_mode_select.value is True:
            MVBS_ds = self.MVBS_ds_in_gram_box
        else:
            MVBS_ds = self.MVBS_ds_in_track_box

        table = table_plot(MVBS_ds=MVBS_ds)

        return table.opts(self.table_opts)

    def get_data_from_box(self):
        """
        Get the data from the currently selected box (gram or track).

        Returns
        -------
        xarray.Dataset
            Selected dataset from echogram or track.
        """
        if self.control_mode_select.value is True:
            return self.MVBS_ds_in_gram_box
        else:
            return self.MVBS_ds_in_track_box
