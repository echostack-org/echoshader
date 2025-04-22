import holoviews
import hvplot.xarray  # noqa
import pandas
import xarray

from .utils import gram_opts


def hist_plot(MVBS_ds: xarray.Dataset, bins: int = 24, overlay: bool = True):
    """
    Create and display a histogram plot for the 'Sv' data in the given xarray dataset.

    Parameters
    ----------
    MVBS_ds : xarray.Dataset
        A dataset containing the 'Sv' data for the histogram plot.

    bins : int, optional
        Number of bins to use for the histogram. Default is 24.

    overlay : bool, optional
        If True, multiple histograms will be overlaid on the same plot.
        If False, each histogram will be plotted vertically. Default is True.

    Returns
    -------
    holoviews.core.overlay.Overlay
        The histogram plot as a HoloViews Overlay object.

    Notes
    -----
    This function uses HoloViews' hvplot library to generate the histogram plot based on the
    'Sv' data in the provided xarray dataset `MVBS_ds`. The 'Sv' data should contain the values
    to be plotted.

    If `overlay` is set to True, the histogram plot will show multiple histograms, each
    representing a different channel from the dataset, stacked on top of each other.
    If set to False, each channel will have its histogram plotted separately.

    Example
    -------
        hist = hist_plot(MVBS_ds, bins=30, overlay=False)
        panel.Row(hist)
    """
    if overlay is True:
        hist = MVBS_ds.Sv.hvplot.hist(
            "Sv",
            by="channel",
            bins=bins,
            subplots=False,
            alpha=0.6,
            legend="top",
        ).opts(gram_opts)
    else:
        hist = (
            MVBS_ds.Sv.hvplot.hist(
                "Sv",
                by="channel",
                bins=bins,
                subplots=True,
                legend="top",
            )
            .opts(gram_opts)
            .cols(1)
        )

    return hist


def table_plot(MVBS_ds: xarray.Dataset):
    """
    Create and display a table containing summary statistics for the 'Sv' data in the given
    xarray dataset.

    Parameters
    ----------
    MVBS_ds : xarray.Dataset
        A dataset containing the 'Sv' data for the table.

    Returns
    -------
    holoviews.Table
        A HoloViews Table object containing summary statistics such as mean, min, max,
        skewness, and kurtosis for each channel's 'Sv' data.

    Notes
    -----
    This function calculates summary statistics for the 'Sv' data in the provided xarray
    dataset `MVBS_ds`, including mean, standard deviation, minimum, 25th percentile, median,
    75th percentile, and maximum. Additionally, it computes the skewness and kurtosis for
    each channel's 'Sv' data.

    The resulting table will display summary statistics for each channel in the dataset,
    with an additional row for the skewness and kurtosis values.

    Example
    -------
        table = table_plot(MVBS_ds)
        panel.Row(table)
    """
    obj_df_sum = MVBS_ds.Sv.to_dataframe()

    skew_sum = obj_df_sum["Sv"].skew()
    kurt_sum = obj_df_sum["Sv"].kurt()

    obj_desc = obj_df_sum.describe().reset_index()[["index", "Sv"]]

    obj_desc = obj_desc.rename(columns={"Sv": "Sum"})
    obj_desc.loc[len(obj_desc)] = ["skew", skew_sum]
    obj_desc.loc[len(obj_desc)] = ["kurtosis", kurt_sum]

    for channel in MVBS_ds.channel.values:
        obj_df_channel = MVBS_ds.sel(channel=channel).Sv.to_dataframe()

        skew_channel = obj_df_channel["Sv"].skew()
        kurt_channel = obj_df_channel["Sv"].kurt()

        obj_df_channel = obj_df_channel.describe().reset_index()[["index", "Sv"]]

        obj_df_channel = obj_df_channel.rename(columns={"Sv": channel})
        obj_df_channel.loc[len(obj_df_channel)] = ["skew", skew_channel]
        obj_df_channel.loc[len(obj_df_channel)] = ["kurtosis", kurt_channel]

        obj_desc = pandas.merge(obj_desc, obj_df_channel, on="index")

    table = holoviews.Table(obj_desc).opts(gram_opts)

    return table
