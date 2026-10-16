import hvplot.xarray  # noqa
import xarray


def hist(
    MVBS_ds: xarray.Dataset,
    bins: int = 24,
    overlay: bool = True,
):
    """
    Create histograms of Sv by acoustic channel.

    Parameters
    ----------
    MVBS_ds : xarray.Dataset
        Dataset containing the ``Sv`` variable.
    bins : int, optional
        Number of histogram bins. Default is 24.
    overlay : bool, optional
        Overlay channels when True; otherwise arrange them vertically.

    Returns
    -------
    holoviews object
        Histogram visualization.
    """
    if overlay:
        return MVBS_ds.Sv.hvplot.hist(
            "Sv",
            by="channel",
            bins=bins,
            subplots=False,
            alpha=0.6,
            legend="top",
        )

    return MVBS_ds.Sv.hvplot.hist(
        "Sv",
        by="channel",
        bins=bins,
        subplots=True,
        legend="top",
    ).cols(1)
