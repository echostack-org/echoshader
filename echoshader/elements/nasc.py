import holoviews
import xarray


def nasc(
    ds: xarray.Dataset,
    var_name: str = "NASC",
    x_dim: str = "distance",
    channel: str | None = None,
):
    """Create a NASC plot."""

    data = ds[var_name]

    if channel is not None and "channel" in data.dims:
        data = data.sel(channel=channel)

    data = data.squeeze(drop=True)

    if data.ndim != 1:
        raise ValueError(
            "NASC visualization expects a one-dimensional variable "
            "after channel selection and squeezing."
        )

    if x_dim not in data.coords:
        x_dim = data.dims[0]

    return holoviews.Curve(
        (
            data[x_dim].values,
            data.values,
        ),
        kdims=[x_dim],
        vdims=[var_name],
    )
