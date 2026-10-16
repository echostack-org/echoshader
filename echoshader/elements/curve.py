import holoviews


def curve(
    data,
    x: str,
    y: str,
):
    return holoviews.Curve(
        data,
        kdims=[x],
        vdims=[y],
    )
