import numpy
import xarray


def _normalize_coordinate_values(
    values,
):
    """Convert dataset coordinates to numeric values."""

    values = numpy.asarray(values)

    if numpy.issubdtype(
        values.dtype,
        numpy.datetime64,
    ):
        return values.astype("datetime64[ns]").astype("int64")

    return values.astype(float)


def _normalize_selection_values(
    values,
    reference,
):
    """
    Convert Bokeh/HoloViews selection coordinates to dataset units.

    Bokeh represents datetimes as milliseconds since the Unix epoch,
    while xarray commonly stores datetime64 values at nanosecond
    precision.
    """

    values = numpy.asarray(values)

    reference = numpy.asarray(reference)

    if numpy.issubdtype(
        reference.dtype,
        numpy.datetime64,
    ):
        if not numpy.issubdtype(
            values.dtype,
            numpy.datetime64,
        ):
            return (
                values.astype("int64")
                .astype("datetime64[ms]")
                .astype("datetime64[ns]")
                .astype("int64")
            )

        return values.astype("datetime64[ns]").astype("int64")

    return values.astype(float)


def _broadcast_coordinates(
    ds: xarray.Dataset,
    x: str,
    y: str,
):
    """Broadcast two coordinates to a common shape."""

    if x not in ds:
        raise ValueError(f"Coordinate {x!r} is not present in the dataset.")

    if y not in ds:
        raise ValueError(f"Coordinate {y!r} is not present in the dataset.")

    return xarray.broadcast(
        ds[x],
        ds[y],
    )


def point_index_mask(
    ds: xarray.Dataset,
    index,
    dim: str = "ping_time",
    x: str = "longitude",
    y: str = "latitude",
):
    """
    Convert selected map-point indices into a 1-D dataset mask.

    HoloViews/Bokeh performs the geographic hit-testing. We only map
    the selected point indices back to the original dataset rows.
    """

    if index is None:
        return None

    index = numpy.asarray(
        index,
        dtype=int,
    )

    if index.size == 0:
        return None

    if dim not in ds.dims:
        raise ValueError(f"Dimension {dim!r} is not present in the dataset.")

    if x not in ds or y not in ds:
        raise ValueError(f"Coordinates {x!r} and {y!r} must be present.")

    x_values = numpy.asarray(ds[x].values)

    y_values = numpy.asarray(ds[y].values)

    valid = numpy.isfinite(x_values) & numpy.isfinite(y_values)

    valid_positions = numpy.flatnonzero(valid)

    index = index[(index >= 0) & (index < len(valid_positions))]

    if index.size == 0:
        return None

    selected_positions = valid_positions[index]

    mask = numpy.zeros(
        ds.sizes[dim],
        dtype=bool,
    )

    mask[selected_positions] = True

    return xarray.DataArray(
        mask,
        coords={
            dim: ds[dim],
        },
        dims=(dim,),
    )


def box_dimension_mask(
    ds: xarray.Dataset,
    bounds,
    dim: str = "ping_time",
    axis: str = "x",
):
    """Convert plot bounds into a 1-D coordinate mask."""

    if bounds is None:
        return None

    if dim not in ds:
        raise ValueError(f"Coordinate {dim!r} is not present.")

    left, bottom, right, top = bounds

    if axis == "x":
        lower = left
        upper = right

    elif axis == "y":
        lower = bottom
        upper = top

    else:
        raise ValueError("'axis' must be 'x' or 'y'.")

    coordinate = ds[dim]

    values = _normalize_coordinate_values(coordinate.values)

    lower, upper = _normalize_selection_values(
        [
            lower,
            upper,
        ],
        coordinate.values,
    )

    mask = (values >= min(lower, upper)) & (values <= max(lower, upper))

    if not numpy.any(mask):
        return None

    return xarray.DataArray(
        mask,
        coords=coordinate.coords,
        dims=coordinate.dims,
    )


def polygon_mask(
    ds: xarray.Dataset,
    geometry,
    x: str = "ping_time",
    y: str = "echo_range",
):
    """Return an exact mask for an arbitrary polygon."""

    if geometry is None:
        return None

    geometry = numpy.asarray(geometry)

    if geometry.size == 0 or len(geometry) < 3:
        return None

    x_coord, y_coord = _broadcast_coordinates(
        ds,
        x=x,
        y=y,
    )

    x_values = _normalize_coordinate_values(x_coord.values)

    y_values = _normalize_coordinate_values(y_coord.values)

    polygon_x = _normalize_selection_values(
        geometry[:, 0],
        ds[x].values,
    )

    polygon_y = _normalize_selection_values(
        geometry[:, 1],
        ds[y].values,
    )

    mask = _points_in_polygon(
        x_values,
        y_values,
        polygon_x,
        polygon_y,
    )

    if not numpy.any(mask):
        return None

    return xarray.DataArray(
        mask,
        coords=x_coord.coords,
        dims=x_coord.dims,
    )


def project_mask(
    mask: xarray.DataArray,
    keep_dims,
):
    """Project a mask onto selected dimensions using logical any."""

    if isinstance(
        keep_dims,
        str,
    ):
        keep_dims = [
            keep_dims,
        ]

    keep_dims = set(keep_dims)

    reduce_dims = [dim for dim in mask.dims if dim not in keep_dims]

    if reduce_dims:
        mask = mask.any(dim=reduce_dims)

    return mask


def apply_selection(
    ds: xarray.Dataset,
    selection,
    projection=None,
    drop: bool = True,
):
    """Apply a shared selection to a dataset."""

    if selection is None:
        return ds

    mask = selection

    if projection is not None:
        mask = project_mask(
            mask,
            keep_dims=projection,
        )

    return ds.where(
        mask,
        drop=drop,
    )


def _points_in_polygon(
    x,
    y,
    polygon_x,
    polygon_y,
):
    """Return a point-in-polygon mask using ray casting."""

    inside = numpy.zeros(
        numpy.broadcast(
            x,
            y,
        ).shape,
        dtype=bool,
    )

    x = numpy.broadcast_to(
        x,
        inside.shape,
    )

    y = numpy.broadcast_to(
        y,
        inside.shape,
    )

    j = len(polygon_x) - 1

    for i in range(len(polygon_x)):
        xi = polygon_x[i]
        yi = polygon_y[i]

        xj = polygon_x[j]
        yj = polygon_y[j]

        denominator = yj - yi

        if denominator == 0:
            denominator = numpy.finfo(float).eps

        intersects = ((yi > y) != (yj > y)) & (x < ((xj - xi) * (y - yi) / denominator + xi))

        inside ^= intersects

        j = i

    return inside


def point_indices_from_mask(
    ds: xarray.Dataset,
    mask,
    dim: str = "ping_time",
    x: str = "longitude",
    y: str = "latitude",
):
    """
    Convert a shared selection mask back into displayed map-point indices.

    This is the inverse operation of point_index_mask().
    """

    if mask is None:
        return []

    mask = project_mask(
        mask,
        keep_dims=dim,
    )

    mask = mask.reindex(
        {
            dim: ds[dim],
        },
        fill_value=False,
    )

    selected_positions = numpy.flatnonzero(
        numpy.asarray(
            mask.values,
            dtype=bool,
        )
    )

    x_values = numpy.asarray(ds[x].values)

    y_values = numpy.asarray(ds[y].values)

    valid = numpy.isfinite(x_values) & numpy.isfinite(y_values)

    valid_positions = numpy.flatnonzero(valid)

    position_to_display_index = {position: index for index, position in enumerate(valid_positions)}

    return [
        position_to_display_index[position]
        for position in selected_positions
        if position in position_to_display_index
    ]


SELECTIONS = {
    "point_select": point_index_mask,
    "box_select": box_dimension_mask,
    "lasso_select": polygon_mask,
}
