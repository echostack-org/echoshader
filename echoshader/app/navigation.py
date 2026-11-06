"""Keep native Panel axis linking scoped to one group of echograms."""


def navigation_group():
    def hook(plot, element):
        prefix = f"echoshader_{id(hook)}_"
        for name in ("x_range", "y_range"):
            axis = plot.handles[name]
            # Bokeh limits navigation to its initial frame without recalculating
            # the Datashader element extents through apply_hard_bounds.
            axis.bounds = "auto"
            if axis.tags:
                tag = axis.tags[0]
                if not str(tag[0][0]).startswith(prefix):
                    axis.tags = [
                        tuple((prefix + str(item[0]), *item[1:]) for item in tag),
                        *axis.tags[1:],
                    ]

    return hook


def attach_navigation(plot, hook):
    """Keep existing formatters/hooks while enabling native Panel axis linking."""
    hooks = plot.opts.get("plot", backend="bokeh").kwargs.get("hooks", [])
    return plot.opts(hooks=[*hooks, hook], shared_axes=True)
