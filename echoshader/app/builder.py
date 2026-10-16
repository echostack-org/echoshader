import holoviews
import numpy
import panel as pn
import xarray
from bokeh.models import BoxSelectTool, GlyphRenderer

from echoshader.elements.track import track_points

from .context import AppContext
from .controls import build_control
from .interactions import INTERACTIONS
from .registry import ELEMENTS
from .selection import SELECTIONS, apply_selection, project_mask
from .sources import DataSource, build_source
from .state import AppState
from .summary import selection_metadata, selection_summary


def _build_sources(config: dict, context: AppContext):
    """Build configured data sources."""
    source_configs = config.get("data", {})
    data_sources = {}
    pending = dict(source_configs)

    while pending:
        progress = False

        for name in list(pending):
            source_config = pending[name]
            dependency = source_config.get("depends_on")

            if dependency is not None and dependency not in data_sources:
                continue

            source = build_source(source_config, context=context)
            source.reload()
            data_sources[name] = source
            del pending[name]
            progress = True

        if not progress:
            raise ValueError(f"Could not resolve data-source dependencies: {list(pending)}")

    for name, source_config in source_configs.items():
        dependency = source_config.get("depends_on")
        if dependency is None:
            continue

        parent = data_sources[dependency]
        child = data_sources[name]
        parent.param.watch(
            lambda event, child=child: child.reload(),
            "data",
        )

    for source in data_sources.values():
        source.start_refresh()

    return data_sources


def _get_data(source):
    """Return reactive or static source data."""
    if isinstance(source, DataSource):
        return source.param.data
    return source


def _get_current_data(source):
    """Return current concrete source data."""
    if isinstance(source, DataSource):
        return source.data
    if callable(source):
        return source()
    return source


def _resolve_control_options(control_config: dict, data_sources: dict):
    """Populate widget options from a dataset coordinate."""
    config_copy = dict(control_config)
    options_from = config_copy.pop("options_from", None)

    if options_from is None:
        return config_copy

    data_name = options_from["data"]
    coord_name = options_from["coord"]

    if data_name not in data_sources:
        raise ValueError(f"Unknown control options data source: {data_name!r}.")

    data = _get_current_data(data_sources[data_name])

    if not isinstance(data, xarray.Dataset):
        raise TypeError("Control options_from currently requires an xarray.Dataset.")

    if coord_name not in data.coords:
        raise ValueError(f"Coordinate {coord_name!r} is not present in data source {data_name!r}.")

    values = [
        value.item() if hasattr(value, "item") else value
        for value in data[coord_name].values.tolist()
    ]

    args = dict(config_copy.get("args", {}))
    args["options"] = values
    config_copy["args"] = args

    return config_copy


def _build_controls(
    config: dict,
    state: AppState,
    data_sources: dict,
):
    """Build controls and connect them to shared values."""
    controls = {}

    for name, control_config in config.get("controls", {}).items():
        config_copy = _resolve_control_options(
            control_config,
            data_sources,
        )

        write_name = config_copy.pop("write", None)
        args = dict(config_copy.get("args", {}))

        if write_name is not None:
            shared_value = state.get_value(write_name)

            if shared_value.value is not None and "value" not in args:
                args["value"] = shared_value.value

        config_copy["args"] = args
        control = build_control(config_copy)
        controls[name] = control

        if write_name is None:
            continue

        shared_value = state.get_value(write_name)

        if shared_value.value is None:
            shared_value.value = control.value
        elif shared_value.value in list(control.options):
            control.value = shared_value.value
        elif len(control.options):
            control.value = next(iter(control.options))
            shared_value.value = control.value

        def update_value(event, value_state=shared_value):
            value_state.value = event.new

        control.param.watch(update_value, "value")

    return controls


def _apply_element_options(element, options: dict):
    """Apply options to simple or composite HoloViews elements."""
    if not options:
        return element

    options = dict(options)

    if isinstance(options.get("clim"), list):
        options["clim"] = tuple(options["clim"])

    if not isinstance(
        element,
        (
            holoviews.Layout,
            holoviews.NdLayout,
        ),
    ):
        return element.opts(**options)

    option_specs = []
    handled_types = set()
    container_types = {
        "Layout",
        "NdLayout",
        "Overlay",
        "NdOverlay",
    }

    for child in element.traverse(lambda child: child):
        child_type = type(child).__name__

        if child_type in container_types or child_type in handled_types:
            continue

        try:
            option_type = getattr(
                holoviews.opts,
                child_type,
            )
        except AttributeError:
            continue

        option_specs.append(option_type(**options))
        handled_types.add(child_type)

    if not option_specs:
        return element

    return element.opts(*option_specs)


def _apply_view(data, view_config: dict | None):
    """Apply display-only data reduction."""
    if not view_config:
        return data

    max_pings = view_config.get("max_pings")
    if max_pings is None:
        return data

    ping_dim = view_config.get(
        "ping_dim",
        "ping_time",
    )

    if ping_dim not in data.dims:
        return data

    n_pings = data.sizes[ping_dim]

    if n_pings <= max_pings:
        return data

    indices = numpy.linspace(
        0,
        n_pings - 1,
        max_pings,
        dtype=int,
    )

    return data.isel({ping_dim: indices})


def _build_element(
    element_config: dict,
    data,
    runtime_values: dict | None = None,
):
    """Build one visualization element."""
    element_type = element_config["type"]

    if element_type not in ELEMENTS:
        raise ValueError(f"Unknown element type: {element_type!r}.")

    data = _apply_view(
        data,
        element_config.get("view"),
    )

    args = dict(
        element_config.get(
            "args",
            {},
        )
    )

    if runtime_values:
        args.update(runtime_values)

    plot = ELEMENTS[element_type](
        data,
        **args,
    )

    return _apply_element_options(
        plot,
        dict(element_config.get("opts", {})),
    )


def _parse_observe(element_config: dict):
    """Normalize observed state configuration."""
    observe = element_config.get(
        "observe",
        {},
    )

    if observe is None:
        observe = {}

    if isinstance(observe, str):
        observe = {
            "selection": observe,
        }

    if not isinstance(observe, dict):
        raise TypeError("'observe' must be a mapping.")

    mode = observe.get(
        "mode",
        "filter",
    )

    if mode not in {
        "filter",
        "highlight",
    }:
        raise ValueError(f"Unknown selection observation mode: {mode!r}.")

    return {
        "selection": observe.get("selection"),
        "projection": observe.get("projection"),
        "drop": observe.get("drop", True),
        "mode": mode,
        "highlight": observe.get("highlight", {}),
        "values": observe.get("values", {}),
    }


def _resolve_value_dependencies(
    state: AppState,
    values_config: dict,
):
    """Convert shared-value observation into Param dependencies."""
    dependencies = {}
    specifications = {}

    for argument_name, value_names in values_config.items():
        if isinstance(value_names, str):
            value_names = [value_names]

        if not isinstance(value_names, list):
            raise TypeError("Observed values must be a value name or list of value names.")

        dependency_names = []

        for index, value_name in enumerate(value_names):
            dependency_name = f"value__{argument_name}__{index}"

            dependencies[dependency_name] = state.get_value(value_name).param.value

            dependency_names.append(dependency_name)

        specifications[argument_name] = dependency_names

    return dependencies, specifications


def _build_runtime_values(
    bindings: dict,
    specifications: dict,
):
    """Build element function values from shared state."""
    runtime_values = {}

    for argument_name, dependency_names in specifications.items():
        values = [bindings[dependency_name] for dependency_name in dependency_names]

        runtime_values[argument_name] = values[0] if len(values) == 1 else values

    return runtime_values


def _selection_intervals(
    source_data: xarray.Dataset,
    selection,
    projection: str,
):
    """
    Return contiguous selected coordinate intervals.

    A spatial map selection can correspond to several separate time
    intervals along a survey track, so we preserve each contiguous run
    rather than collapsing everything to one first/last span.
    """
    if selection is None:
        return []

    mask = project_mask(
        selection,
        keep_dims=projection,
    )

    mask = mask.reindex(
        {
            projection: source_data[projection],
        },
        fill_value=False,
    )

    selected = numpy.asarray(
        mask.values,
        dtype=bool,
    )

    if not selected.any():
        return []

    indices = numpy.flatnonzero(selected)

    # split non-consecutive selections
    breaks = numpy.where(numpy.diff(indices) > 1)[0] + 1

    runs = numpy.split(
        indices,
        breaks,
    )

    values = source_data[projection].values

    return [
        (
            values[run[0]],
            values[run[-1]],
        )
        for run in runs
        if len(run)
    ]


def _selection_overlay(
    source_data: xarray.Dataset,
    selection,
    observe: dict,
):
    """
    Create lightweight vector highlights for a shared time selection.

    Always returns a HoloViews Overlay. DynamicMap requires its callback
    to return the same object type on every update; returning Overlay when
    empty and VSpan when selected causes the rendering failure seen before.
    """
    projection = observe.get("projection") or "ping_time"

    intervals = _selection_intervals(
        source_data,
        selection,
        projection,
    )

    if not intervals:
        return holoviews.Overlay([])

    style = observe.get(
        "highlight",
        {},
    )

    color = style.get(
        "color",
        "dodgerblue",
    )

    fill_alpha = style.get(
        "fill_alpha",
        0.08,
    )

    line_alpha = style.get(
        "line_alpha",
        0.9,
    )

    line_width = style.get(
        "line_width",
        2,
    )

    spans = [
        holoviews.VSpan(
            start,
            end,
        ).opts(
            color=color,
            alpha=fill_alpha,
            line_color=color,
            line_alpha=line_alpha,
            line_width=line_width,
        )
        for start, end in intervals
    ]

    return holoviews.Overlay(spans)


def _build_reactive_element(
    state: AppState,
    element_config: dict,
    data_source,
):
    """
    Build an element observing shared state.

    Filter mode applies the selection to the data. Highlight mode keeps
    the acoustic raster intact and adds only a lightweight vector overlay.
    """
    data = _get_data(data_source)
    observe = _parse_observe(element_config)

    dependencies = {
        "data": data,
    }

    selection_name = observe["selection"]

    if selection_name is not None and (
        observe["mode"] == "filter" or element_config["type"] == "tricolor_echogram"
    ):
        dependencies["selection_value"] = state.get_selection(selection_name).param.value

    (
        value_dependencies,
        value_specifications,
    ) = _resolve_value_dependencies(
        state,
        observe["values"],
    )

    dependencies.update(value_dependencies)

    def build(
        data,
        selection_value=None,
        **bindings,
    ):
        source_data = data

        if selection_name is not None and observe["mode"] == "filter":
            data = apply_selection(
                data,
                selection_value,
                projection=observe["projection"],
                drop=observe["drop"],
            )

        runtime_values = _build_runtime_values(
            bindings,
            value_specifications,
        )

        plot = _build_element(
            element_config,
            data,
            runtime_values=runtime_values,
        )

        # rebuild RGB when its channels or selection change
        if (
            element_config["type"] == "tricolor_echogram"
            and selection_name is not None
            and observe["mode"] == "highlight"
        ):
            return plot * _selection_overlay(
                source_data,
                selection_value,
                observe,
            )

        return plot

    return pn.bind(
        build,
        **dependencies,
    )


def _build_component(
    component_config: dict,
    state: AppState,
    data_sources: dict,
):
    """Build read-only application components."""
    component_type = component_config["type"]
    data_name = component_config["data"]

    if data_name not in data_sources:
        raise ValueError(f"Unknown data source: {data_name!r}.")

    data = _get_data(data_sources[data_name])

    selection_name = component_config.get("selection")

    selection_value = None

    if selection_name is not None:
        selection_value = state.get_selection(selection_name).param.value

    args = dict(
        component_config.get(
            "args",
            {},
        )
    )

    if component_type == "selection_summary":
        return pn.bind(
            selection_summary,
            data=data,
            selection=selection_value,
            **args,
        )

    if component_type == "selection_metadata":
        return pn.bind(
            selection_metadata,
            data=data,
            selection=selection_value,
            **args,
        )

    raise ValueError(f"Unknown component type: {component_type!r}.")


def _layout_children(plot):
    """Return display children from a simple element or Layout."""
    if isinstance(
        plot,
        (
            holoviews.Layout,
            holoviews.NdLayout,
        ),
    ):
        return list(plot.values())

    return [plot]


def _get_interaction_sources(
    plot,
    element_name: str | None = None,
):
    """Return visualization elements that can own streams."""
    elements = plot.traverse(lambda element: element)

    if element_name is not None:
        matches = [element for element in elements if type(element).__name__ == element_name]

        if not matches:
            available = sorted({type(element).__name__ for element in elements})

            raise ValueError(
                f"Could not find interaction element "
                f"{element_name!r}. "
                f"Available elements: {available}."
            )

        return matches

    if not isinstance(
        plot,
        (
            holoviews.Layout,
            holoviews.NdLayout,
        ),
    ):
        return [plot]

    container_types = {
        "Layout",
        "NdLayout",
        "Overlay",
        "NdOverlay",
    }

    return [element for element in elements if type(element).__name__ not in container_types]


def _interaction_points(element):
    """
    Add an invisible Points renderer covering one raster.

    BoxSelectTool is restricted to this renderer, so Bokeh never applies
    native selected/non-selected styling to Image or QuadMesh glyphs.
    """
    x_dim = element.kdims[0].name
    y_dim = element.kdims[1].name

    x0, x1 = element.range(x_dim)
    y0, y1 = element.range(y_dim)

    return holoviews.Points(
        (
            [x0, x1, x0, x1],
            [y0, y0, y1, y1],
        ),
        kdims=[
            x_dim,
            y_dim,
        ],
    ).opts(
        alpha=0.0,
        size=1,
        nonselection_alpha=0.0,
        selection_alpha=0.0,
    )


def _add_box_interaction_layers(plot):
    """Overlay an invisible Points interaction layer on each echogram."""
    children = _layout_children(plot)
    wrapped = [child * _interaction_points(child) for child in children]

    if len(wrapped) == 1:
        return wrapped[0]

    return holoviews.Layout(wrapped).cols(1)


def _style_box_select_tool(plot, element):
    """Keep only an outline and restrict box select to point renderers."""
    point_renderers = []

    for renderer in plot.state.renderers:
        if not isinstance(
            renderer,
            GlyphRenderer,
        ):
            continue

        glyph_name = type(renderer.glyph).__name__

        if glyph_name in {
            "Scatter",
            "Circle",
        }:
            point_renderers.append(renderer)

    for tool in plot.state.tools:
        if not isinstance(
            tool,
            BoxSelectTool,
        ):
            continue

        if point_renderers:
            tool.renderers = point_renderers

        overlay = tool.overlay
        overlay.fill_alpha = 0.0
        overlay.line_alpha = 0.9
        overlay.line_color = "dodgerblue"
        overlay.line_width = 2
        overlay.line_dash = [
            6,
            4,
        ]


def _add_interaction_tools(
    plot,
    tools: list,
    active_tools: list,
    element_name: str | None = None,
):
    """Add a minimal interaction toolbar."""
    if not tools:
        return plot

    hooks = [_style_box_select_tool] if "box_select" in tools else []

    if element_name is not None:
        option_type = getattr(
            holoviews.opts,
            element_name,
        )

        return plot.opts(
            option_type(
                tools=tools,
                active_tools=active_tools,
                hooks=hooks,
            )
        )

    if not isinstance(
        plot,
        (
            holoviews.Layout,
            holoviews.NdLayout,
        ),
    ):
        return plot.opts(
            tools=tools,
            active_tools=active_tools,
            hooks=hooks,
        )

    option_specs = []
    handled_types = set()

    for element in _get_interaction_sources(plot):
        element_type = type(element).__name__

        if element_type in handled_types:
            continue

        try:
            option_type = getattr(
                holoviews.opts,
                element_type,
            )
        except AttributeError:
            continue

        option_specs.append(
            option_type(
                tools=tools,
                active_tools=active_tools,
                hooks=hooks,
            )
        )

        handled_types.add(element_type)

    if not option_specs:
        return plot

    return plot.opts(*option_specs)


def _selection_is_empty(value):
    """Return whether a generated selection is empty."""
    if value is None:
        return True

    if isinstance(
        value,
        xarray.DataArray,
    ):
        return not bool(value.any().item())

    return False


def _build_interactions(
    config: dict,
    state: AppState,
    data_sources: dict,
    objects: dict,
    element_configs: dict,
):
    """
    Attach interactions that write named shared selections.

    Multiple interactions can write the same state; the latest writer wins.
    """
    for interaction in config.get(
        "interactions",
        [],
    ):
        interaction_type = interaction["type"]
        source_name = interaction["source"]
        selection_name = interaction["write"]

        if interaction_type not in INTERACTIONS:
            raise ValueError(f"Unknown interaction type: {interaction_type!r}.")

        if interaction_type not in SELECTIONS:
            raise ValueError(f"No selector registered for {interaction_type!r}.")

        source_object = objects[source_name]
        source_plot = source_object() if callable(source_object) else source_object

        element_config = element_configs[source_name]

        source_element = interaction.get("source_element")

        # use an invisible points layer for echogram box selection
        if interaction_type == "box_select" and element_config["type"] == "echogram":
            source_plot = _add_box_interaction_layers(source_plot)
            source_element = "Points"

        interaction_definition = INTERACTIONS[interaction_type]

        source_plot = _add_interaction_tools(
            source_plot,
            tools=interaction_definition.get(
                "tools",
                [],
            ),
            active_tools=interaction_definition.get(
                "active_tools",
                [],
            ),
            element_name=source_element,
        )

        # keep streams attached to the rendered objects
        objects[source_name] = source_plot

        stream_sources = _get_interaction_sources(
            source_plot,
            element_name=source_element,
        )

        source_data_name = element_config["data"]

        source_data = data_sources[source_data_name]

        selector = SELECTIONS[interaction_type]

        parameter_name = interaction_definition["parameter"]

        selection_args = interaction.get(
            "args",
            {},
        )

        selection = state.get_selection(selection_name)

        for stream_source in stream_sources:
            stream = interaction_definition["stream"](stream_source)

            def update_selection(
                event,
                *,
                selector=selector,
                source_data=source_data,
                parameter_name=parameter_name,
                selection_args=selection_args,
                selection=selection,
                source_name=source_name,
                interaction_type=interaction_type,
            ):
                data = _get_current_data(source_data)

                value = selector(
                    data,
                    **{
                        parameter_name: event.new,
                    },
                    **selection_args,
                )

                if _selection_is_empty(value):
                    selection.clear()
                    return

                selection.set(
                    value,
                    source=source_name,
                    kind=interaction_type,
                )

            stream.param.watch(
                update_selection,
                parameter_name,
            )

            selection.add_stream(stream)


def _selection_pipe(selection):
    """Create a Pipe kept in sync with one shared selection."""
    pipe = holoviews.streams.Pipe(data=selection.value)

    def update_pipe(event):
        pipe.send(event.new)

    selection.param.watch(
        update_pipe,
        "value",
    )

    selection.add_stream(pipe)

    return pipe


def _build_track_highlight(
    plot,
    source_data,
    selection,
    observe: dict,
):
    """Overlay the shared ping selection on the full black track."""
    pipe = _selection_pipe(selection)

    highlight = observe["highlight"]
    projection = observe["projection"] or "ping_time"

    color = highlight.get(
        "color",
        "dodgerblue",
    )

    size = highlight.get(
        "size",
        7,
    )

    def selected_points(data):
        if data is None:
            selected = source_data.isel({projection: slice(0, 0)})
        else:
            selected = apply_selection(
                source_data,
                data,
                projection=projection,
                drop=True,
            )

        return track_points(selected).opts(
            color=color,
            size=size,
            alpha=1.0,
        )

    highlight_plot = holoviews.DynamicMap(
        selected_points,
        streams=[pipe],
    )

    return plot * highlight_plot


def _build_time_highlight(
    plot,
    source_data,
    selection,
    observe: dict,
):
    """Overlay one cheap shared-selection span on each echogram."""
    pipe = _selection_pipe(selection)

    def highlight(data):
        return _selection_overlay(
            source_data,
            data,
            observe,
        )

    overlay = holoviews.DynamicMap(
        highlight,
        streams=[pipe],
    )

    children = _layout_children(plot)
    highlighted = [child * overlay for child in children]

    if len(highlighted) == 1:
        return highlighted[0]

    return holoviews.Layout(highlighted).cols(1)


def _build_selection_views(
    state: AppState,
    data_sources: dict,
    objects: dict,
    element_configs: dict,
):
    """Add lightweight visual observers of shared selections."""
    for name, element_config in element_configs.items():
        observe = _parse_observe(element_config)

        if observe["selection"] is None or observe["mode"] != "highlight":
            continue

        selection = state.get_selection(observe["selection"])

        data = _get_current_data(data_sources[element_config["data"]])

        element_type = element_config["type"]

        if element_type == "track":
            objects[name] = _build_track_highlight(
                objects[name],
                data,
                selection,
                observe,
            )

        elif element_type == "echogram":
            objects[name] = _build_time_highlight(
                objects[name],
                data,
                selection,
                observe,
            )

        elif element_type == "tricolor_echogram":
            # RGB highlight is already handled inside its reactive
            # element because the RGB channel widgets are reactive too.
            continue

        else:
            raise ValueError(
                "Highlight observation is currently supported for "
                "'track', 'echogram', and "
                "'tricolor_echogram' elements."
            )


def _resolve_layout_child(
    child,
    objects: dict,
):
    """Resolve one layout child."""
    if isinstance(child, str):
        if child not in objects:
            raise ValueError(f"Unknown layout object: {child!r}.")

        return objects[child]

    if isinstance(child, dict):
        return _build_layout(
            child,
            objects,
        )

    raise TypeError("Layout children must be names or nested layout dictionaries.")


def _build_layout(
    layout_config: dict,
    objects: dict,
):
    """Build recursive Panel content layout."""
    layout_type = layout_config.get(
        "type",
        "column",
    )

    children = layout_config.get(
        "children",
        [],
    )

    kwargs = {}

    sizing_mode = layout_config.get("sizing_mode")

    if sizing_mode is not None:
        kwargs["sizing_mode"] = sizing_mode

    if layout_type == "column":
        return pn.Column(
            *[
                _resolve_layout_child(
                    child,
                    objects,
                )
                for child in children
            ],
            **kwargs,
        )

    if layout_type == "row":
        return pn.Row(
            *[
                _resolve_layout_child(
                    child,
                    objects,
                )
                for child in children
            ],
            **kwargs,
        )

    if layout_type == "tabs":
        tabs = []

        for child in children:
            title = child.get(
                "title",
                "Untitled",
            )

            tab_config = dict(child)
            tab_config.pop(
                "title",
                None,
            )

            tab_config.setdefault(
                "type",
                "column",
            )

            tabs.append(
                (
                    title,
                    _build_layout(
                        tab_config,
                        objects,
                    ),
                )
            )

        return pn.Tabs(
            *tabs,
            dynamic=layout_config.get(
                "dynamic",
                False,
            ),
            **kwargs,
        )

    raise ValueError(f"Unknown layout type: {layout_type!r}.")


def _build_template(
    config: dict,
    objects: dict,
    main,
):
    """Build the permanent application shell."""
    app_config = config.get(
        "app",
        {},
    )

    template_name = app_config.get(
        "template",
        "fast_list",
    )

    if template_name != "fast_list":
        raise ValueError(f"Unknown app template: {template_name!r}.")

    template = pn.template.FastListTemplate(
        title=app_config.get(
            "title",
            "Echoshader",
        ),
        sidebar_width=app_config.get(
            "sidebar_width",
            420,
        ),
    )

    subtitle = app_config.get("subtitle")

    if subtitle:
        template.header.append(
            pn.pane.Markdown(
                subtitle,
                margin=(8, 15),
            )
        )

    for child in config.get(
        "sidebar",
        {},
    ).get(
        "children",
        [],
    ):
        template.sidebar.append(
            _resolve_layout_child(
                child,
                objects,
            )
        )

    template.main.append(main)

    return template


def build_app(
    config: dict,
    datasets: dict[str, xarray.Dataset] | None = None,
    context: dict | None = None,
):
    """Build an Echoshader application."""
    state = AppState()

    context_values = dict(
        config.get(
            "context",
            {},
        )
    )

    if context is not None:
        context_values.update(context)

    app_context = AppContext(values=context_values)

    for name, value in config.get(
        "values",
        {},
    ).items():
        state.get_value(
            name,
            default=value,
        )

    for name in config.get(
        "selections",
        {},
    ):
        state.get_selection(name)

    data_sources = _build_sources(
        config=config,
        context=app_context,
    )

    if datasets is not None:
        data_sources.update(datasets)

    for name, source in data_sources.items():
        state.set_dataset(
            name,
            source,
        )

    controls = _build_controls(
        config=config,
        state=state,
        data_sources=data_sources,
    )

    element_configs = config.get(
        "elements",
        {},
    )

    objects = {}

    for name, element_config in element_configs.items():
        data_name = element_config["data"]

        objects[name] = _build_reactive_element(
            state=state,
            element_config=element_config,
            data_source=data_sources[data_name],
        )

    for name, component_config in config.get(
        "components",
        {},
    ).items():
        objects[name] = _build_component(
            component_config=component_config,
            state=state,
            data_sources=data_sources,
        )

    _build_interactions(
        config=config,
        state=state,
        data_sources=data_sources,
        objects=objects,
        element_configs=element_configs,
    )

    _build_selection_views(
        state=state,
        data_sources=data_sources,
        objects=objects,
        element_configs=element_configs,
    )

    objects.update(controls)

    main = _build_layout(
        config.get(
            "layout",
            {
                "type": "column",
                "children": list(objects),
            },
        ),
        objects,
    )

    app = _build_template(
        config=config,
        objects=objects,
        main=main,
    )

    app._echoshader_state = state

    return app
