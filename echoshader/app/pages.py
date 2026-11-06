"""Small Panel page shell; acoustic elements and mission readers stay separate."""

from copy import deepcopy
from importlib import import_module

import panel as pn
import xarray as xr

from .data import open_source, source_revision
from .lazy import lazy_view
from .refresh import watch_changes


def _factory(path):
    module, name = path.split(":", 1)
    return getattr(import_module(module), name)


def build_pages(config, datasets=None, *, config_dir=None):
    from .builder import _build_file_app, _data_path

    template = pn.template.FastListTemplate(
        title="Echoshader",
        theme_toggle=False,
        busy_indicator=None,
        sidebar_width=330,
        main=[],
    )
    file_sidebar = pn.Column(sizing_mode="stretch_width")
    template.sidebar.append(file_sidebar)
    viewers = []
    extension_sources = []

    pn.state.on_session_destroyed(
        lambda context: [data.close() for data in (*viewers, *extension_sources)]
    )

    def file_page():
        settings = config.get("file_catalogue", {})
        entries = {}
        menu = pn.widgets.Select(name="File", options=["Latest completed transect"])
        refresh = pn.widgets.Button(name="Refresh files", width=150)
        panel = pn.Column(sizing_mode="stretch_width")
        controls = pn.Column(sizing_mode="stretch_width")
        notice = pn.pane.Alert(visible=False)
        name = config.get("app", {}).get("source", "survey")
        state = {"signature": None, "menu_update": False}

        def entry_revision():
            entry = entries.get(menu.value)
            if callable(entry):
                # A loader may expose a cheap revision without opening its dataset.
                return (menu.value, getattr(entry, "revision", menu.value))
            if entry is None and datasets and name in datasets:
                return (menu.value, "provided-dataset")
            value = entry if entry is not None else config["data"][name]["path"]
            path = _data_path(value, config_dir)
            return (menu.value, str(path), source_revision(path))

        def load(event=None):
            if state["menu_update"]:
                return
            selected_datasets = datasets
            owned = False
            panel.loading = True
            try:
                signature = entry_revision()
                if signature == state["signature"]:
                    return
                local = deepcopy(config)
                local.pop("extensions", None)
                entry = entries.get(menu.value)
                if entry is not None:
                    if callable(entry):
                        opened = entry()
                        selected_datasets = {name: opened}
                        owned = True
                        local["data"][name].pop("path", None)
                    else:
                        local["data"][name]["path"] = str(entry)
                        selected_datasets = None
                    local["data"][name]["label"] = menu.value
                if not selected_datasets:
                    source_settings = local["data"][name]
                    path = _data_path(source_settings["path"], config_dir)
                    opened = open_source(source_settings, path)
                    selected_datasets = {name: opened}
                    owned = True
                child = _build_file_app(local, selected_datasets, config_dir=config_dir)
                if entry_revision() != signature:
                    raise OSError("Source is being updated; retrying on the next refresh")
                for old in viewers:
                    old.close()
                viewers.clear()
                if owned:
                    viewers.append(selected_datasets[name])
                controls.objects = list(child.sidebar)
                panel.objects = list(child.main)
                state["signature"] = signature
                notice.visible = False
            except (
                ImportError,
                KeyError,
                OSError,
                RuntimeError,
                TypeError,
                ValueError,
            ) as exc:
                if owned and selected_datasets:
                    selected_datasets[name].close()
                if not panel.objects:
                    panel.objects = [
                        pn.pane.Alert("Waiting for acoustic data…", alert_type="warning")
                    ]
                notice.object = (
                    "Waiting for acoustic data…"
                    if isinstance(exc, FileNotFoundError)
                    else f"Could not load source; will retry: {exc}"
                )
                notice.alert_type = (
                    "warning" if isinstance(exc, (FileNotFoundError, OSError)) else "danger"
                )
                notice.visible = True
                return False
            finally:
                panel.loading = False

        def catalogue_revision():
            nonlocal entries
            updated_entries = {}
            if settings.get("factory"):
                options = {k: v for k, v in settings.items() if k != "factory"}
                options["root"] = _data_path(options.get("root", "."), config_dir)
                updated_entries = _factory(settings["factory"])(**options)
            # Catalogue membership and the selected source can change independently.
            entries = updated_entries
            return (tuple(entries), entry_revision())

        def update(revision):
            selected = menu.value
            options = list(dict.fromkeys(["Latest completed transect", *entries]))
            state["menu_update"] = True
            try:
                if list(menu.options) != options:
                    menu.options = options
                menu.value = selected if selected in options else options[0]
            finally:
                state["menu_update"] = False
            return load()

        def error(exc):
            notice.object = f"Could not check for new files; will retry: {exc}"
            notice.alert_type = "warning"
            notice.visible = True
            if not panel.objects:
                panel.objects = [pn.pane.Alert("Waiting for acoustic data…", alert_type="warning")]

        check = watch_changes(
            catalogue_revision,
            update,
            seconds=float(config.get("app", {}).get("refresh_seconds", 60)),
            on_error=error,
        )
        menu.param.watch(load, "value")
        refresh.on_click(lambda event: check(force=True))
        file_sidebar.objects = [menu, refresh, notice, controls]
        check()
        return panel

    def extension_page(name, options):
        # Existing recipe factory contract is retained. No selection is shared.
        source_name = config.get("app", {}).get("source", "survey")
        source_settings = config.get("data", {}).get(source_name, {})
        path = (
            _data_path(source_settings["path"], config_dir) if "path" in source_settings else None
        )
        requires_source = (options or {}).get("requires_source", True)
        owned = False
        if not requires_source:
            data = xr.Dataset()
        elif datasets and source_name in datasets:
            data = datasets[source_name]
        else:
            data = open_source(source_settings, path)
            owned = True
        try:
            settings = dict(options or {})
            settings.pop("requires_source", None)
            factory_path = settings.pop("factory", None)
            if factory_path:
                factory = _factory(factory_path)
            else:
                from .registry import EXTENSIONS

                factory = EXTENSIONS[name]()
            root = _data_path(
                settings.get("root", str(path.parent.parent) if path else "."),
                config_dir,
            )
            _, tabs = factory(
                ds=data,
                root=root,
                config=settings,
                max_pings=int(config.get("navigation", {}).get("max_pings", 500)),
                max_range_samples=int(config.get("display", {}).get("max_range_samples", 1200)),
            )
            return pn.Column(
                *[
                    content(None, (-100, -30)) if callable(content) else content
                    for _, content in tabs
                ],
                sizing_mode="stretch_width",
            )
        finally:
            if owned:
                # Lazy extension plots may still depend on the open source.
                extension_sources.append(data)

    home = pn.Column(
        "## Welcome to Echoshader",
        "Explore acoustic and environmental data using reusable visualizations.",
        "Open **Single file** to inspect an acoustic dataset, or choose one of the "
        "optional views enabled in your app configuration. Each view loads independently.",
        "### Explore the ecosystem",
        "- [Echoshader](https://github.com/echostack-org/echoshader): interactive visualization.\n"
        "- [Echopype](https://github.com/echostack-org/echopype): acoustic data processing.\n"
        "- [Echodataflow](https://github.com/echostack-org/echodataflow): workflow orchestration.\n"
        "- [Recipes](https://github.com/echostack-org/echodataflow-recipes): example configurations.",
        sizing_mode="stretch_width",
        margin=20,
    )
    pages = pn.Tabs(
        ("Home", home),
        *([("Single file", lazy_view(file_page))] if config.get("data") or datasets else []),
        *[
            (
                "Transects"
                if name == "recent_transects"
                else "SPASSO"
                if name == "spasso"
                else name.replace("_", " ").title(),
                lazy_view(lambda name=name, opts=opts: extension_page(name, opts)),
            )
            for name, opts in config.get("extensions", {}).items()
        ],
        dynamic=True,
        sizing_mode="stretch_width",
    )

    def layout(event=None):
        file_sidebar.visible = bool(config.get("data") or datasets) and pages.active == 1
        # Template sidebar width is fixed when its HTML is first rendered.
        # Hide the contents only; keep the sidebar available for Single file.

    pages.param.watch(layout, "active")
    layout()
    template.main.append(pages)
    return template
