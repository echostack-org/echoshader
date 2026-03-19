"""
Region Browser - Interactive annotation and editing tool for ocean sonar regions.

Allows users to browse, edit, save, and export regions on echogram data.
"""

import logging
import warnings

import holoviews as hv
import numpy as np
import pandas as pd
import panel as pn
from holoviews.streams import PolyDraw, PolyEdit

warnings.filterwarnings("ignore")
logging.getLogger("root").setLevel(logging.ERROR)


def region_browser(ds, regions_df, cache_backgrounds=True):
    """
    Create an interactive region browser for echogram data.

    Parameters
    ----------
    ds : xarray.Dataset
        MVBS dataset containing echogram data with required variables:
        - Sv: backscatter data
        - ping_time: time dimension
        - echo_range or depth: vertical dimension
    regions_df : pandas.DataFrame
        DataFrame with regions in Echoregions format
        Required columns: region_id, time, depth
    cache_backgrounds : bool, optional
        Pre-cache echogram backgrounds for faster navigation (default: True)

    Returns
    -------
    panel.Row
        Interactive region browser panel with Browse/Edit modes

    Examples
    --------
    >>> import echoshader
    >>> import xarray as xr
    >>> import pandas as pd
    >>>
    >>> # Load your data
    >>> ds = xr.open_zarr('path/to/data.zarr')
    >>> regions = pd.DataFrame({
    ...     'region_id': [1, 2, 3],
    ...     'time': [...],
    ...     'depth': [...]
    ... })
    >>>
    >>> # Create browser
    >>> browser = echoshader.region_browser(ds=ds, regions_df=regions)
    >>> browser.show()
    """

    pn.extension()

    def format_time(x):
        """Format milliseconds to readable time"""
        try:
            dt = pd.to_datetime(x, unit="ms")
            return dt.strftime("%H:%M:%S")
        except Exception:
            return f"{x:.0f}"

    def parse_polygons_from_df(df):
        """Parse polygon data from DataFrame"""
        results = []
        for _, row in df.iterrows():
            try:
                times = row["time"]
                depths = row["depth"]
                time_ms = times.astype("datetime64[ms]").astype(np.int64)
                results.append(
                    {
                        "ping_time": time_ms.tolist(),
                        "depth": depths.tolist(),
                        "region_id": row["region_id"],
                    }
                )
            except Exception:
                pass
        return results

    def validate_loaded_regions(loaded_df, echogram_data):
        """Validate that loaded regions match the current echogram"""
        try:
            echogram_start = pd.Timestamp(echogram_data.ping_time.min().values)
            echogram_end = pd.Timestamp(echogram_data.ping_time.max().values)

            for idx, row in loaded_df.iterrows():
                region_id = row["region_id"]
                times = row["time"]

                region_start = pd.Timestamp(times.min())
                region_end = pd.Timestamp(times.max())

                if region_start < echogram_start or region_end > echogram_end:
                    return (
                        False,
                        f"Validation error: Region {region_id} times "
                        f"({region_start} to {region_end}) fall outside the valid "
                        f"echogram range ({echogram_start} to {echogram_end}).",
                    )

            return True, "Valid"

        except Exception as e:
            return False, f"Validation processing error: {e}"

    # Store baseline for reset functionality
    baseline_df = regions_df.copy()

    sample_df = regions_df.copy()

    poly_draw_stream = None
    poly_edit_stream = None

    time_dim = hv.Dimension("ping_time", label="Time", value_format=format_time)

    background_cache = {}

    if cache_backgrounds:
        print("Pre-caching backgrounds...")

        for region_id in sample_df["region_id"]:
            current_df = sample_df[sample_df["region_id"] == region_id]
            parsed = parse_polygons_from_df(current_df)

            if parsed:
                try:
                    time_values = np.array(parsed[0]["ping_time"])
                    start_t = pd.to_datetime(time_values.min(), unit="ms")
                    end_t = pd.to_datetime(time_values.max(), unit="ms")
                    buffer = pd.Timedelta(minutes=5)
                    ds_slice = ds.sel(ping_time=slice(start_t - buffer, end_t + buffer))

                    if len(ds_slice.ping_time) > 0:
                        ds_slice = ds_slice.assign_coords(
                            {
                                "ping_time": (
                                    ("ping_time",),
                                    ds_slice["ping_time"].data.astype("int64") // 10**6,
                                )
                            }
                        )
                        if "depth" in ds_slice.dims:
                            ds_slice = ds_slice.rename({"depth": "echo_range"})

                        background = ds_slice.eshader.echogram(
                            ds_slice.channel.values.tolist()
                        )()
                        background_cache[region_id] = background
                        print(f"  Processed Region {region_id}")
                except Exception:
                    pass

        print(f"Cached {len(background_cache)} backgrounds successfully.")

    region_ids = list(sample_df["region_id"])

    mode_selector = pn.widgets.RadioButtonGroup(
        name="Mode",
        options=["Browse", "Edit"],
        value="Browse",
        button_type="primary",
        width=200,
    )

    region_dropdown = pn.widgets.Select(
        name="Select Region", options=region_ids, value=region_ids[0], width=200
    )

    prev_btn = pn.widgets.Button(name="Previous", button_type="light", width=95)
    next_btn = pn.widgets.Button(name="Next", button_type="light", width=95)

    reset_btn = pn.widgets.Button(
        name="Reset Region", button_type="warning", width=200
    )

    apply_btn = pn.widgets.Button(
        name="Apply Edit", button_type="success", width=200
    )

    export_btn = pn.widgets.Button(
        name="Export to CSV", button_type="primary", width=200
    )

    load_btn = pn.widgets.FileInput(name="Load CSV", accept=".csv", width=200)

    status = pn.pane.Markdown(
        "**Browse Mode** - View Only",
        styles={
            "background": "#e3f2fd",
            "padding": "10px 15px",
            "border-radius": "8px",
            "border-left": "4px solid #2196f3",
            "margin": "10px 0",
        },
    )

    actions_section = pn.Column(
        pn.pane.Markdown(
            "### Actions",
            styles={"font-size": "14px", "color": "#666", "margin-bottom": "5px"},
        ),
        pn.pane.Markdown(
            """
            **Editing Workflow:**
            1. Edit polygon vertices
            2. Click "Apply Edit"
            3. Edit other regions as needed
            4. Click "Export to CSV" to save
            
            ⚠️ *Changes are not saved to disk until exported!*
            """,
            styles={
                "font-size": "11px",
                "color": "#666",
                "background": "#fff3cd",
                "padding": "8px",
                "border-radius": "5px",
                "margin-bottom": "10px",
                "border-left": "3px solid #ffc107"
            }
        ),
        load_btn,
        pn.Spacer(height=5),
        reset_btn,
        pn.Spacer(height=5),
        apply_btn,
        pn.Spacer(height=5),
        export_btn,
        visible=False,
    )

    def update_nav(event):
        """Navigate between regions"""
        idx = region_ids.index(region_dropdown.value)
        if event.obj == next_btn:
            region_dropdown.value = region_ids[(idx + 1) % len(region_ids)]
        else:
            region_dropdown.value = region_ids[(idx - 1) % len(region_ids)]

    def on_mode_change(event):
        """Toggle between Browse and Edit modes"""
        is_edit = event.new == "Edit"
        actions_section.visible = is_edit

        if is_edit:
            status.object = (
                "**Edit Mode** - "
                "**PolyDraw:** Click to add vertices, double-click last "
                "vertex to finish. Drag to move entire polygon. "
                "**PolyEdit:** Drag vertices to reposition, double-click "
                "a vertex to delete it. "
                "**Reset Region** restores the original."
            )
            status.styles = {
                "background": "#e3f2fd",
                "padding": "10px 15px",
                "border-radius": "8px",
                "border-left": "4px solid #2196f3",
                "margin": "10px 0",
            }
        else:
            status.object = "**Browse Mode** - View Only"
            status.styles = {
                "background": "#e3f2fd",
                "padding": "10px 15px",
                "border-radius": "8px",
                "border-left": "4px solid #2196f3",
                "margin": "10px 0",
            }

    def apply_edits(event):
        """Apply edited polygon to DataFrame in memory"""
        if poly_draw_stream is None:
            status.object = "No edits to apply. Please ensure a region is selected."
            return

        try:
            data = poly_draw_stream.data
            if not data or len(data.get("xs", [])) == 0:
                return

            xs = data["xs"][0]
            ys = data["ys"][0]
            times_dt = pd.to_datetime(xs, unit="ms").values
            depths_arr = np.array(ys, dtype=np.float64)

            selected_id = region_dropdown.value
            idx = sample_df[sample_df["region_id"] == selected_id].index[0]
            sample_df.at[idx, "time"] = times_dt
            sample_df.at[idx, "depth"] = depths_arr

            status.object = (
                f"**Applied!** Region {selected_id} updated. "
                "Export to CSV to save to disk."
            )
            status.styles = {
                "background": "#e8f5e9",
                "padding": "10px 15px",
                "border-radius": "8px",
                "border-left": "4px solid #4caf50",
                "margin": "10px 0",
            }
        except Exception as e:
            status.object = (
                f"Failed to apply edits to Region {selected_id}. "
                f"Please check polygon format: {e}"
            )
            status.styles = {
                "background": "#ffebee",
                "padding": "10px 15px",
                "border-radius": "8px",
                "border-left": "4px solid #f44336",
                "margin": "10px 0",
            }

    def reset_region(event):
        """Reset current region to baseline version"""
        try:
            selected_id = region_dropdown.value
            
            # Find baseline version
            baseline_row = baseline_df[baseline_df["region_id"] == selected_id]
            
            if baseline_row.empty:
                status.object = f"No baseline found for Region {selected_id}."
                status.styles = {
                    "background": "#ffebee",
                    "padding": "10px 15px",
                    "border-radius": "8px",
                    "border-left": "4px solid #f44336",
                    "margin": "10px 0",
                }
                return
            
            # Reset to baseline
            idx = sample_df[sample_df["region_id"] == selected_id].index[0]
            sample_df.at[idx, "time"] = baseline_row.iloc[0]["time"].copy()
            sample_df.at[idx, "depth"] = baseline_row.iloc[0]["depth"].copy()
            
            # Force UI refresh
            region_dropdown.param.trigger('value')
            
            status.object = f"**Reset!** Region {selected_id} restored to original."
            status.styles = {
                "background": "#e8f5e9",
                "padding": "10px 15px",
                "border-radius": "8px",
                "border-left": "4px solid #4caf50",
                "margin": "10px 0",
            }
            
            print(f"Reset Region {selected_id} to baseline")
            
        except Exception as e:
            status.object = f"Reset failed: {e}"
            status.styles = {
                "background": "#ffebee",
                "padding": "10px 15px",
                "border-radius": "8px",
                "border-left": "4px solid #f44336",
                "margin": "10px 0",
            }
            print(f"Reset error: {e}")

    def export_csv(event):
        """Export edited regions to CSV"""
        try:
            from datetime import datetime

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"edited_regions_{timestamp}.csv"

            export_data = []

            for _, row in sample_df.iterrows():
                region_id = row["region_id"]
                times = row["time"]
                depths = row["depth"]

                # Add single quotes around timestamps to avoid leading zero errors
                time_str = "[" + ", ".join([f"'{str(t)}'" for t in times]) + "]"
                depth_str = "[" + ", ".join([str(d) for d in depths]) + "]"

                export_data.append(
                    {"region_id": region_id, "time": time_str, "depth": depth_str}
                )

            export_df = pd.DataFrame(export_data)
            export_df.to_csv(output_filename, index=False)

            status.object = (
                f"**Exported Successfully!** All {len(sample_df)} regions "
                f"saved to: {output_filename}"
            )
            print(f"CSV exported to: {output_filename}")

        except Exception as e:
            status.object = (
                f"Export failed. Please ensure you have write permissions "
                f"in this directory: {e}"
            )
            print(f"Export error: {e}")

    def load_csv_file(event):
        """Load regions from uploaded CSV"""
        nonlocal sample_df, baseline_df

        if load_btn.value is None:
            status.object = "No file selected. Please choose a CSV file to load."
            return

        try:
            import io
            import ast

            csv_data = io.BytesIO(load_btn.value)
            loaded_df = pd.read_csv(csv_data)

            required_columns = ["region_id", "time", "depth"]
            if not all(col in loaded_df.columns for col in required_columns):
                status.object = (
                    f"Invalid CSV format. Missing required columns. "
                    f"Required: {required_columns}"
                )
                return

            parsed_times = []
            parsed_depths = []

            for idx, row in loaded_df.iterrows():
                try:
                    # Try ast.literal_eval first (works for CSVs with quotes)
                    time_list = ast.literal_eval(str(row["time"]))
                    depth_list = ast.literal_eval(str(row["depth"]))

                    parsed_times.append(np.array(time_list, dtype="datetime64[ns]"))
                    parsed_depths.append(np.array(depth_list, dtype=np.float64))

                except Exception:
                    # FALLBACK: For old CSVs without quotes, parse manually
                    try:
                        time_str = str(row["time"]).strip("[]")
                        time_list = [t.strip(" '\"") for t in time_str.split(",")]
                        parsed_times.append(np.array(time_list, dtype="datetime64[ns]"))

                        depth_str = str(row["depth"]).strip("[]")
                        depth_list = [float(d.strip()) for d in depth_str.split(",")]
                        parsed_depths.append(np.array(depth_list, dtype=np.float64))
                    except Exception as fallback_e:
                        status.object = (
                            f"Parse error in row {idx}. Please check data format "
                            f"matches Echoregions standard: {fallback_e}"
                        )
                        return

            # Apply parsed arrays all at once
            loaded_df["time"] = parsed_times
            loaded_df["depth"] = parsed_depths

            is_valid, message = validate_loaded_regions(loaded_df, ds)
            if not is_valid:
                status.object = f"**Validation Failed:** {message}"
                status.styles = {
                    "background": "#ffebee",
                    "padding": "10px 15px",
                    "border-radius": "8px",
                    "border-left": "4px solid #f44336",
                    "margin": "10px 0",
                }
                return

            sample_df = loaded_df.copy()
            
            # Update baseline to loaded CSV (new reset point)
            baseline_df = loaded_df.copy()
            
            new_region_ids = list(sample_df["region_id"])
            region_dropdown.options = new_region_ids
            region_dropdown.value = new_region_ids[0]

            status.object = (
                f"**Loaded Successfully!** {len(loaded_df)} regions imported from CSV."
            )
            status.styles = {
                "background": "#e8f5e9",
                "padding": "10px 15px",
                "border-radius": "8px",
                "border-left": "4px solid #4caf50",
                "margin": "10px 0",
            }
            print(f"Loaded {len(loaded_df)} regions successfully.")

        except Exception as e:
            status.object = f"Failed to load file: {e}"
            print(f"Load error: {e}")

    @pn.depends(region_dropdown.param.value, mode_selector.param.value)
    def get_region_view(selected_id, mode):
        """Generate region view with cached backgrounds"""
        nonlocal poly_draw_stream, poly_edit_stream

        current_df = sample_df[sample_df["region_id"] == selected_id]
        if current_df.empty:
            return pn.pane.Markdown("No data available for this region.")

        parsed = parse_polygons_from_df(current_df)
        if not parsed:
            return pn.pane.Markdown("No polygon data for this region.")

        # Check if background is cached, if not, generate it on-demand
        if selected_id not in background_cache:
            try:
                time_values = np.array(parsed[0]["ping_time"])
                start_t = pd.to_datetime(time_values.min(), unit="ms")
                end_t = pd.to_datetime(time_values.max(), unit="ms")
                buffer = pd.Timedelta(minutes=5)
                ds_slice = ds.sel(ping_time=slice(start_t - buffer, end_t + buffer))

                if len(ds_slice.ping_time) > 0:
                    ds_slice = ds_slice.assign_coords(
                        {
                            "ping_time": (
                                ("ping_time",),
                                ds_slice["ping_time"].data.astype("int64") // 10**6,
                            )
                        }
                    )
                    if "depth" in ds_slice.dims:
                        ds_slice = ds_slice.rename({"depth": "echo_range"})

                    background = ds_slice.eshader.echogram(
                        ds_slice.channel.values.tolist()
                    )()
                    background_cache[selected_id] = background
                    print(f"Cached background for Region {selected_id}")
                else:
                    return pn.pane.Markdown("No echogram data in this time range.")
            except Exception as e:
                return pn.pane.Markdown(f"Error generating background: {e}")

        # Retrieve cached background (fast)
        background = background_cache[selected_id]

        # Generate polygon (only happens on first view of this region+mode)
        poly = hv.Polygons(
            [
                {
                    "ping_time": r["ping_time"],
                    "echo_range": r["depth"],
                    "region_id": r["region_id"],
                }
                for r in parsed
            ],
            kdims=[time_dim, hv.Dimension("echo_range", label="Depth (m)")],
            vdims=["region_id"],
        ).opts(color="red", fill_alpha=0.3, line_width=2)

        if mode == "Edit":
            # Attach streams - they automatically add tools when source=poly is set
            poly_draw_stream = PolyDraw(
                source=poly,
                drag=True,
                num_objects=1,
                show_vertices=True,
                vertex_style={"size": 10, "color": "red", "fill_alpha": 0.8},
            )

            poly_edit_stream = PolyEdit(
                source=poly,
                vertex_style={"size": 10, "color": "orange", "fill_alpha": 0.8},
                shared=True,
            )

        # Create composed plot (streams auto-attach in Edit mode)
        plot = background * poly
        return plot

    prev_btn.on_click(update_nav)
    next_btn.on_click(update_nav)
    mode_selector.param.watch(on_mode_change, "value")
    reset_btn.on_click(reset_region)
    apply_btn.on_click(apply_edits)
    export_btn.on_click(export_csv)
    load_btn.param.watch(load_csv_file, "value")

    sidebar = pn.Column(
        pn.pane.Markdown(
            "## Controls", styles={"color": "#1976d2", "margin-bottom": "15px"}
        ),
        pn.pane.Markdown(
            "### Mode",
            styles={"font-size": "14px", "color": "#666", "margin-bottom": "5px"},
        ),
        mode_selector,
        pn.Spacer(height=20),
        pn.pane.Markdown(
            "### Regions",
            styles={"font-size": "14px", "color": "#666", "margin-bottom": "5px"},
        ),
        region_dropdown,
        pn.Spacer(height=5),
        pn.Row(prev_btn, next_btn),
        pn.Spacer(height=20),
        actions_section,
        styles={
            "background": "#f5f5f5",
            "padding": "20px",
            "border-radius": "8px",
            "box-shadow": "2px 0 5px rgba(0,0,0,0.1)",
        },
        width=250,
    )

    main_content = pn.Column(
        pn.pane.Markdown(
            "# Hake School Region Browser",
            styles={"color": "#1976d2", "margin-bottom": "5px"},
        ),
        pn.pane.Markdown(
            "*Interactive region browser with Browse/Edit modes*",
            styles={"color": "#666", "font-size": "14px", "margin-bottom": "15px"},
        ),
        status,
        get_region_view,
        styles={"padding": "20px"},
        sizing_mode="stretch_width",
    )

    layout = pn.Row(
        sidebar,
        main_content,
        styles={"background": "white"},
        sizing_mode="stretch_width",
    )

    return layout
