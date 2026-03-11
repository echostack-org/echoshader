"""
Region Browser - Interactive annotation and editing tool for ocean sonar regions.

Allows users to browse, edit, save, and export regions on echogram data.
"""

import panel as pn
import holoviews as hv
from holoviews.streams import PolyDraw, PolyEdit
import numpy as np
import pandas as pd
import warnings
import logging

warnings.filterwarnings('ignore')
logging.getLogger('root').setLevel(logging.ERROR)


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
            dt = pd.to_datetime(x, unit='ms')
            return dt.strftime('%H:%M:%S')
        except Exception:
            return f'{x:.0f}'

    def parse_polygons_from_df(df):
        """Parse polygon data from DataFrame"""
        results = []
        for _, row in df.iterrows():
            try:
                times = row['time']
                depths = row['depth']
                time_ms = times.astype('datetime64[ms]').astype(np.int64)
                results.append({
                    'ping_time': time_ms.tolist(),
                    'depth': depths.tolist(),
                    'region_id': row['region_id']
                })
            except Exception:
                pass
        return results

    def validate_loaded_regions(loaded_df, echogram_data):
        """Validate that loaded regions match the current echogram"""
        try:
            echogram_start = pd.Timestamp(
                echogram_data.ping_time.min().values
            )
            echogram_end = pd.Timestamp(echogram_data.ping_time.max().values)

            for idx, row in loaded_df.iterrows():
                region_id = row['region_id']
                times = row['time']

                region_start = pd.Timestamp(times.min())
                region_end = pd.Timestamp(times.max())

                if region_start < echogram_start or region_end > echogram_end:
                    return (
                        False,
                        f"Region {region_id} times "
                        f"({region_start} to {region_end}) "
                        f"outside echogram range "
                        f"({echogram_start} to {echogram_end})"
                    )

            return True, "Valid"

        except Exception as e:
            return False, f"Validation error: {e}"

    original_df = regions_df.copy()
    sample_df = regions_df.copy()

    poly_draw_stream = None
    poly_edit_stream = None

    time_dim = hv.Dimension('ping_time', label='Time', value_format=format_time)

    background_cache = {}

    if cache_backgrounds:
        print("⏳ Pre-caching backgrounds...")

        for region_id in sample_df['region_id']:
            current_df = sample_df[sample_df['region_id'] == region_id]
            parsed = parse_polygons_from_df(current_df)

            if parsed:
                try:
                    time_values = np.array(parsed[0]['ping_time'])
                    start_t = pd.to_datetime(time_values.min(), unit='ms')
                    end_t = pd.to_datetime(time_values.max(), unit='ms')
                    buffer = pd.Timedelta(minutes=5)
                    ds_slice = ds.sel(
                        ping_time=slice(start_t - buffer, end_t + buffer)
                    )

                    if len(ds_slice.ping_time) > 0:
                        ds_slice = ds_slice.assign_coords({
                            'ping_time': (
                                ('ping_time',),
                                ds_slice['ping_time'].data.astype('int64') // 10**6
                            )
                        })
                        if 'depth' in ds_slice.dims:
                            ds_slice = ds_slice.rename({'depth': 'echo_range'})

                        background = ds_slice.eshader.echogram(
                            ds_slice.channel.values.tolist()
                        )()
                        background_cache[region_id] = background
                        print(f"  ✓ Region {region_id}")
                except Exception:
                    pass

        print(f"✅ Cached {len(background_cache)} backgrounds")

    region_ids = list(sample_df['region_id'])

    mode_selector = pn.widgets.RadioButtonGroup(
        name='Mode',
        options=['Browse', 'Edit'],
        value='Browse',
        button_type='primary',
        width=200
    )

    region_dropdown = pn.widgets.Select(
        name='Select Region',
        options=region_ids,
        value=region_ids[0],
        width=200
    )

    prev_btn = pn.widgets.Button(
        name='◀ Previous',
        button_type='light',
        width=95
    )

    next_btn = pn.widgets.Button(
        name='Next ▶',
        button_type='light',
        width=95
    )

    save_btn = pn.widgets.Button(
        name='💾 Save Changes',
        button_type='success',
        width=200
    )

    reset_btn = pn.widgets.Button(
        name='🔄 Reset Edits',
        button_type='warning',
        width=200
    )

    export_btn = pn.widgets.Button(
        name='📥 Export CSV',
        button_type='primary',
        width=200
    )

    load_btn = pn.widgets.FileInput(
        name='📂 Load CSV',
        accept='.csv',
        width=200
    )

    status = pn.pane.Markdown(
        "🟢 **Browse Mode** - View Only",
        styles={
            'background': '#e8f5e9',
            'padding': '10px 15px',
            'border-radius': '20px',
            'border-left': '4px solid #4caf50',
            'margin': '10px 0'
        }
    )

    actions_section = pn.Column(
        pn.pane.Markdown(
            "### Actions",
            styles={
                'font-size': '14px',
                'color': '#666',
                'margin-bottom': '5px'
            }
        ),
        load_btn,
        pn.Spacer(height=5),
        save_btn,
        pn.Spacer(height=5),
        reset_btn,
        pn.Spacer(height=5),
        export_btn,
        visible=False
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
        is_edit = event.new == 'Edit'

        actions_section.visible = is_edit

        if is_edit:
            status.object = (
                "🔴 **Edit Mode** - Click to add | "
                "Drag to move | Double-click to delete"
            )
            status.styles = {
                'background': '#ffebee',
                'padding': '10px 15px',
                'border-radius': '20px',
                'border-left': '4px solid #f44336',
                'margin': '10px 0'
            }
        else:
            status.object = "🟢 **Browse Mode** - View Only"
            status.styles = {
                'background': '#e8f5e9',
                'padding': '10px 15px',
                'border-radius': '20px',
                'border-left': '4px solid #4caf50',
                'margin': '10px 0'
            }

    def save_changes(event):
        """Save edited polygon to DataFrame"""
        nonlocal poly_draw_stream, sample_df

        if poly_draw_stream is None:
            status.object = "⚠️ No edits to save"
            return

        try:
            data = poly_draw_stream.data
            if not data or len(data.get('xs', [])) == 0:
                return

            xs = data['xs'][0]
            ys = data['ys'][0]
            times_dt = pd.to_datetime(xs, unit='ms').values
            depths_arr = np.array(ys, dtype=np.float64)

            selected_id = region_dropdown.value
            idx = sample_df[sample_df['region_id'] == selected_id].index[0]
            sample_df.at[idx, 'time'] = times_dt
            sample_df.at[idx, 'depth'] = depths_arr

            status.object = (
                f"✅ **Saved!** Region {selected_id} "
                f"({len(times_dt)} vertices)"
            )
        except Exception as e:
            status.object = f"❌ Error: {e}"

    def export_csv(event):
        """Export edited regions to CSV"""
        nonlocal sample_df

        try:
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"edited_regions_{timestamp}.csv"

            export_data = []

            for _, row in sample_df.iterrows():
                region_id = row['region_id']
                times = row['time']
                depths = row['depth']

                time_str = "[" + ", ".join([str(t) for t in times]) + "]"
                depth_str = "[" + ", ".join([str(d) for d in depths]) + "]"

                export_data.append({
                    'region_id': region_id,
                    'time': time_str,
                    'depth': depth_str
                })

            export_df = pd.DataFrame(export_data)
            export_df.to_csv(output_filename, index=False)

            status.object = f"✅ **Exported!** Saved as: {output_filename}"
            print(f"✅ CSV exported to: {output_filename}")

        except Exception as e:
            status.object = f"❌ Export failed: {e}"
            print(f"❌ Export error: {e}")

    def reset_edits(event):
        """Reset to original polygon"""
        nonlocal sample_df, original_df

        try:
            selected_id = region_dropdown.value
            idx = sample_df[sample_df['region_id'] == selected_id].index[0]

            sample_df.at[idx, 'time'] = original_df.at[idx, 'time'].copy()
            sample_df.at[idx, 'depth'] = original_df.at[idx, 'depth'].copy()

            status.object = (
                f"🔄 **Reset** Region {selected_id} - "
                f"Navigate away and back to see original"
            )
            print(f"✅ Reset Region {selected_id} to original")

        except Exception as e:
            status.object = f"❌ Reset error: {e}"
            print(f"❌ Reset error: {e}")

    def load_csv_file(event):
        """Load regions from uploaded CSV"""
        nonlocal sample_df, region_dropdown

        if load_btn.value is None:
            status.object = "⚠️ No file selected"
            return

        try:
            import io
            csv_data = io.BytesIO(load_btn.value)
            loaded_df = pd.read_csv(csv_data)

            required_columns = ['region_id', 'time', 'depth']
            if not all(col in loaded_df.columns for col in required_columns):
                status.object = (
                    f"❌ Invalid CSV: Missing required columns. "
                    f"Need: {required_columns}"
                )
                return

            for idx, row in loaded_df.iterrows():
                try:
                    time_str = row['time'].strip('[]')
                    time_list = [t.strip() for t in time_str.split(',')]
                    loaded_df.at[idx, 'time'] = np.array(
                        time_list,
                        dtype='datetime64[ns]'
                    )

                    depth_str = row['depth'].strip('[]')
                    depth_list = [
                        float(d.strip()) for d in depth_str.split(',')
                    ]
                    loaded_df.at[idx, 'depth'] = np.array(
                        depth_list,
                        dtype=np.float64
                    )

                except Exception as e:
                    status.object = f"❌ Parse error in row {idx}: {e}"
                    return

            is_valid, message = validate_loaded_regions(loaded_df, ds)
            if not is_valid:
                status.object = f"❌ Validation failed: {message}"
                return

            sample_df = loaded_df.copy()
            new_region_ids = list(sample_df['region_id'])
            region_dropdown.options = new_region_ids
            region_dropdown.value = new_region_ids[0]

            status.object = f"✅ **Loaded** {len(loaded_df)} regions from CSV"
            print(f"✅ Loaded {len(loaded_df)} regions successfully")

        except Exception as e:
            status.object = f"❌ Load error: {e}"
            print(f"❌ Load error: {e}")

    @pn.depends(region_dropdown.param.value, mode_selector.param.value)
    def get_region_view(selected_id, mode):
        """Generate region view with cached backgrounds"""
        nonlocal poly_draw_stream, poly_edit_stream

        current_df = sample_df[sample_df['region_id'] == selected_id]
        if current_df.empty:
            return pn.pane.Markdown("⚠️ No data")

        parsed = parse_polygons_from_df(current_df)
        if not parsed or selected_id not in background_cache:
            return pn.pane.Markdown("⚠️ No background")

        background = background_cache[selected_id]

        poly = hv.Polygons(
            [{
                'ping_time': r['ping_time'],
                'echo_range': r['depth'],
                'region_id': r['region_id']
            } for r in parsed],
            kdims=[time_dim, hv.Dimension('echo_range', label='Depth (m)')],
            vdims=['region_id']
        ).opts(color='red', fill_alpha=0.3, line_width=2)

        if mode == 'Edit':
            poly_draw_stream = PolyDraw(
                source=poly,
                drag=True,
                num_objects=1,
                show_vertices=True,
                vertex_style={'size': 10, 'color': 'red', 'fill_alpha': 0.8}
            )

            poly_edit_stream = PolyEdit(
                source=poly,
                vertex_style={
                    'size': 10,
                    'color': 'orange',
                    'fill_alpha': 0.8
                },
                shared=True
            )

        return background * poly

    prev_btn.on_click(update_nav)
    next_btn.on_click(update_nav)
    mode_selector.param.watch(on_mode_change, 'value')
    save_btn.on_click(save_changes)
    export_btn.on_click(export_csv)
    reset_btn.on_click(reset_edits)
    load_btn.param.watch(load_csv_file, 'value')

    sidebar = pn.Column(
        pn.pane.Markdown(
            "## 🎛️ Controls",
            styles={'color': '#1976d2', 'margin-bottom': '15px'}
        ),

        pn.pane.Markdown(
            "### Mode",
            styles={
                'font-size': '14px',
                'color': '#666',
                'margin-bottom': '5px'
            }
        ),
        mode_selector,
        pn.Spacer(height=20),

        pn.pane.Markdown(
            "### Regions",
            styles={
                'font-size': '14px',
                'color': '#666',
                'margin-bottom': '5px'
            }
        ),
        region_dropdown,
        pn.Spacer(height=5),
        pn.Row(prev_btn, next_btn),
        pn.Spacer(height=20),

        actions_section,

        styles={
            'background': '#f5f5f5',
            'padding': '20px',
            'border-radius': '8px',
            'box-shadow': '2px 0 5px rgba(0,0,0,0.1)'
        },
        width=250
    )

    main_content = pn.Column(
        pn.pane.Markdown(
            "# 🐟 Hake School Region Browser",
            styles={'color': '#1976d2', 'margin-bottom': '5px'}
        ),
        pn.pane.Markdown(
            "*Interactive region browser with Browse/Edit modes*",
            styles={
                'color': '#666',
                'font-size': '14px',
                'margin-bottom': '15px'
            }
        ),
        status,
        get_region_view,
        styles={'padding': '20px'},
        sizing_mode='stretch_width'
    )

    layout = pn.Row(
        sidebar,
        main_content,
        styles={'background': 'white'},
        sizing_mode='stretch_width'
    )

    return layout