"""Simple read-only viewer for the CPS processing database."""

import sqlite3
from pathlib import Path

import pandas as pd
import panel as pn


def build_status(*, ds=None, root, config, max_pings=500, max_range_samples=1200):
    database = Path(root) / config.get("processing_db", "processing.db")

    refresh = pn.widgets.Button(name="Refresh database", width=160)
    table_select = pn.widgets.Select(name="Table", options=[])
    notice = pn.pane.Alert("", alert_type="warning", visible=False)
    table_view = pn.widgets.Tabulator(
        pd.DataFrame(),
        pagination="local",
        page_size=25,
        height=500,
        sizing_mode="stretch_width",
    )

    def load_tables(event=None):
        if not database.is_file():
            table_select.options = []
            table_view.value = pd.DataFrame()
            notice.object = f"Database not found: {database}"
            notice.visible = True
            return

        try:
            with sqlite3.connect(database) as connection:
                names = pd.read_sql_query(
                    "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name",
                    connection,
                )["name"].tolist()

            table_select.options = names
            if not names:
                table_view.value = pd.DataFrame()
                notice.object = "The database has no tables yet."
                notice.visible = True
            else:
                notice.visible = False
                if table_select.value not in names:
                    table_select.value = names[0]
                load_table()

        except (sqlite3.Error, pd.errors.DatabaseError) as exc:
            notice.object = f"Could not read the database: {exc}"
            notice.visible = True

    def load_table(event=None):
        name = table_select.value
        if not name or not database.is_file():
            table_view.value = pd.DataFrame()
            return

        # Quote the table name so SQLite treats it as an identifier.
        quoted_name = '"' + name.replace('"', '""') + '"'
        try:
            with sqlite3.connect(database) as connection:
                table_view.value = pd.read_sql_query(
                    f"SELECT * FROM {quoted_name} LIMIT 5000",
                    connection,
                )
            notice.visible = False
        except (sqlite3.Error, pd.errors.DatabaseError) as exc:
            notice.object = f"Could not read table {name}: {exc}"
            notice.visible = True

    table_select.param.watch(load_table, "value")
    refresh.on_click(load_tables)
    load_tables()

    content = pn.Column(
        pn.Row(refresh, table_select),
        notice,
        table_view,
        sizing_mode="stretch_width",
    )
    return None, [("Database", content)]
