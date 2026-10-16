import param


class SelectionState(param.Parameterized):
    """One named reactive selection."""

    value = param.Parameter(
        default=None,
        allow_None=True,
    )

    source = param.String(default="")
    kind = param.String(default="")
    streams = param.List(default=[])

    def set(
        self,
        value,
        source: str = "",
        kind: str = "",
    ):
        """Update the selection."""

        self.source = source
        self.kind = kind

        # Set this last so observers react after metadata changes.
        self.value = value

    def clear(self):
        """Clear the selection."""

        self.source = ""
        self.kind = ""
        self.value = None

    def add_stream(
        self,
        stream,
    ):
        """Keep a HoloViews stream alive."""

        streams = list(self.streams)

        streams.append(stream)

        self.streams = streams


class ValueState(param.Parameterized):
    """One named reactive application value."""

    value = param.Parameter(
        default=None,
        allow_None=True,
    )


class AppState(param.Parameterized):
    """Shared reactive state for an Echoshader application."""

    datasets = param.Dict(default={})
    selections = param.Dict(default={})
    values = param.Dict(default={})

    def set_dataset(
        self,
        name,
        dataset,
    ):
        datasets = dict(self.datasets)

        datasets[name] = dataset

        self.datasets = datasets

    def get_selection(
        self,
        name: str,
    ) -> SelectionState:
        """Return a named selection, creating it when necessary."""

        if name not in self.selections:
            selections = dict(self.selections)

            selections[name] = SelectionState(
                name=name,
            )

            self.selections = selections

        return self.selections[name]

    def get_value(
        self,
        name: str,
        default=None,
    ) -> ValueState:
        """Return a named reactive value."""

        if name not in self.values:
            values = dict(self.values)

            values[name] = ValueState(
                name=name,
                value=default,
            )

            self.values = values

        return self.values[name]

    def set_value(
        self,
        name: str,
        value,
    ):
        """Set a named reactive value."""

        self.get_value(name).value = value
