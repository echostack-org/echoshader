from string import Template

import param


class AppContext(param.Parameterized):
    """Shared runtime values available to an Echoshader app."""

    values = param.Dict(default={})

    def update(self, **values):
        """Update runtime context values."""
        updated = dict(self.values)
        updated.update(values)
        self.values = updated

    def resolve(self, value):
        """Resolve ${name} placeholders using runtime context."""

        if not isinstance(value, str):
            return value

        return Template(value).safe_substitute(self.values)
