"""Small session-local polling utilities; unchanged revisions do no work."""

import panel as pn


def file_revision(path):
    """Cheap revision for ordinary files (including growing navigation chunks)."""
    from pathlib import Path

    path = Path(path)
    try:
        stat = path.stat()
    except FileNotFoundError:
        return None
    return (str(path.resolve()), stat.st_mtime_ns, stat.st_size)


def watch_changes(revision, update, *, seconds=60, on_error=None):
    """Return a check function and optionally poll it until session destruction.

    update receives the new revision. Return False to retry an incomplete source
    on the next poll. Exceptions are reported through on_error, never committed.
    The caller triggers the initial check so view construction stays lazy.
    """
    previous = object()
    busy = False

    def check(event=None, *, force=False):
        nonlocal previous, busy
        if busy:
            return False
        busy = True
        try:
            current = revision()
            if not force and current == previous:
                return False
            if update(current) is False:
                return False
            previous = current
            return True
        except Exception as exc:
            if on_error is None:
                raise
            on_error(exc)
            return False
        finally:
            busy = False

    if seconds > 0:
        callback = pn.state.add_periodic_callback(check, period=max(1, int(seconds * 1000)))
        pn.state.on_session_destroyed(lambda context: callback.stop())
    return check
