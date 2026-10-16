from echoshader.elements.curtain import curtain
from echoshader.elements.echogram import (
    echogram,
    tricolor_echogram,
)
from echoshader.elements.hist import hist
from echoshader.elements.table import table
from echoshader.elements.track import track

ELEMENTS = {
    "echogram": echogram,
    "tricolor_echogram": tricolor_echogram,
    "track": track,
    "hist": hist,
    "table": table,
    "curtain": curtain,
}
