import holoviews
import pandas

from echoshader.elements.curve import curve


def test_curve():
    data = pandas.DataFrame(
        {
            "x": [0, 1, 2],
            "y": [1, 2, 3],
        }
    )

    plot = curve(data, x="x", y="y")

    assert isinstance(plot, holoviews.Curve)
