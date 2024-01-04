import plotly.io as pio
import io
from PIL import Image


def show_plotly_fig(fig):
    """
    Displays a plotly figure in a window

    Reference:
    https://stackoverflow.com/questions/53570384/plotly-how-to-make-a-standalone-plot-in-a-window

    :group: utils

    """
    buf = io.BytesIO()
    pio.write_image(fig, buf)
    img = Image.open(buf)
    img.show()
