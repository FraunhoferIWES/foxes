import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse

from foxes.output.flow_plots_2d.get_fig import get_fig


def _show_rotor_dict():
    return {
        "color": "red",
        "D": np.array([100.0]),
        "H": np.array([90.0]),
        "X": np.array([750.0]),
        "Y": np.array([0.0]),
        "AMB_WD": np.array([270.0]),
        "turb_angle": np.array([270.0]),
    }


def test_get_fig_yz_rotor_plane_uses_y_and_h_coordinates():
    x_pos = np.array([-10.0, 10.0])
    y_pos = np.array([0.0, 20.0])
    data = np.array([[1.0, 2.0, 3.0, 4.0]])

    fig, im = get_fig(
        var="WS",
        data=data,
        si=0,
        s=0,
        x_pos=x_pos,
        y_pos=y_pos,
        ret_im=True,
        show_rotor_dict=_show_rotor_dict(),
        rotor_plane="yz",
    )

    # im contains image artist followed by rotor patch artists
    rotor_patch = im[1]
    assert isinstance(rotor_patch, Ellipse)
    assert np.allclose(rotor_patch.center, [0.0, 90.0])
    assert np.isclose(rotor_patch.height, 100.0)
    assert np.isclose(rotor_patch.width, 100.0)

    ax = fig.axes[0]
    assert ax.get_xlabel() == "y [m]"
    assert ax.get_ylabel() == "z [m]"

    plt.close(fig)


def test_get_fig_rotor_slice_filters_non_intersecting_rotors():
    x_pos = np.array([-10.0, 10.0])
    y_pos = np.array([0.0, 20.0])
    data = np.array([[1.0, 2.0, 3.0, 4.0]])

    fig, im = get_fig(
        var="WS",
        data=data,
        si=0,
        s=0,
        x_pos=x_pos,
        y_pos=y_pos,
        ret_im=True,
        show_rotor_dict=_show_rotor_dict(),
        rotor_plane="yz",
        rotor_slice={"axis": "x", "value": 700.0, "tol": 0.0},
    )

    # Only image artist remains when no rotor intersects the slice.
    assert len(im) == 1

    plt.close(fig)


def test_get_fig_yz_rotor_edge_on_is_line():
    x_pos = np.array([-10.0, 10.0])
    y_pos = np.array([0.0, 20.0])
    data = np.array([[1.0, 2.0, 3.0, 4.0]])

    rdict = _show_rotor_dict()
    rdict["turb_angle"] = np.array([180.0])

    fig, im = get_fig(
        var="WS",
        data=data,
        si=0,
        s=0,
        x_pos=x_pos,
        y_pos=y_pos,
        ret_im=True,
        show_rotor_dict=rdict,
        rotor_plane="yz",
    )

    rotor_line = im[1]
    assert isinstance(rotor_line, Line2D)
    assert np.allclose(rotor_line.get_xdata(), [0.0, 0.0])
    assert np.allclose(np.sort(rotor_line.get_ydata()), [40.0, 140.0])

    plt.close(fig)
