"""Ambient states reduced to spatially varying histogram bins.

The package provides binned states on either a regular field grid or a
scattered point cloud. Reduction is performed during model initialization.
Each retained histogram bin becomes a FOXES state, while bin statistics and
weights remain spatially resolved on the selected support topology.
The :func:`read_binned_data` factory selects the matching topology class from
artifact metadata.
"""

from .field_data import BinnedFieldData as BinnedFieldData
from .point_cloud_data import BinnedPointCloudData as BinnedPointCloudData
from ._factory import read_binned_data as read_binned_data
