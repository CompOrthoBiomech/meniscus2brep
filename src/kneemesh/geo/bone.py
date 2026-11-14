"""Surfaces of the bones: the femur, tibia, and patella.

A bone's geometry is not a B-rep but its closed, triangulated surface, which
gmsh reparametrizes when meshing it (mesh.bone). Bones are not meshed with
hexahedra. The surface is the isocontour of the bone's mask in the voxel
map, its cavities filled and, optionally, its shafts cut shorter, smoothed
and decimated.
"""

import math
from dataclasses import dataclass

import numpy as np
import SimpleITK as sitk
from loguru import logger
from vtkmodules.util.numpy_support import vtk_to_numpy
from vtkmodules.vtkCommonDataModel import vtkPolyData
from vtkmodules.vtkFiltersCore import vtkCleanPolyData, vtkMassProperties, vtkQuadricDecimation, vtkTriangleFilter

from kneemesh.vol import require_3d_label_map

DEFAULT_TRIANGLE_SIZE = 1.0
"""Default edge length of the bone surface's triangles after decimation, in physical units."""

# A label is cut off by the scan where its last slice along an image axis holds more than _SHAFT_END_FRACTION
# of its largest slice's voxels and more than _SHAFT_END_FLATNESS of the slice _SHAFT_END_DEPTH inward: a cut
# shaft's cross-section barely changes there. The real femur's and tibia's cuts hold 15-22% and 80-83%; their
# other ends and sides at most 5% and 27%, and the side of a synthetic cylinder 29% and 33%.
_SHAFT_END_FRACTION = 0.05
_SHAFT_END_FLATNESS = 0.6
_SHAFT_END_DEPTH = 2.0


@dataclass(frozen=True)
class ShaftEnd:
    """Where a bone's shaft is cut off by the edge of the scan.

    Attributes:
        axis: Index of the image axis along the shaft.
        side: -1 if the shaft leaves the scan at the axis's low end, 1 at its high end.
        last: Index along axis of the label's last slice at that end.
        first: Index along axis of the label's last slice at the other end.
    """

    axis: int
    side: int
    last: int
    first: int


def find_shaft_ends(label_map: sitk.Image, label: int = 1) -> list[ShaftEnd]:
    """Find where a bone's label is cut off by the edge of the scan, as the femur's and tibia's shafts are.

    A bone that ends inside the scan tapers to a few voxels in its last slice
    along every image axis, and its slices grow quickly inward; one cut off
    by the scan ends in a whole cross-section, nearly the same as the slices
    just inside it.

    Raises:
        ValueError: If the image is not a 3D LabelMap or the label is absent.
    """
    require_3d_label_map(label_map)
    mask = sitk.GetArrayViewFromImage(label_map) == label
    if not mask.any():
        raise ValueError(f"Label {label} is not present in the image")
    ends = []
    for axis in range(3):
        # SimpleITK's arrays index axes in reverse.
        counts = mask.sum(axis=tuple(other for other in range(3) if other != 2 - axis))
        occupied = np.flatnonzero(counts)
        low, high = int(occupied[0]), int(occupied[-1])
        depth = math.ceil(_SHAFT_END_DEPTH / label_map.GetSpacing()[axis])
        for side, last, first in ((-1, low, high), (1, high, low)):
            inward = int(np.clip(last - side * depth, low, high))
            if counts[last] > _SHAFT_END_FRACTION * counts.max() and counts[last] > _SHAFT_END_FLATNESS * counts[inward]:
                ends.append(ShaftEnd(axis, side, last, first))
                logger.debug("Label {} is cut off by the scan at the {} end of image axis {}", label, "low" if side < 0 else "high", axis)
    return ends


def bone_mask(label_map: sitk.Image, label: int = 1, shaft_cut_fraction: float = 0.0) -> sitk.Image:
    """Return a mask of a bone (1 inside, 0 outside) with its cavities filled and its shafts optionally cut shorter.

    Each shaft the scan cuts off (find_shaft_ends) loses shaft_cut_fraction
    of the label's extent along it, cut square in the voxel map, so the
    surface ends in a flat face there, rounded at its edge by smoothing.
    Cavities are then filled, first in each shaft's new last slice, where a
    pocket would open onto the flat face, then throughout.

    Args:
        label_map: 3D LabelMap.
        label: Value of the bone.
        shaft_cut_fraction: Fraction of the bone's extent along each shaft cut
            off by the scan to remove from that end, in [0, 1). 0 keeps the
            shaft as the scan has it.

    Raises:
        ValueError: If the image is not a 3D LabelMap, the label is absent, or
            shaft_cut_fraction is out of range.
    """
    if not 0 <= shaft_cut_fraction < 1:
        raise ValueError(f"shaft_cut_fraction must be in [0, 1), got {shaft_cut_fraction}")
    ends = find_shaft_ends(label_map, label)
    array = (sitk.GetArrayFromImage(label_map) == label).astype(np.uint8)
    for end in ends:
        slices = np.moveaxis(array, 2 - end.axis, 0)
        removed = round(shaft_cut_fraction * (abs(end.last - end.first) + 1))
        last = end.last - end.side * removed
        if end.side < 0:
            slices[:last] = 0
        else:
            slices[last + 1 :] = 0
        slices[last] = sitk.GetArrayFromImage(sitk.BinaryFillhole(sitk.GetImageFromArray(np.ascontiguousarray(slices[last]))))
        if removed:
            logger.debug("Cut {} slices off the shaft along image axis {}", removed, end.axis)
    mask = sitk.GetImageFromArray(array)
    mask.CopyInformation(label_map)
    return sitk.BinaryFillhole(mask)


def decimate_surface(surface: vtkPolyData, triangle_size: float = DEFAULT_TRIANGLE_SIZE) -> vtkPolyData:
    """Reduce a surface's triangles to about triangle_size across, keeping its shape and volume.

    The isocontour's triangles are a fraction of a voxel across, too many for
    gmsh to reparametrize quickly: the real femur's 340,000 took minutes.
    Quadric decimation to 1 mm edges moves the real bones' surfaces by at most 0.02 mm.

    Raises:
        ValueError: If triangle_size is not positive.
    """
    if triangle_size <= 0:
        raise ValueError(f"triangle_size must be positive, got {triangle_size}")
    properties = vtkMassProperties()
    properties.SetInputData(surface)
    properties.Update()
    target = properties.GetSurfaceArea() / (np.sqrt(3) / 4 * triangle_size**2)
    count = surface.GetNumberOfCells()
    if count <= target:
        return surface
    decimation = vtkQuadricDecimation()
    decimation.SetInputData(surface)
    decimation.SetTargetReduction(1 - target / count)
    decimation.VolumePreservationOn()
    cleaner = vtkCleanPolyData()
    cleaner.SetInputConnection(decimation.GetOutputPort())
    triangles = vtkTriangleFilter()
    triangles.SetInputConnection(cleaner.GetOutputPort())
    triangles.PassLinesOff()
    triangles.PassVertsOff()
    triangles.Update()
    decimated = triangles.GetOutput()
    logger.debug("Decimated the surface from {} to {} triangles", count, decimated.GetNumberOfCells())
    return decimated


def surface_arrays(surface: vtkPolyData) -> tuple[np.ndarray, np.ndarray]:
    """Return a triangulated surface's (N, 3) points and (M, 3) triangles' point indices."""
    points = vtk_to_numpy(surface.GetPoints().GetData()).astype(np.float64)
    triangles = vtk_to_numpy(surface.GetPolys().GetConnectivityArray()).reshape(-1, 3)
    return points, triangles
