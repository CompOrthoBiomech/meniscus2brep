"""Operations on segmented LabelMaps that apply to any knee structure."""

from dataclasses import dataclass

import numpy as np
import SimpleITK as sitk
from loguru import logger
from vtkmodules.util.numpy_support import numpy_to_vtk
from vtkmodules.vtkCommonDataModel import vtkImageData, vtkPolyData
from vtkmodules.vtkFiltersCore import vtkFlyingEdges3D, vtkWindowedSincPolyDataFilter

# A Gaussian's central 95% spans 2 * 1.96 sigma. Setting that to 3 voxel edges gives this sigma in voxels.
_DEFAULT_SIGMA_VOXELS = 3 / (2 * 1.96)

# Windowed sinc smoothing defaults, chosen on the test fixture and a real
# meniscus: they remove most of the ripple that makes slice contours wiggle
# while shrinking the volume by about 3%.
DEFAULT_SMOOTHING_ITERATIONS = 50
DEFAULT_SMOOTHING_PASS_BAND = 0.001

# Integer pixel types accepted for LabelMaps; 16-bit types allow more labels.
LABEL_MAP_PIXEL_IDS = (sitk.sitkUInt8, sitk.sitkUInt16, sitk.sitkInt16)


@dataclass(frozen=True, eq=False)
class ContourSlice:
    """An ordered closed contour of a surface in a plane normal to a curve.

    Attributes:
        points: (N, 3) physical points, evenly spaced around the contour and
            counterclockwise about normal. The closing point is not repeated.
            The first slice starts in the +u_axis direction from the contour's
            center. Each later slice starts wherever best lines its points up
            with the previous slice's, so corresponding points match along the curve.
        center: Point on the curve where the plane is centered.
        u_axis: Unit vector in the plane.
        v_axis: Unit vector in the plane. (u_axis, v_axis, normal) is right-handed.
        normal: Unit tangent of the curve at center.
        area: Area enclosed by the contour.
    """

    points: np.ndarray
    center: np.ndarray
    u_axis: np.ndarray
    v_axis: np.ndarray
    normal: np.ndarray
    area: float


def signed_distance_field(label_map: sitk.Image, label: int = 1, sigma: float | None = None) -> sitk.Image:
    """Compute a smoothed signed distance to a label's boundary, in physical units.

    A recursive Gaussian removes the voxel staircase from the field's level sets.

    Args:
        label_map: 3D LabelMap of sitkUInt8, sitkUInt16, or sitkInt16 pixels.
        label: Value of the structure.
        sigma: Gaussian standard deviation in physical units, applied equally
            to every axis. By default each axis uses the sigma whose central
            95% spans 3 of that axis's voxel edges. 0 disables smoothing.

    Returns:
        A float image that is negative inside the label and positive outside.
        Without smoothing, its zero level set lies on the voxel edges of the
        label's boundary.

    Raises:
        ValueError: If the image is not a 3D LabelMap of an accepted pixel type, the label is
            absent, or sigma is negative.
    """
    if sigma is not None and sigma < 0:
        raise ValueError(f"sigma must not be negative, got {sigma}")
    require_3d_label_map(label_map)
    mask = sitk.BinaryThreshold(label_map, label, label, 1, 0)
    if not sitk.GetArrayViewFromImage(mask).any():
        raise ValueError(f"Label {label} is not present in the image")

    # One map alone is zero on the centers of the boundary voxels, half a voxel
    # inside the label's edge. Averaging with the inverse's map centers it.
    inside = _maurer_distance(mask)
    outside = _maurer_distance(sitk.BinaryNot(mask))
    field = (inside - outside) * 0.5
    if sigma == 0:
        return field
    sigmas = [_DEFAULT_SIGMA_VOXELS * size for size in label_map.GetSpacing()] if sigma is None else [sigma] * 3
    logger.debug("Smoothing the distance field with sigmas {}", [round(value, 4) for value in sigmas])
    return sitk.SmoothingRecursiveGaussian(field, sigmas)


def extract_isocontour(distance_field: sitk.Image, level: float = 0.0) -> vtkPolyData:
    """Extract a triangle mesh at a given distance from a label's boundary.

    Args:
        distance_field: Float image from signed_distance_field.
        level: Distance of the surface from the label boundary. Positive
            values lie outside the label.

    Returns:
        A closed triangle mesh in physical coordinates.

    Raises:
        ValueError: If the distance field never reaches level.
    """
    values = sitk.GetArrayViewFromImage(distance_field)
    if not values.min() <= level <= values.max():
        raise ValueError(f"The distance field has no surface at level {level}")

    # Padding with an outside value closes the surface where the structure
    # touches the edge of the image.
    outside = float(values.max()) + max(distance_field.GetSpacing())
    padded = sitk.ConstantPad(distance_field, (1, 1, 1), (1, 1, 1), outside)

    flying_edges = vtkFlyingEdges3D()
    flying_edges.SetInputData(_to_vtk_image(padded))
    flying_edges.SetValue(0, level)
    flying_edges.ComputeNormalsOff()
    flying_edges.ComputeGradientsOff()
    flying_edges.ComputeScalarsOff()
    flying_edges.Update()

    surface = flying_edges.GetOutput()
    logger.debug("Surface at level {} has {} points and {} triangles", level, surface.GetNumberOfPoints(), surface.GetNumberOfCells())
    return surface


def smooth_isocontour(
    mesh: vtkPolyData,
    iterations: int = DEFAULT_SMOOTHING_ITERATIONS,
    pass_band: float = DEFAULT_SMOOTHING_PASS_BAND,
    non_manifold_smoothing: bool = True,
) -> vtkPolyData:
    """Smooth a surface mesh with a windowed sinc filter, keeping its connectivity.

    Boundary and feature-edge smoothing are off, so any open edges stay in
    place, and non-manifold smoothing is on unless turned off. Coordinates are normalized to the unit cube
    while filtering, so the result does not depend on the physical scale.

    Args:
        mesh: Surface from extract_isocontour.
        iterations: Number of filter iterations, which is the degree of the
            filter's polynomial. More iterations follow the pass band more
            closely. 0 returns the mesh unchanged.
        pass_band: Pass band of the filter, between 0 and 2. Lower values
            smooth more and shrink the surface more.
        non_manifold_smoothing: Smooth vertices on non-manifold edges too.

    Returns:
        A new smoothed mesh, or mesh itself when iterations is 0.

    Raises:
        ValueError: If iterations is negative or pass_band is not in (0, 2].
    """
    if iterations < 0:
        raise ValueError(f"Smoothing iterations must not be negative, got {iterations}")
    if not 0 < pass_band <= 2:
        raise ValueError(f"Smoothing pass band must be in (0, 2], got {pass_band}")
    if iterations == 0:
        return mesh

    smoother = vtkWindowedSincPolyDataFilter()
    smoother.SetInputData(mesh)
    smoother.SetNumberOfIterations(iterations)
    smoother.SetPassBand(pass_band)
    smoother.BoundarySmoothingOff()
    smoother.FeatureEdgeSmoothingOff()
    smoother.SetNonManifoldSmoothing(non_manifold_smoothing)
    smoother.NormalizeCoordinatesOn()
    smoother.Update()
    logger.debug("Smoothed the surface with {} iterations and pass band {}", iterations, pass_band)
    return smoother.GetOutput()


def require_3d_label_map(label_map: sitk.Image) -> None:
    """Raise ValueError unless label_map is a 3D LabelMap of an accepted pixel type."""
    if label_map.GetDimension() != 3 or label_map.GetPixelID() not in LABEL_MAP_PIXEL_IDS:
        raise ValueError(
            f"LabelMap must be 3D sitkUInt8, sitkUInt16, or sitkInt16, got {label_map.GetDimension()}D {label_map.GetPixelIDTypeAsString()}"
        )


def _maurer_distance(mask: sitk.Image) -> sitk.Image:
    return sitk.SignedMaurerDistanceMap(mask, insideIsPositive=False, squaredDistance=False, useImageSpacing=True)


def _to_vtk_image(image: sitk.Image) -> vtkImageData:
    result = vtkImageData()
    result.SetDimensions(image.GetSize())
    result.SetSpacing(image.GetSpacing())
    result.SetOrigin(image.GetOrigin())
    result.SetDirectionMatrix(image.GetDirection())
    # SimpleITK's z, y, x array flattens to VTK's x-fastest point order.
    result.GetPointData().SetScalars(numpy_to_vtk(sitk.GetArrayFromImage(image).ravel(), deep=True))
    return result
