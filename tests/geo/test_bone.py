from pathlib import Path

import numpy as np
import pytest
import SimpleITK as sitk
from vtkmodules.vtkFiltersCore import vtkFeatureEdges, vtkMassProperties

from kneemesh.disk import read_label_map
from kneemesh.geo.bone import ShaftEnd, bone_mask, decimate_surface, find_shaft_ends, surface_arrays
from kneemesh.vol import extract_isocontour, signed_distance_field, smooth_isocontour

# The bone_label fixture's shaft radius and length and cavity radius, in mm, and its voxel spacing.
SHAFT_RADIUS = 6.0
SHAFT_LENGTH = 25.0
CAVITY_RADIUS = 2.0
SPACING = 0.5

DATA = Path(__file__).parent.parent / "dat"


def volume(surface):
    properties = vtkMassProperties()
    properties.SetInputData(surface)
    properties.Update()
    return properties.GetVolume()


def open_edges(surface):
    edges = vtkFeatureEdges()
    edges.SetInputData(surface)
    edges.BoundaryEdgesOn()
    edges.NonManifoldEdgesOn()
    edges.FeatureEdgesOff()
    edges.ManifoldEdgesOff()
    edges.Update()
    return edges.GetOutput().GetNumberOfCells()


def extent(mask, axis=0):
    """The number of slices along an image axis that hold the mask."""
    counts = sitk.GetArrayViewFromImage(mask).sum(axis=tuple(other for other in range(3) if other != 2 - axis))
    return int(np.count_nonzero(counts))


def test_find_shaft_ends_finds_where_the_scan_cuts_the_shaft(bone_label):
    assert find_shaft_ends(bone_label) == [ShaftEnd(axis=0, side=-1, last=0, first=61)]


@pytest.mark.parametrize(("bone", "count"), [("femur", 1), ("tibia", 1), ("patella", 0)])
def test_find_shaft_ends_of_the_real_bones(bone, count):
    assert len(find_shaft_ends(read_label_map(DATA / f"{bone}.nii"))) == count


def test_bone_mask_fills_the_cavity(bone_label):
    mask = sitk.GetArrayFromImage(bone_mask(bone_label))
    label = sitk.GetArrayFromImage(bone_label)

    filled = np.count_nonzero(mask) - np.count_nonzero(label)
    assert filled == pytest.approx(4 / 3 * np.pi * CAVITY_RADIUS**3 / SPACING**3, rel=0.15)
    assert np.all(mask >= label)


def test_bone_mask_fills_a_pocket_opening_onto_the_cut(bone_label):
    # A pocket in the last slices, open only toward the cut.
    array = sitk.GetArrayFromImage(bone_label)
    array[16:20, 16:20, :4] = 0
    pocketed = sitk.GetImageFromArray(array)
    pocketed.CopyInformation(bone_label)

    mask = sitk.GetArrayFromImage(bone_mask(pocketed))

    assert np.all(mask[16:20, 16:20, :4] == 1)


@pytest.mark.parametrize("fraction", [0.2, 0.5])
def test_bone_mask_cuts_the_shaft_shorter(bone_label, fraction):
    full = extent(bone_label)

    cut = bone_mask(bone_label, shaft_cut_fraction=fraction)

    assert extent(cut) == full - round(fraction * full)
    # Only the shaft's end is cut: the rounded end stays.
    assert sitk.GetArrayViewFromImage(cut)[:, :, -1].sum() == sitk.GetArrayViewFromImage(bone_label)[:, :, -1].sum()


def test_bone_mask_leaves_a_bone_inside_the_scan_whole():
    patella = read_label_map(DATA / "patella.nii")

    assert extent(bone_mask(patella, shaft_cut_fraction=0.5)) == extent(patella)


@pytest.mark.parametrize("fraction", [-0.1, 1.0])
def test_bone_mask_rejects_fractions_out_of_range(bone_label, fraction):
    with pytest.raises(ValueError, match="shaft_cut_fraction"):
        bone_mask(bone_label, shaft_cut_fraction=fraction)


def test_bone_surface_is_closed_and_holds_the_filled_bone(bone_surface):
    expected = np.pi * SHAFT_RADIUS**2 * SHAFT_LENGTH + 2 / 3 * np.pi * SHAFT_RADIUS**3

    assert open_edges(bone_surface) == 0
    # Smoothing rounds the flat end where the scan cuts the shaft.
    assert volume(bone_surface) == pytest.approx(expected, rel=0.05)


def test_decimation_reaches_the_triangle_size_and_keeps_the_shape(bone_label):
    surface = smooth_isocontour(extract_isocontour(signed_distance_field(bone_mask(bone_label), 1)), non_manifold_smoothing=False)

    decimated = decimate_surface(surface, 1.0)

    points, triangles = surface_arrays(decimated)
    edges = np.linalg.norm(points[triangles[:, [1, 2, 0]]] - points[triangles], axis=2)
    assert decimated.GetNumberOfCells() < surface.GetNumberOfCells() / 2
    assert np.median(edges) == pytest.approx(1.0, rel=0.3)
    assert open_edges(decimated) == 0
    assert volume(decimated) == pytest.approx(volume(surface), rel=0.005)


def test_decimation_leaves_a_surface_already_coarser_alone(bone_surface):
    assert decimate_surface(bone_surface, 0.1) is bone_surface


def test_decimation_rejects_a_size_that_is_not_positive(bone_surface):
    with pytest.raises(ValueError, match="triangle_size"):
        decimate_surface(bone_surface, 0.0)
