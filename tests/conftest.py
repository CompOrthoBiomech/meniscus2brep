import math
from pathlib import Path

import numpy as np
import pytest
import SimpleITK as sitk
from scipy.interpolate import BSpline
from vtkmodules.vtkCommonDataModel import vtkPolyData

from kneemesh.brep import BRep, contours_to_polydata
from kneemesh.cartilage import CartilageSides, classify_sides
from kneemesh.config import DEFAULT_CARTILAGE_ELEMENT_SIZE
from kneemesh.disk import read_label_map
from kneemesh.geo.bone import bone_mask, decimate_surface
from kneemesh.geo.femur import fit_femoral_sheet
from kneemesh.geo.meniscus import centerline, contours_to_brep, contours_to_hex_blocks, slice_isocontour
from kneemesh.geo.sheet import CartilageSheet, fit_sheet, sheet_to_brep
from kneemesh.mesh import sheet
from kneemesh.mesh.meniscus import hexahedral_mesh
from kneemesh.segmentation import Segmentation, assemble_label_maps
from kneemesh.tet_meshing import Mesh, tetrahedral_mesh
from kneemesh.vol import ContourSlice, extract_isocontour, signed_distance_field, smooth_isocontour

INNER_RADIUS = 10.0
OUTER_RADIUS = 20.0
HEIGHT = 8.0
SWEEP_DEGREES = 270.0

BONE_RADIUS = 40.0
CARTILAGE_THICKNESS = 3.0
CARTILAGE_SEMI_AXES = (15.0, 10.0)
THIN_CENTER = (-6.0, 0.0)
DEFECT_CENTER = (6.0, 0.0)
DEFECT_RADIUS = 2.5

SHAFT_RADIUS = 6.0
SHAFT_LENGTH = 25.0
CAVITY_RADIUS = 2.0

FEMUR_RADIUS = 20.0
FEMORAL_THICKNESS = 2.0
TROCHLEA_DEGREES = 68.0
CONDYLE_DEGREES = 136.0
NOTCH_HALF_WIDTH = 6.0
FEMORAL_HALF_WIDTH = 18.0


@pytest.fixture(scope="session")
def meniscus_label() -> sitk.Image:
    """Approximate knee meniscus as a sitkUInt8 LabelMap with label 1.

    A triangular (r, z) cross section, with vertices (INNER_RADIUS, -HEIGHT / 2),
    (OUTER_RADIUS, -HEIGHT / 2), and (OUTER_RADIUS, HEIGHT / 2), swept
    SWEEP_DEGREES about the z axis through the image center. The sweep is
    centered on +x, leaving the gap facing -x. Physical units are mm.
    """
    source = sitk.PhysicalPointImageSource()
    source.SetOutputPixelType(sitk.sitkVectorFloat64)
    source.SetSize((101, 101, 17))
    source.SetSpacing((0.5, 0.5, 1.0))
    source.SetOrigin((-25.0, -25.0, -8.0))
    points = source.Execute()

    x = sitk.VectorIndexSelectionCast(points, 0)
    y = sitk.VectorIndexSelectionCast(points, 1)
    z = sitk.VectorIndexSelectionCast(points, 2)

    radius = sitk.Sqrt(x * x + y * y)
    angle = sitk.Abs(sitk.Atan2(y, x))
    height_above_base = z + HEIGHT / 2
    rim_fraction = (radius - INNER_RADIUS) / (OUTER_RADIUS - INNER_RADIUS)

    cross_section = (radius <= OUTER_RADIUS) & (height_above_base >= 0.0) & (height_above_base <= HEIGHT * rim_fraction)
    return cross_section & (angle <= math.radians(SWEEP_DEGREES / 2))


@pytest.fixture(scope="session")
def meniscus_centerline(meniscus_label: sitk.Image) -> BSpline:
    """Centerline of meniscus_label."""
    return centerline(meniscus_label)


@pytest.fixture(scope="session")
def meniscus_distance_field(meniscus_label: sitk.Image) -> sitk.Image:
    """Signed distance field of meniscus_label."""
    return signed_distance_field(meniscus_label)


@pytest.fixture(scope="session")
def meniscus_mesh(meniscus_distance_field: sitk.Image) -> vtkPolyData:
    """Surface mesh on the boundary of meniscus_label."""
    return extract_isocontour(meniscus_distance_field)


@pytest.fixture(scope="session")
def meniscus_smoothed_mesh(meniscus_mesh: vtkPolyData) -> vtkPolyData:
    """meniscus_mesh after default windowed sinc smoothing."""
    return smooth_isocontour(meniscus_mesh)


@pytest.fixture(scope="session")
def meniscus_slices(meniscus_mesh: vtkPolyData, meniscus_centerline: BSpline) -> list[ContourSlice]:
    """Slices of meniscus_mesh every 5 mm along meniscus_centerline."""
    return slice_isocontour(meniscus_mesh, meniscus_centerline, spacing=5.0)


@pytest.fixture(scope="session")
def meniscus_contours(meniscus_slices: list[ContourSlice]) -> vtkPolyData:
    """Slices of meniscus_mesh every 5 mm along meniscus_centerline."""
    return contours_to_polydata(meniscus_slices)


@pytest.fixture(scope="session")
def meniscus_surface(meniscus_slices: list[ContourSlice]) -> BRep:
    """Open cubic surface lofted through meniscus_slices."""
    return contours_to_brep(meniscus_slices)


@pytest.fixture(scope="session")
def meniscus_solid(meniscus_slices: list[ContourSlice]) -> BRep:
    """Cubic solid lofted through meniscus_slices with planar caps."""
    return contours_to_brep(meniscus_slices, cap="planar")


@pytest.fixture(scope="session")
def meniscus_domed_solid(meniscus_slices: list[ContourSlice]) -> BRep:
    """Cubic solid lofted through meniscus_slices with smooth caps."""
    return contours_to_brep(meniscus_slices, cap="smooth")


@pytest.fixture(scope="session")
def meniscus_hex_slices(meniscus_mesh: vtkPolyData, meniscus_centerline: BSpline) -> list[ContourSlice]:
    """10-point slices of meniscus_mesh every 5 mm, for building hexahedral blocks."""
    return slice_isocontour(meniscus_mesh, meniscus_centerline, spacing=5.0, points_per_slice=10)


@pytest.fixture(scope="session")
def meniscus_hex_blocks(meniscus_hex_slices: list[ContourSlice]) -> BRep:
    """Three-block solid for hexahedral meshing, built from meniscus_hex_slices."""
    return contours_to_hex_blocks(meniscus_hex_slices)


@pytest.fixture(scope="session")
def real_label() -> sitk.Image:
    """A real lateral meniscus LabelMap (label 1) whose arc touches the edge of the image."""
    return read_label_map(Path(__file__).parent / "dat" / "lateral_meniscus.nii")


@pytest.fixture(scope="session")
def meniscus_tetrahedra(meniscus_solid: BRep) -> Mesh:
    """Linear tetrahedral mesh of meniscus_solid with a 2 mm maximum size."""
    return tetrahedral_mesh(meniscus_solid, max_size=2.0)


@pytest.fixture(scope="session")
def meniscus_hexahedra(meniscus_hex_blocks: BRep) -> Mesh:
    """Linear hexahedral mesh of meniscus_hex_blocks, 4 elements per cap edge and 30 along."""
    return hexahedral_mesh(meniscus_hex_blocks, elements_per_edge=4, elements_along=30)


@pytest.fixture(scope="session")
def real_medial_label() -> sitk.Image:
    """A real medial meniscus LabelMap (label 1) from the same knee as real_label, on its own grid."""
    return read_label_map(Path(__file__).parent / "dat" / "medial_meniscus.nii")


def _cartilage_on_bone(defect: bool) -> Segmentation:
    """A synthetic tibial cartilage cap (label 2) on a spherical bone (label 1), physical units mm.

    The bone is a sphere of radius BONE_RADIUS whose top touches z = 0. The
    cartilage lies on it, over an ellipse with CARTILAGE_SEMI_AXES in x and
    y, with a thickness of CARTILAGE_THICKNESS at its center tapering to 0 at
    its rim like an ellipsoid's. A Gaussian dip of sigma 2.5 mm around
    THIN_CENTER thins it to 15% there, and with defect a cylinder of DEFECT_RADIUS around
    DEFECT_CENTER has no cartilage at all.
    """
    spacing, origin, size = 0.5, (-18.75, -13.75, -6.0), (76, 56, 22)
    z, y, x = (
        origin[axis] + spacing * np.arange(size[axis])[shape]
        for axis, shape in ((2, np.s_[:, None, None]), (1, np.s_[None, :, None]), (0, np.s_[None, None, :]))
    )
    radius = np.sqrt(x**2 + y**2 + (z + BONE_RADIUS) ** 2)
    semi_x, semi_y = CARTILAGE_SEMI_AXES
    taper = np.sqrt(np.clip(1 - (x / semi_x) ** 2 - (y / semi_y) ** 2, 0, None))
    dip = 1 - 0.85 * np.exp(-((x - THIN_CENTER[0]) ** 2 + (y - THIN_CENTER[1]) ** 2) / (2 * 2.5**2))
    cartilage = (radius >= BONE_RADIUS) & (radius <= BONE_RADIUS + CARTILAGE_THICKNESS * taper * dip)
    if defect:
        cartilage &= (x - DEFECT_CENTER[0]) ** 2 + (y - DEFECT_CENTER[1]) ** 2 > DEFECT_RADIUS**2
    image = sitk.GetImageFromArray(np.where(cartilage, 2, np.where(radius < BONE_RADIUS, 1, 0)).astype(np.uint8))
    image.SetSpacing((spacing,) * 3)
    image.SetOrigin(origin)
    return Segmentation(image, {"tibia": 1, "lateral_tibial_cartilage": 2})


def _sides(segmentation: Segmentation) -> CartilageSides:
    surface = smooth_isocontour(extract_isocontour(signed_distance_field(segmentation.image, 2)))
    return classify_sides(surface, signed_distance_field(segmentation.image, 1))


@pytest.fixture(scope="session")
def cartilage_segmentation() -> Segmentation:
    """A synthetic tibial cartilage on its bone, with a thin dip; see _cartilage_on_bone."""
    return _cartilage_on_bone(defect=False)


@pytest.fixture(scope="session")
def cartilage_defect_segmentation() -> Segmentation:
    """cartilage_segmentation with a full-thickness defect."""
    return _cartilage_on_bone(defect=True)


@pytest.fixture(scope="session")
def cartilage_sides(cartilage_segmentation: Segmentation) -> CartilageSides:
    """The articular and bone sides of cartilage_segmentation's smoothed surface."""
    return _sides(cartilage_segmentation)


@pytest.fixture(scope="session")
def cartilage_defect_sides(cartilage_defect_segmentation: Segmentation) -> CartilageSides:
    """The articular and bone sides of cartilage_defect_segmentation's smoothed surface."""
    return _sides(cartilage_defect_segmentation)


@pytest.fixture(scope="session")
def cartilage_sheet(cartilage_sides: CartilageSides) -> CartilageSheet:
    """The sheet of cartilage_sides with the defaults for hexahedra."""
    return fit_sheet(cartilage_sides)


@pytest.fixture(scope="session")
def cartilage_hex_solid(cartilage_sheet: CartilageSheet) -> BRep:
    """The solid of cartilage_sheet, with the defaults for hexahedra."""
    return sheet_to_brep(cartilage_sheet)


@pytest.fixture(scope="session")
def cartilage_hexahedra(cartilage_hex_solid: BRep) -> Mesh:
    """Linear hexahedral mesh of cartilage_hex_solid at the default size."""
    return sheet.hexahedral_mesh(cartilage_hex_solid, DEFAULT_CARTILAGE_ELEMENT_SIZE)


@pytest.fixture(scope="session")
def cartilage_solid(cartilage_sides: CartilageSides) -> BRep:
    """The solid of cartilage_sides's sheet with the defaults for tetrahedra."""
    return sheet_to_brep(fit_sheet(cartilage_sides, perimeter_radius=1.0, min_thickness=0.1))


@pytest.fixture(scope="session")
def real_cartilage() -> dict[str, CartilageSides]:
    """The articular and bone sides of each real tibial and patellar cartilage, assembled with its bone by nearest neighbor."""
    data = Path(__file__).parent / "dat"
    sides = {}
    for cartilage, bone in (("lateral_tibial_cartilage", "tibia"), ("medial_tibial_cartilage", "tibia"), ("patellar_cartilage", "patella")):
        images = {bone: read_label_map(data / f"{bone}.nii"), cartilage: read_label_map(data / f"{cartilage}.nii")}
        sides[cartilage] = _sides(assemble_label_maps(images, "nearest_neighbor"))
    return sides


@pytest.fixture(scope="session")
def real_cartilage_hex_solids(real_cartilage: dict[str, CartilageSides]) -> dict[str, BRep]:
    """The solid of each real cartilage's sheet with the defaults for hexahedra."""
    return {structure: sheet_to_brep(fit_sheet(sides)) for structure, sides in real_cartilage.items()}


def _femoral_on_bone() -> Segmentation:
    """A synthetic femoral cartilage (label 2) on a cylindrical femur (label 1), physical units mm.

    The femur is a cylinder of radius FEMUR_RADIUS along x. The cartilage is
    a shell FEMORAL_THICKNESS thick on it: a trochlea over TROCHLEA_DEGREES
    from the +y axis toward +z across |x| <= FEMORAL_HALF_WIDTH, continued
    by two condyles over a further CONDYLE_DEGREES where NOTCH_HALF_WIDTH <=
    |x| <= FEMORAL_HALF_WIDTH, so unrolled it is a U around the notch.
    """
    spacing, origin, size = 0.5, (-22.0, -24.0, -24.0), (89, 97, 97)
    z, y, x = (
        origin[axis] + spacing * np.arange(size[axis])[shape]
        for axis, shape in ((2, np.s_[:, None, None]), (1, np.s_[None, :, None]), (0, np.s_[None, None, :]))
    )
    radius = np.sqrt(y**2 + z**2)
    angle = np.degrees(np.arctan2(z, y)) % 360
    across = np.abs(x)
    trochlea = (angle <= TROCHLEA_DEGREES) & (across <= FEMORAL_HALF_WIDTH)
    condyles = (
        (angle > TROCHLEA_DEGREES) & (angle <= TROCHLEA_DEGREES + CONDYLE_DEGREES) & (across >= NOTCH_HALF_WIDTH) & (across <= FEMORAL_HALF_WIDTH)
    )
    cartilage = (radius >= FEMUR_RADIUS) & (radius <= FEMUR_RADIUS + FEMORAL_THICKNESS) & (trochlea | condyles)
    bone = (radius < FEMUR_RADIUS) & (across <= FEMORAL_HALF_WIDTH + 4.0)
    image = sitk.GetImageFromArray(np.where(cartilage, 2, np.where(bone, 1, 0)).astype(np.uint8))
    image.SetSpacing((spacing,) * 3)
    image.SetOrigin(origin)
    return Segmentation(image, {"femur": 1, "femoral_cartilage": 2})


@pytest.fixture(scope="session")
def femoral_segmentation() -> Segmentation:
    """A synthetic femoral cartilage on its femur; see femoral_on_bone."""
    return _femoral_on_bone()


@pytest.fixture(scope="session")
def femoral_sides(femoral_segmentation: Segmentation) -> CartilageSides:
    """The articular and bone sides of femoral_segmentation's smoothed surface."""
    return _sides(femoral_segmentation)


@pytest.fixture(scope="session")
def femoral_sheet(femoral_sides: CartilageSides) -> CartilageSheet:
    """The sheet of femoral_sides over its cylinder, with the defaults for hexahedra."""
    return fit_femoral_sheet(femoral_sides)


@pytest.fixture(scope="session")
def femoral_hex_solid(femoral_sheet: CartilageSheet) -> BRep:
    """The solid of femoral_sheet, with the defaults for hexahedra."""
    return sheet_to_brep(femoral_sheet)


@pytest.fixture(scope="session")
def femoral_hexahedra(femoral_hex_solid: BRep) -> Mesh:
    """Linear hexahedral mesh of femoral_hex_solid at the default size."""
    return sheet.hexahedral_mesh(femoral_hex_solid, DEFAULT_CARTILAGE_ELEMENT_SIZE)


@pytest.fixture(scope="session")
def femoral_solid(femoral_sides: CartilageSides) -> BRep:
    """The solid of femoral_sides's sheet with the defaults for tetrahedra."""
    return sheet_to_brep(fit_femoral_sheet(femoral_sides, perimeter_radius=1.0, min_thickness=0.1, hole_thickness=0.05))


@pytest.fixture(scope="session")
def real_femoral_sides() -> CartilageSides:
    """The articular and bone sides of the real femoral cartilage, assembled with the femur by nearest neighbor."""
    data = Path(__file__).parent / "dat"
    images = {"femur": read_label_map(data / "femur.nii"), "femoral_cartilage": read_label_map(data / "femoral_cartilage.nii")}
    segmentation = assemble_label_maps(images, "nearest_neighbor")
    surface = smooth_isocontour(extract_isocontour(signed_distance_field(segmentation.image, segmentation.value("femoral_cartilage"))))
    return classify_sides(surface, signed_distance_field(segmentation.image, segmentation.value("femur")))


@pytest.fixture(scope="session")
def real_femoral_sheet(real_femoral_sides: CartilageSides) -> CartilageSheet:
    """The real femoral cartilage's sheet with the defaults for hexahedra."""
    return fit_femoral_sheet(real_femoral_sides)


@pytest.fixture(scope="session")
def real_femoral_hex_solid(real_femoral_sheet: CartilageSheet) -> BRep:
    """The real femoral cartilage's solid at the defaults for hexahedra."""
    return sheet_to_brep(real_femoral_sheet)


def bone_surface_of(label_map: sitk.Image, label: int = 1, shaft_cut_fraction: float = 0.0) -> vtkPolyData:
    """A bone's decimated surface, made as the B-rep pipeline makes it."""
    distance_field = signed_distance_field(bone_mask(label_map, label, shaft_cut_fraction), 1)
    return decimate_surface(smooth_isocontour(extract_isocontour(distance_field), non_manifold_smoothing=False))


@pytest.fixture(scope="session")
def bone_label() -> sitk.Image:
    """A synthetic bone (label 1), physical units mm: a shaft of radius SHAFT_RADIUS along x, cut off by the image at x = 0.

    The shaft runs SHAFT_LENGTH from the image's edge to a hemispherical end,
    and holds a spherical cavity of radius CAVITY_RADIUS at its middle.
    """
    spacing, origin, size = 0.5, (0.25, -8.75, -8.75), (66, 36, 36)
    z, y, x = (
        origin[axis] + spacing * np.arange(size[axis])[shape]
        for axis, shape in ((2, np.s_[:, None, None]), (1, np.s_[None, :, None]), (0, np.s_[None, None, :]))
    )
    radial = y**2 + z**2
    shaft = (radial <= SHAFT_RADIUS**2) & (x <= SHAFT_LENGTH)
    end = radial + (x - SHAFT_LENGTH) ** 2 <= SHAFT_RADIUS**2
    cavity = radial + (x - SHAFT_LENGTH / 2) ** 2 <= CAVITY_RADIUS**2
    image = sitk.GetImageFromArray(((shaft | end) & ~cavity).astype(np.uint8))
    image.SetSpacing((spacing,) * 3)
    image.SetOrigin(origin)
    return image


@pytest.fixture(scope="session")
def bone_surface(bone_label: sitk.Image) -> vtkPolyData:
    """The closed surface of bone_label, its cavity filled."""
    return bone_surface_of(bone_label)


@pytest.fixture(scope="session")
def real_patella_surface() -> vtkPolyData:
    """The closed surface of the real patella."""
    return bone_surface_of(read_label_map(Path(__file__).parent / "dat" / "patella.nii"))
