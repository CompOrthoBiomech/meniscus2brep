import json

import gmsh
import numpy as np
import pytest
import SimpleITK as sitk
from vtkmodules.util.numpy_support import vtk_to_numpy

from kneemesh.brep import BRep, gmsh_session
from kneemesh.disk import (
    read_brep,
    read_label_map,
    read_poly_data,
    read_segmentation,
    read_surface,
    write_brep,
    write_label_map,
    write_mesh,
    write_poly_data,
    write_segmentation,
    write_surface,
)
from kneemesh.geo.bone import surface_arrays
from kneemesh.mesh.bone import surface_mesh
from kneemesh.mesh.meniscus import hexahedral_mesh
from kneemesh.segmentation import Segmentation
from kneemesh.tet_meshing import tetrahedral_mesh


@pytest.mark.parametrize("pixel_id", [sitk.sitkUInt8, sitk.sitkUInt16, sitk.sitkInt16])
@pytest.mark.parametrize("filename", ["seg.nii", "seg.nrrd", "seg.mhd"])
def test_round_trip_preserves_labels_and_geometry(tmp_path, meniscus_label, filename, pixel_id):
    meniscus_label = sitk.Cast(meniscus_label, pixel_id)
    path = tmp_path / filename

    write_label_map(meniscus_label, path)
    result = read_label_map(path)

    assert result.GetPixelID() == pixel_id
    assert result.GetSize() == meniscus_label.GetSize()
    assert sitk.Hash(result) == sitk.Hash(meniscus_label)
    assert result.GetSpacing() == pytest.approx(meniscus_label.GetSpacing())
    assert result.GetOrigin() == pytest.approx(meniscus_label.GetOrigin())
    assert result.GetDirection() == pytest.approx(meniscus_label.GetDirection())


def test_mhd_writes_separate_raw_file(tmp_path, meniscus_label):
    write_label_map(meniscus_label, tmp_path / "seg.mhd")

    assert (tmp_path / "seg.raw").is_file()


@pytest.mark.parametrize("filename", ["seg.png", "seg.nii.gz", "seg"])
def test_write_rejects_unsupported_format(tmp_path, meniscus_label, filename):
    with pytest.raises(ValueError, match="Unsupported file format"):
        write_label_map(meniscus_label, tmp_path / filename)


def test_read_rejects_unsupported_format(tmp_path):
    with pytest.raises(ValueError, match="Unsupported file format"):
        read_label_map(tmp_path / "seg.png")


@pytest.mark.parametrize("pixel_id", [sitk.sitkInt8, sitk.sitkUInt32, sitk.sitkInt32, sitk.sitkFloat32])
def test_write_rejects_other_pixel_types(tmp_path, meniscus_label, pixel_id):
    image = sitk.Cast(meniscus_label, pixel_id)

    with pytest.raises(ValueError, match="must be sitkUInt8"):
        write_label_map(image, tmp_path / "seg.nrrd")


@pytest.mark.parametrize("pixel_id", [sitk.sitkInt8, sitk.sitkUInt32, sitk.sitkInt32, sitk.sitkFloat32])
def test_read_rejects_other_pixel_types(tmp_path, meniscus_label, pixel_id):
    path = tmp_path / "seg.nrrd"
    sitk.WriteImage(sitk.Cast(meniscus_label, pixel_id), str(path))

    with pytest.raises(ValueError, match="must be sitkUInt8"):
        read_label_map(path)


def test_poly_data_round_trip_preserves_points_and_cells(tmp_path, meniscus_contours):
    path = tmp_path / "contours.vtp"

    write_poly_data(meniscus_contours, path)
    result = read_poly_data(path)

    assert np.array_equal(vtk_to_numpy(result.GetPoints().GetData()), vtk_to_numpy(meniscus_contours.GetPoints().GetData()))
    assert result.GetNumberOfLines() == meniscus_contours.GetNumberOfLines()
    assert np.array_equal(
        vtk_to_numpy(result.GetLines().GetConnectivityArray()),
        vtk_to_numpy(meniscus_contours.GetLines().GetConnectivityArray()),
    )
    assert np.array_equal(
        vtk_to_numpy(result.GetLines().GetOffsetsArray()),
        vtk_to_numpy(meniscus_contours.GetLines().GetOffsetsArray()),
    )


def test_write_poly_data_overwrites_existing_file(tmp_path, meniscus_contours):
    path = tmp_path / "contours.vtp"
    path.write_text("old contents")

    write_poly_data(meniscus_contours, path)

    assert read_poly_data(path).GetNumberOfLines() == meniscus_contours.GetNumberOfLines()


@pytest.mark.parametrize("filename", ["contours.vtk", "contours.stl", "contours"])
def test_poly_data_rejects_unsupported_format(tmp_path, meniscus_contours, filename):
    with pytest.raises(ValueError, match="Unsupported file format"):
        write_poly_data(meniscus_contours, tmp_path / filename)
    with pytest.raises(ValueError, match="Unsupported file format"):
        read_poly_data(tmp_path / filename)


def test_read_poly_data_rejects_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="No such file"):
        read_poly_data(tmp_path / "missing.vtp")


@pytest.mark.parametrize(
    "contents",
    ["not xml", '<?xml version="1.0"?><VTKFile type="ImageData" version="1.0"><ImageData></ImageData></VTKFile>'],
)
def test_read_poly_data_rejects_other_file_contents(tmp_path, contents):
    path = tmp_path / "contours.vtp"
    path.write_text(contents)

    with pytest.raises(ValueError, match="Not a VTK XML poly data"):
        read_poly_data(path)


def test_write_poly_data_raises_when_it_cannot_write(tmp_path, meniscus_contours):
    with pytest.raises(OSError, match="Could not write"):
        write_poly_data(meniscus_contours, tmp_path / "missing_directory" / "contours.vtp")


def test_brep_round_trip_preserves_text(tmp_path, meniscus_surface):
    path = tmp_path / "surface.brep"

    write_brep(meniscus_surface, path)

    assert read_brep(path) == meniscus_surface


@pytest.mark.parametrize("filename", ["surface.step", "surface.brp", "surface"])
def test_brep_rejects_unsupported_format(tmp_path, meniscus_surface, filename):
    with pytest.raises(ValueError, match="Unsupported file format"):
        write_brep(meniscus_surface, tmp_path / filename)
    with pytest.raises(ValueError, match="Unsupported file format"):
        read_brep(tmp_path / filename)


def test_read_brep_rejects_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="No such file"):
        read_brep(tmp_path / "missing.brep")


def test_read_brep_rejects_invalid_contents(tmp_path):
    path = tmp_path / "surface.brep"
    path.write_text("not a b-rep")

    with pytest.raises(ValueError, match="Not a valid B-rep"):
        read_brep(path)


def test_write_brep_raises_when_it_cannot_write(tmp_path):
    with pytest.raises(OSError):
        write_brep(BRep("text"), tmp_path / "missing_directory" / "surface.brep")


@pytest.fixture(scope="module")
def small_tetrahedra(meniscus_solid):
    return tetrahedral_mesh(meniscus_solid, max_size=4.0)


def read_gmsh_mesh(path):
    with gmsh_session():
        gmsh.merge(str(path))
        _, coordinates, _ = gmsh.model.mesh.getNodes()
        coordinates = np.array(coordinates)
        counts = {
            gmsh.model.mesh.getElementProperties(element_type)[0]: len(gmsh.model.mesh.getElementsByType(element_type)[0])
            for element_type in gmsh.model.mesh.getElementTypes(3)
        }
    return coordinates.reshape(-1, 3), counts


def test_write_mesh_msh_round_trips_nodes_and_elements(tmp_path, small_tetrahedra):
    path = tmp_path / "mesh.msh"

    write_mesh(small_tetrahedra, path)
    nodes, counts = read_gmsh_mesh(path)

    assert np.allclose(nodes, small_tetrahedra.nodes)
    assert counts == {"Tetrahedron 4": len(small_tetrahedra.elements["Tetrahedron 4"])}


def test_write_mesh_keeps_high_order_elements(tmp_path, meniscus_hex_blocks):
    mesh = hexahedral_mesh(meniscus_hex_blocks, elements_per_edge=2, elements_along=6, element_order=2, second_order_incomplete=True)
    path = tmp_path / "mesh.msh"

    write_mesh(mesh, path)

    assert read_gmsh_mesh(path)[1] == {"Hexahedron 20": len(mesh.elements["Hexahedron 20"])}


@pytest.mark.parametrize("extension", [".vtk", ".inp", ".unv", ".bdf", ".mesh", ".su2"])
def test_write_mesh_infers_format_from_extension(tmp_path, small_tetrahedra, extension):
    path = tmp_path / f"mesh{extension}"

    write_mesh(small_tetrahedra, path)

    assert path.stat().st_size > 0


@pytest.mark.parametrize("filename", ["mesh.bogus", "mesh"])
def test_write_mesh_rejects_unknown_format(tmp_path, small_tetrahedra, filename):
    with pytest.raises(ValueError, match="cannot write meshes"):
        write_mesh(small_tetrahedra, tmp_path / filename)


def test_write_mesh_raises_when_it_cannot_write(tmp_path, small_tetrahedra):
    with pytest.raises(OSError, match="Could not write mesh"):
        write_mesh(small_tetrahedra, tmp_path / "missing_directory" / "mesh.msh")


@pytest.mark.parametrize("filename", ["knee.nii", "knee.nrrd", "knee.mhd"])
def test_segmentation_round_trips_with_its_labels(tmp_path, meniscus_label, filename):
    segmentation = Segmentation(meniscus_label * 3, {"meniscus": 3})

    write_segmentation(segmentation, tmp_path / filename)
    result = read_segmentation(tmp_path / filename)

    assert json.loads((tmp_path / "knee_labels.json").read_text()) == {"meniscus": 3}
    assert result.labels == {"meniscus": 3}
    assert np.array_equal(sitk.GetArrayFromImage(result.image), sitk.GetArrayFromImage(segmentation.image))


def test_segmentation_rejects_unsupported_format_before_writing_labels(tmp_path, meniscus_label):
    with pytest.raises(ValueError):
        write_segmentation(Segmentation(meniscus_label, {"meniscus": 1}), tmp_path / "knee.nii.gz")

    assert list(tmp_path.iterdir()) == []


def test_reading_a_segmentation_needs_its_labels_file(tmp_path, meniscus_label):
    write_label_map(meniscus_label, tmp_path / "knee.nii")

    with pytest.raises(OSError):
        read_segmentation(tmp_path / "knee.nii")


def test_surface_round_trip_keeps_points_and_triangles(tmp_path, bone_surface):
    path = tmp_path / "bone.stl"

    write_surface(bone_surface, path)
    points, triangles = surface_arrays(read_surface(path))

    original_points, original_triangles = surface_arrays(bone_surface)
    assert len(points) == len(original_points)
    assert len(triangles) == len(original_triangles)
    # STL stores single precision.
    assert np.allclose(np.sort(points, axis=0), np.sort(original_points, axis=0), atol=1e-4)


@pytest.mark.parametrize("filename", ["bone.vtp", "bone.obj"])
def test_surface_rejects_unsupported_format(tmp_path, bone_surface, filename):
    with pytest.raises(ValueError, match="Unsupported file format"):
        write_surface(bone_surface, tmp_path / filename)
    with pytest.raises(ValueError, match="Unsupported file format"):
        read_surface(tmp_path / filename)


def test_read_surface_rejects_missing_or_empty_files(tmp_path):
    with pytest.raises(FileNotFoundError):
        read_surface(tmp_path / "missing.stl")
    (tmp_path / "empty.stl").write_text("solid empty\nendsolid empty\n")
    with pytest.raises(ValueError, match="Not a triangulated surface"):
        read_surface(tmp_path / "empty.stl")


def test_write_mesh_writes_surface_elements(tmp_path, bone_surface):
    mesh = surface_mesh(bone_surface, 3.0, quadrilaterals=True)
    path = tmp_path / "mesh.msh"

    write_mesh(mesh, path)

    with gmsh_session():
        gmsh.merge(str(path))
        assert len(gmsh.model.mesh.getElementTypes(3)) == 0
        assert len(gmsh.model.mesh.getElementsByType(3)[0]) == len(mesh.elements["Quadrilateral 4"])
