"""Reading and writing LabelMaps as .nii, .nrrd, or .mhd + .raw, segmentations, poly data as .vtp, B-reps as .brep, and bone surfaces as .stl, and writing meshes."""

import json
from os import PathLike
from pathlib import Path

import gmsh
import numpy as np
import SimpleITK as sitk
from loguru import logger
from vtkmodules.vtkCommonDataModel import vtkPolyData
from vtkmodules.vtkIOGeometry import vtkSTLReader, vtkSTLWriter
from vtkmodules.vtkIOXML import vtkXMLPolyDataReader, vtkXMLPolyDataWriter

from kneemesh.brep import BRep, gmsh_session
from kneemesh.segmentation import Segmentation
from kneemesh.tet_meshing import Mesh
from kneemesh.vol import LABEL_MAP_PIXEL_IDS

SUPPORTED_SUFFIXES = frozenset({".nii", ".nrrd", ".mhd"})
POLY_DATA_SUFFIXES = frozenset({".vtp"})
BREP_SUFFIXES = frozenset({".brep"})
SURFACE_SUFFIXES = frozenset({".stl"})


def read_label_map(path: str | PathLike[str]) -> sitk.Image:
    """Read a segmented LabelMap from disk.

    For a .mhd file, the .raw file it references must also exist.

    Raises:
        ValueError: If the file format is unsupported or the image is not
            sitkUInt8, sitkUInt16, or sitkInt16.
    """
    path = _validated_path(path)
    image = sitk.ReadImage(str(path))
    _require_label_map(image)
    logger.debug("Read {} LabelMap of size {} and spacing {} from {}", image.GetPixelIDTypeAsString(), image.GetSize(), image.GetSpacing(), path)
    return image


def write_label_map(image: sitk.Image, path: str | PathLike[str]) -> None:
    """Write a segmented LabelMap to disk, overwriting any existing file.

    A .mhd path also writes a .raw file alongside it.

    Raises:
        ValueError: If the file format is unsupported or the image is not
            sitkUInt8, sitkUInt16, or sitkInt16.
    """
    path = _validated_path(path)
    _require_label_map(image)
    sitk.WriteImage(image, str(path), useCompression=False)


def segmentation_labels_path(path: str | PathLike[str]) -> Path:
    """Return where write_segmentation puts the labels of a segmentation written to path: <stem>_labels.json beside it."""
    path = Path(path)
    return path.with_name(f"{path.stem}_labels.json")


def write_segmentation(segmentation: Segmentation, path: str | PathLike[str]) -> None:
    """Write a segmentation's LabelMap to path and its {name: value} labels as JSON beside it.

    The labels go to segmentation_labels_path(path), so the image can be read
    back with read_segmentation or given to a pipeline as label_map with those labels.

    Raises:
        ValueError: If the file format is unsupported.
        OSError: If a file cannot be written.
    """
    write_label_map(segmentation.image, path)
    labels_path = segmentation_labels_path(path)
    labels_path.write_text(json.dumps(segmentation.labels, indent=2) + "\n", encoding="utf-8")
    logger.debug("Wrote labels {} to {}", segmentation.labels, labels_path)


def read_segmentation(path: str | PathLike[str]) -> Segmentation:
    """Read a segmentation written by write_segmentation.

    Raises:
        ValueError: If the file format is unsupported, the image is not a
            LabelMap, or the labels are invalid for it.
        OSError: If the image or its labels file cannot be read.
    """
    labels = json.loads(segmentation_labels_path(path).read_text(encoding="utf-8"))
    return Segmentation(read_label_map(path), labels)


def read_poly_data(path: str | PathLike[str]) -> vtkPolyData:
    """Read poly data from a VTK XML .vtp file.

    Raises:
        ValueError: If the suffix is not .vtp or the file is not VTK XML poly data.
        FileNotFoundError: If the file does not exist.
    """
    path = _validated_path(path, POLY_DATA_SUFFIXES)
    if not path.is_file():
        raise FileNotFoundError(f"No such file: {path}")

    # The reader logs failures and returns empty output without any error state.
    reader = vtkXMLPolyDataReader()
    if not reader.CanReadFile(str(path)):
        raise ValueError(f"Not a VTK XML poly data file: {path}")
    reader.SetFileName(str(path))
    reader.Update()
    return reader.GetOutput()


def write_poly_data(poly_data: vtkPolyData, path: str | PathLike[str]) -> None:
    """Write poly data to a VTK XML .vtp file, overwriting any existing file.

    Raises:
        ValueError: If the suffix is not .vtp.
        OSError: If the file cannot be written.
    """
    path = _validated_path(path, POLY_DATA_SUFFIXES)

    writer = vtkXMLPolyDataWriter()
    writer.SetFileName(str(path))
    writer.SetInputData(poly_data)
    if not writer.Write():
        raise OSError(f"Could not write poly data to {path}")


def read_brep(path: str | PathLike[str]) -> BRep:
    """Read a B-rep from an Open Cascade .brep file.

    Raises:
        ValueError: If the suffix is not .brep or the file holds no valid shape.
        FileNotFoundError: If the file does not exist.
    """
    path = _validated_path(path, BREP_SUFFIXES)
    if not path.is_file():
        raise FileNotFoundError(f"No such file: {path}")

    with gmsh_session():
        try:
            shapes = gmsh.model.occ.importShapes(str(path))
        except Exception as error:  # gmsh raises only the bare Exception type.
            raise ValueError(f"Not a valid B-rep file: {path}") from error
    if not shapes:
        raise ValueError(f"Not a valid B-rep file: {path}")
    return BRep(path.read_text(encoding="utf-8"))


def write_brep(brep: BRep, path: str | PathLike[str]) -> None:
    """Write a B-rep to an Open Cascade .brep file, overwriting any existing file.

    Raises:
        ValueError: If the suffix is not .brep.
        OSError: If the file cannot be written.
    """
    path = _validated_path(path, BREP_SUFFIXES)
    path.write_text(brep.text, encoding="utf-8")
    logger.debug("Wrote B-rep to {}", path)


def read_surface(path: str | PathLike[str]) -> vtkPolyData:
    """Read a triangulated surface, such as a bone's, from a .stl file, merging coincident points.

    Raises:
        ValueError: If the suffix is not .stl or the file holds no triangles.
        FileNotFoundError: If the file does not exist.
    """
    path = _validated_path(path, SURFACE_SUFFIXES)
    if not path.is_file():
        raise FileNotFoundError(f"No such file: {path}")
    reader = vtkSTLReader()
    reader.SetFileName(str(path))
    reader.MergingOn()
    reader.Update()
    surface = reader.GetOutput()
    if not surface.GetNumberOfPolys():
        raise ValueError(f"Not a triangulated surface: {path}")
    return surface


def write_surface(surface: vtkPolyData, path: str | PathLike[str]) -> None:
    """Write a triangulated surface to a binary .stl file, overwriting any existing file.

    Raises:
        ValueError: If the suffix is not .stl.
        OSError: If the file cannot be written.
    """
    path = _validated_path(path, SURFACE_SUFFIXES)
    writer = vtkSTLWriter()
    writer.SetFileName(str(path))
    writer.SetInputData(surface)
    writer.SetFileTypeToBinary()
    if not writer.Write():
        raise OSError(f"Could not write the surface to {path}")
    logger.debug("Wrote surface to {}", path)


def write_mesh(mesh: Mesh, path: str | PathLike[str]) -> None:
    """Write a mesh in the format gmsh infers from the extension, overwriting any existing file.

    Formats include .msh, .vtk, .inp, .unv, .bdf, .med, .mesh, and .su2, and
    .stl for meshes of triangles.

    Raises:
        ValueError: If gmsh does not recognize the extension.
        OSError: If the file cannot be written.
    """
    path = Path(path)
    with gmsh_session():
        dimension = max((gmsh.model.mesh.getElementProperties(element_type)[1] for element_type in mesh.element_types.values()), default=3)
        entity = gmsh.model.addDiscreteEntity(dimension)
        gmsh.model.mesh.addNodes(dimension, entity, np.arange(1, len(mesh.nodes) + 1).tolist(), mesh.nodes.ravel().tolist())
        first_tag = 1
        for name, connectivity in mesh.elements.items():
            element_tags = np.arange(first_tag, first_tag + len(connectivity))
            gmsh.model.mesh.addElementsByType(entity, mesh.element_types[name], element_tags.tolist(), (connectivity + 1).ravel().tolist())
            first_tag += len(connectivity)
        try:
            gmsh.write(str(path))
        except Exception as error:  # gmsh raises only the bare Exception type.
            if "Unknown output file format" in str(error):
                raise ValueError(f"gmsh cannot write meshes in the format of {path.name!r}") from error
            raise OSError(f"Could not write mesh to {path}") from error
    logger.debug("Wrote mesh to {}", path)


def _validated_path(path: str | PathLike[str], suffixes: frozenset[str] = SUPPORTED_SUFFIXES) -> Path:
    path = Path(path)
    if path.suffix.lower() not in suffixes:
        supported = ", ".join(sorted(suffixes))
        raise ValueError(f"Unsupported file format {path.name!r}; expected one of: {supported}")
    return path


def _require_label_map(image: sitk.Image) -> None:
    if image.GetPixelID() not in LABEL_MAP_PIXEL_IDS:
        raise ValueError(f"LabelMap must be sitkUInt8, sitkUInt16, or sitkInt16, got {image.GetPixelIDTypeAsString()}")
