"""Tetrahedral meshing of any solid B-rep with gmsh, and the Mesh it returns."""

import tempfile
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Self, override

import gmsh
import numpy as np
from loguru import logger

from kneemesh.brep import BRep, gmsh_session


class _GmshAlgorithm(StrEnum):
    """A gmsh meshing algorithm, valued by a readable name and carrying gmsh's option number.

    Lookup by value ignores case, so "HXT" and "hxt" both name the same member.
    """

    number: int

    def __new__(cls, name: str, number: int) -> Self:
        member = str.__new__(cls, name)
        member._value_ = name
        member.number = number
        return member

    @override
    @classmethod
    def _missing_(cls, value: object) -> Self | None:
        if isinstance(value, str):
            return next((member for member in cls if member.value == value.lower()), None)
        return None


class Algorithm2D(_GmshAlgorithm):
    """gmsh surface meshing algorithms that produce triangles (option Mesh.Algorithm).

    gmsh's "initial mesh only" algorithm crashes the process and BAMG fails on
    these shapes, so both are omitted.
    """

    MESH_ADAPT = "mesh_adapt", 1
    AUTOMATIC = "automatic", 2
    DELAUNAY = "delaunay", 5
    FRONTAL_DELAUNAY = "frontal_delaunay", 6


class Algorithm3D(_GmshAlgorithm):
    """gmsh volume meshing algorithms that produce tetrahedra (option Mesh.Algorithm3D).

    gmsh's R-tree algorithm is omitted because it writes a debug file into the
    working directory.
    """

    DELAUNAY = "delaunay", 1
    FRONTAL = "frontal", 4
    MMG3D = "mmg3d", 7
    HXT = "hxt", 10


DEFAULT_MIN_SIZE_FRACTION = 1 / 20
"""Default smallest element size as a fraction of the shortest side of the shape's oriented bounding box."""

DEFAULT_MAX_SIZE_FRACTION = 1 / 4
"""Default largest element size as a fraction of the shortest side of the shape's oriented bounding box."""

_CURVE_SAMPLES = 100
_SURFACE_SAMPLES = 30


@dataclass(frozen=True, eq=False)
class Mesh:
    """A mesh of volume elements, or of surface elements for a bone's surface.

    Attributes:
        nodes: (N, 3) physical node coordinates.
        elements: Connectivity keyed by gmsh element name, such as
            "Tetrahedron 4", "Hexahedron 27", or "Triangle 6". Each array is (M, nodes per
            element) of 0-based indices into nodes, in gmsh's node ordering.
        element_types: gmsh's element type number for each key of elements.
        straight_lines: (L, K) indices into nodes of chains of corner nodes
            that must stay straight, such as the columns through a cartilage
            sheet's walls. optimize_hexahedral_mesh keeps each chain's inner
            nodes on the segment between its ends. Empty for most meshes.
    """

    nodes: np.ndarray
    elements: dict[str, np.ndarray]
    element_types: dict[str, int]
    straight_lines: np.ndarray = field(default_factory=lambda: np.zeros((0, 2), dtype=np.int64))


def tetrahedral_mesh(
    brep: BRep,
    min_size: float | None = None,
    max_size: float | None = None,
    elements_per_circle: int = 0,
    algorithm_2d: Algorithm2D = Algorithm2D.FRONTAL_DELAUNAY,
    algorithm_3d: Algorithm3D = Algorithm3D.DELAUNAY,
    element_order: int = 1,
    second_order_incomplete: bool = False,
) -> Mesh:
    """Mesh a solid B-rep with tetrahedra.

    Args:
        brep: A solid, such as contours_to_brep with a cap.
        min_size: Smallest element size, in physical units. By default
            DEFAULT_MIN_SIZE_FRACTION of the shortest side of the shape's
            oriented bounding box, but no more than max_size, which keeps small
            geometric features and curvature sizing from producing
            excessively small elements. 0 sets no limit.
        max_size: Largest element size, in physical units. By default
            DEFAULT_MAX_SIZE_FRACTION of the shortest side of the shape's
            oriented bounding box, but no less than min_size. gmsh's own
            default, from the shape's overall extent, is too coarse for thin
            shapes and can leave them with no tetrahedra at all.
        elements_per_circle: Size elements so that a full circle of the local
            surface curvature would hold this many. 0 disables curvature sizing.
        algorithm_2d: Algorithm for the boundary triangles.
        algorithm_3d: Algorithm for the tetrahedra.
        element_order: Polynomial order of the elements. 2 gives 10-node tetrahedra.
        second_order_incomplete: Omit interior nodes from high-order elements.
            Second-order tetrahedra have none, so this only changes orders 3
            and up, such as 16-node instead of 20-node third-order tetrahedra.

    Returns:
        The tetrahedral mesh.

    Raises:
        ValueError: If a size or count is out of range, brep has no volume,
            or gmsh fails or produces no tetrahedra.
    """
    if max_size is not None and max_size <= 0:
        raise ValueError(f"max_size must be positive, got {max_size}")
    if min_size is not None and min_size < 0:
        raise ValueError(f"min_size must not be negative, got {min_size}")
    if min_size is not None and max_size is not None and max_size <= min_size:
        raise ValueError(f"max_size must exceed min_size, got {max_size} and {min_size}")
    if elements_per_circle < 0:
        raise ValueError(f"elements_per_circle must not be negative, got {elements_per_circle}")
    require_element_order(element_order)

    options: dict[str, float | bool] = {
        "Mesh.MeshSizeFromCurvature": elements_per_circle,
        "Mesh.Algorithm": Algorithm2D(algorithm_2d).number,
        "Mesh.Algorithm3D": Algorithm3D(algorithm_3d).number,
        "Mesh.ElementOrder": element_order,
        "Mesh.SecondOrderIncomplete": second_order_incomplete,
    }

    with gmsh_session():
        import_solid(brep)
        if min_size is None or max_size is None:
            shortest = float(oriented_extents()[0])
            if min_size is None:
                min_size = DEFAULT_MIN_SIZE_FRACTION * shortest if max_size is None else min(DEFAULT_MIN_SIZE_FRACTION * shortest, max_size)
            if max_size is None:
                max_size = max(DEFAULT_MAX_SIZE_FRACTION * shortest, min_size)
            logger.debug("Element sizes from {:.4g} to {:.4g}, with the oriented bounding box's shortest side {:.4g}", min_size, max_size, shortest)
        options["Mesh.MeshSizeMin"] = min_size
        options["Mesh.MeshSizeMax"] = max_size
        sizes = f"element sizes from {min_size:.4g} to {max_size:.4g}; try a smaller max_size"
        with gmsh_options(options):
            try:
                gmsh.model.mesh.generate(3)
            except Exception as error:
                # gmsh reports every failure as a bare Exception.
                raise ValueError(f"gmsh could not mesh the solid with {sizes}: {error}") from error
        mesh = gmsh_mesh()
    if not mesh.elements:
        raise ValueError(f"gmsh produced no tetrahedra with {sizes}")
    return mesh


def require_element_order(element_order: int) -> None:
    """Raise ValueError unless element_order is at least 1."""
    if element_order < 1:
        raise ValueError(f"element_order must be at least 1, got {element_order}")


@contextmanager
def gmsh_options(options: dict[str, float | bool]) -> Generator[None]:
    """Set gmsh numeric options for a block, restoring them afterward.

    gmsh options are global, so they would otherwise outlive the session when
    gmsh was already running.
    """
    previous = {name: gmsh.option.getNumber(name) for name in options}
    try:
        for name, value in options.items():
            gmsh.option.setNumber(name, float(value))
        yield
    finally:
        for name, value in previous.items():
            gmsh.option.setNumber(name, value)


def import_solid(brep: BRep) -> None:
    """Load a B-rep into the current gmsh model, raising ValueError if it has no volume."""
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "shape.brep"
        path.write_text(brep.text, encoding="utf-8")
        gmsh.model.occ.importShapes(str(path))
    gmsh.model.occ.synchronize()
    if not gmsh.model.getEntities(3):
        raise ValueError("Meshing needs a solid B-rep, such as contours_to_brep with a cap")


def oriented_extents() -> np.ndarray:
    """Return the side lengths of the current gmsh model's oriented bounding box, shortest first.

    The box's axes are the principal axes of points sampled on every curve
    and every non-planar surface. A planar face lies within the convex hull
    of its boundary, so its curves bound it along any axis, while sampling
    its parametric rectangle would also take points outside its trimmed region.
    """
    samples = []
    for dim, tag in gmsh.model.getEntities(1):
        (start,), (end,) = gmsh.model.getParametrizationBounds(dim, tag)
        samples.append(gmsh.model.getValue(dim, tag, np.linspace(start, end, _CURVE_SAMPLES).tolist()))
    for dim, tag in gmsh.model.getEntities(2):
        if gmsh.model.getType(dim, tag) == "Plane":
            continue
        (u_low, v_low), (u_high, v_high) = gmsh.model.getParametrizationBounds(dim, tag)
        u, v = np.meshgrid(np.linspace(u_low, u_high, _SURFACE_SAMPLES), np.linspace(v_low, v_high, _SURFACE_SAMPLES))
        samples.append(gmsh.model.getValue(dim, tag, np.column_stack((u.ravel(), v.ravel())).ravel().tolist()))
    points = np.concatenate([np.asarray(sample) for sample in samples]).reshape(-1, 3)
    offsets = points - points.mean(axis=0)
    axes = np.linalg.eigh(offsets.T @ offsets)[1]
    return np.sort(np.ptp(offsets @ axes, axis=0))


def gmsh_mesh(dimension: int = 3) -> Mesh:
    """Return the current gmsh model's elements of a dimension, 3 for volume elements or 2 for surface ones, and their nodes as a Mesh."""
    node_tags, coordinates, _ = gmsh.model.mesh.getNodes()
    coordinates = np.array(coordinates).reshape(-1, 3)
    node_tags = np.array(node_tags)

    index = np.zeros(int(node_tags.max()) + 1, dtype=np.int64)
    index[node_tags] = np.arange(len(node_tags))

    elements, element_types = {}, {}
    for element_type in gmsh.model.mesh.getElementTypes(dimension):
        name, _, _, nodes_per_element, *_ = gmsh.model.mesh.getElementProperties(element_type)
        _, element_nodes = gmsh.model.mesh.getElementsByType(element_type)
        elements[name] = index[element_nodes].reshape(-1, nodes_per_element)
        element_types[name] = int(element_type)
    # Keep only the nodes the elements use, such as a surface mesh's without the model's other nodes.
    used = np.unique(np.concatenate([connectivity.ravel() for connectivity in elements.values()])) if elements else np.zeros(0, np.int64)
    if len(used) < len(coordinates):
        renumber = np.zeros(len(coordinates), dtype=np.int64)
        renumber[used] = np.arange(len(used))
        elements = {name: renumber[connectivity] for name, connectivity in elements.items()}
        coordinates = coordinates[used]
    logger.debug("gmsh mesh has {} nodes and {}", len(coordinates), {name: len(connectivity) for name, connectivity in elements.items()})
    return Mesh(coordinates, elements, element_types)
