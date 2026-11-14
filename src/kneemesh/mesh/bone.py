"""Surface and tetrahedral meshing of the bones' triangulated surfaces (geo.bone), which gmsh reparametrizes.

gmsh splits the surface into patches until each can be mapped to a plane
(mesh.classifySurfaces with forReparametrization), then builds those maps
(mesh.createGeometry). It meshes the patches as it would any surface, and
the solid they enclose with tetrahedra. Bones are not meshed with hexahedra.
"""

import math

import gmsh
import numpy as np
from scipy.spatial import KDTree
from vtkmodules.vtkCommonDataModel import vtkPolyData
from vtkmodules.vtkFiltersCore import vtkImplicitPolyDataDistance

from kneemesh.brep import gmsh_session
from kneemesh.geo.bone import surface_arrays
from kneemesh.mesh.sheet import quadrangulate
from kneemesh.tet_meshing import Algorithm2D, Algorithm3D, Mesh, gmsh_mesh, gmsh_options, require_element_order

DEFAULT_BONE_ELEMENT_SIZE = 2.0
"""Default size of the elements of a bone's surface mesh, in physical units."""

DEFAULT_CORTICAL_ELEMENT_SIZE = 1.0
"""Default edge length of a bone's tetrahedra within cortical_distance of its surface, in physical units."""

DEFAULT_CORTICAL_DISTANCE = 3.0
"""Default depth of the shell of cortical_element_size tetrahedra below a bone's surface, in physical units."""

DEFAULT_CANCELLOUS_GRAD = 0.5
"""Default growth of a bone's tetrahedra's edge length per unit of depth beyond cortical_distance."""

DEFAULT_BONE_MAX_SIZE = 4.0
"""Default largest edge length of a bone's tetrahedra, in physical units."""

_TRIANGLE = 2
# Sweeps of Laplacian smoothing over the whole surface: on the real bones' quadrilaterals 3 raised the smallest
# scaled Jacobian from 0.04-0.44 to 0.48-0.65, and 10 to 0.57-0.74.
_SMOOTHING_SWEEPS = 10


def surface_mesh(
    surface: vtkPolyData,
    element_size: float = DEFAULT_BONE_ELEMENT_SIZE,
    element_order: int = 1,
    second_order_incomplete: bool = False,
    quadrilaterals: bool = False,
    algorithm_2d: Algorithm2D = Algorithm2D.FRONTAL_DELAUNAY,
) -> Mesh:
    """Mesh a bone's closed surface with triangles or quadrilaterals about element_size across.

    Quadrilateral meshes hold quadrilaterals only; see mesh.sheet.quadrangulate.
    The mesh is smoothed over the whole surface, across the seams between
    gmsh's patches, and every node lies on the surface.

    Args:
        surface: A closed triangulated surface, such as geo.bone gives.
        element_size: Target length of the elements' edges.
        element_order: Polynomial order of the elements, 1 or 2. 2 gives
            6-node triangles or 9-node quadrilaterals, with their new nodes on
            the surface.
        second_order_incomplete: Omit the middle node of second-order
            quadrilaterals, giving 8-node ones.
        quadrilaterals: Mesh with quadrilaterals instead of triangles, by
            Frontal-Delaunay with full-quad recombination. gmsh's
            quasi-structured algorithm took 50 s on the real patella and made
            elements a third of the size.
        algorithm_2d: gmsh's algorithm for triangles.

    Raises:
        ValueError: If an argument is out of range, or gmsh fails.
    """
    _check(element_size, element_order)
    with gmsh_session():
        locator = _reparametrize(surface)
        _mesh_surface(locator, element_size, quadrilaterals, algorithm_2d)
        _raise_order(locator, element_order, second_order_incomplete)
        return gmsh_mesh(2)


def tetrahedral_mesh(
    surface: vtkPolyData,
    cortical_element_size: float = DEFAULT_CORTICAL_ELEMENT_SIZE,
    cortical_distance: float = DEFAULT_CORTICAL_DISTANCE,
    cancellous_grad: float = DEFAULT_CANCELLOUS_GRAD,
    max_size: float = DEFAULT_BONE_MAX_SIZE,
    element_order: int = 1,
    algorithm_2d: Algorithm2D = Algorithm2D.FRONTAL_DELAUNAY,
    algorithm_3d: Algorithm3D = Algorithm3D.DELAUNAY,
) -> Mesh:
    """Mesh the solid a bone's closed surface encloses with tetrahedra, finest in its cortical shell.

    Elements are cortical_element_size within cortical_distance of the
    surface, the shell where the cortical bone lies, and grow deeper in by
    cancellous_grad per unit of distance, up to max_size:
    min(max_size, cortical_element_size + cancellous_grad * max(0, d - cortical_distance))
    at distance d from the surface.

    Args:
        surface: A closed triangulated surface, such as geo.bone gives.
        cortical_element_size: Edge length of the elements within cortical_distance of the surface, the surface's included.
        cortical_distance: Depth below the surface of the shell of cortical_element_size elements.
        cancellous_grad: Growth of the edge length per unit of depth beyond cortical_distance. Smaller grows more smoothly.
        max_size: Largest edge length, deep in the cancellous bone.
        element_order: Polynomial order of the elements, 1 or 2. 2 gives
            10-node tetrahedra, with their new boundary nodes on the surface.
        algorithm_2d: gmsh's algorithm for the boundary triangles.
        algorithm_3d: gmsh's algorithm for the tetrahedra.

    Raises:
        ValueError: If an argument is out of range, or gmsh fails or makes no tetrahedra.
    """
    _check(cortical_element_size, element_order)
    if cortical_distance < 0:
        raise ValueError(f"cortical_distance must not be negative, got {cortical_distance}")
    if cancellous_grad <= 0:
        raise ValueError(f"cancellous_grad must be positive, got {cancellous_grad}")
    if max_size < cortical_element_size:
        raise ValueError(f"max_size must be at least cortical_element_size, got {max_size} and {cortical_element_size}")
    # Distances to the surface's vertices, about a triangle apart, overestimate the distance to the surface by
    # far less than the shell is thick.
    vertices = KDTree(surface_arrays(surface)[0])

    def size(_dimension: int, _tag: int, x: float, y: float, z: float, _size: float) -> float:
        depth = float(vertices.query((x, y, z))[0])
        return min(max_size, cortical_element_size + cancellous_grad * max(0.0, depth - cortical_distance))

    with gmsh_session():
        locator = _reparametrize(surface)
        gmsh.model.geo.addVolume([gmsh.model.geo.addSurfaceLoop([tag for _, tag in gmsh.model.getEntities(2)])])
        gmsh.model.geo.synchronize()
        gmsh.model.mesh.setSizeCallback(size)
        sizes = {"Mesh.MeshSizeMin": cortical_element_size, "Mesh.MeshSizeMax": max_size, "Mesh.MeshSizeExtendFromBoundary": 0}
        with gmsh_options(sizes):
            _mesh_surface(locator, cortical_element_size, False, algorithm_2d)
            # Netgen's optimization removes the slivers the graded sizes leave: on the real femur it raised the
            # smallest volume ratio from 0.01 to 0.29 and the mean from 0.74 to 0.83.
            with gmsh_options({"Mesh.Algorithm3D": Algorithm3D(algorithm_3d).number, "Mesh.OptimizeNetgen": 1}):
                _generate(3)
        gmsh.model.mesh.removeSizeCallback()
        _raise_order(locator, element_order, False)
        mesh = gmsh_mesh(3)
    if not mesh.elements:
        raise ValueError("gmsh produced no tetrahedra in the bone")
    return mesh


def _mesh_surface(locator: vtkImplicitPolyDataDistance, element_size: float, quadrilaterals: bool, algorithm_2d: Algorithm2D) -> None:
    """Mesh the current model's surfaces with triangles or quadrilaterals only, then smooth the mesh over the whole surface."""
    sizes = {"Mesh.MeshSizeMin": element_size, "Mesh.MeshSizeMax": element_size, "Mesh.MeshSizeFromCurvature": 0, "Mesh.MeshSizeFromPoints": 0}
    with gmsh_options(sizes | {"Mesh.Algorithm": Algorithm2D(algorithm_2d).number}):
        if quadrilaterals:
            quadrangulate([tag for _, tag in gmsh.model.getEntities(2)], element_size, "frontal_delaunay", "the bone's surface")
        else:
            _generate(2)
    _smooth(locator)


def _raise_order(locator: vtkImplicitPolyDataDistance, element_order: int, second_order_incomplete: bool) -> None:
    if element_order == 2:
        with gmsh_options({"Mesh.SecondOrderIncomplete": second_order_incomplete, "Mesh.SecondOrderLinear": 1}):
            gmsh.model.mesh.setOrder(2)
        _project_nodes(locator)


def _check(element_size: float, element_order: int) -> None:
    if element_size <= 0:
        raise ValueError(f"element_size must be positive, got {element_size}")
    require_element_order(element_order)
    if element_order > 2:
        raise ValueError(f"Bone elements are linear or quadratic, got element_order {element_order}")


def _reparametrize(surface: vtkPolyData) -> vtkImplicitPolyDataDistance:
    """Load a triangulated surface into the current model as patches gmsh can mesh, and return a locator on it."""
    points, triangles = surface_arrays(surface)
    if not len(triangles):
        raise ValueError("The bone's surface has no triangles")
    entity = gmsh.model.addDiscreteEntity(2)
    gmsh.model.mesh.addNodes(2, entity, np.arange(1, len(points) + 1).tolist(), points.ravel().tolist())
    gmsh.model.mesh.addElementsByType(entity, _TRIANGLE, [], (triangles + 1).ravel().tolist())
    # The surface has no edges to keep, so it is split only where its patches need it to be mapped to a plane
    # (angle pi). Splitting at 40 degrees made patches of single sliver triangles that decimation had folded.
    gmsh.model.mesh.classifySurfaces(math.pi, True, True, math.pi)
    gmsh.model.mesh.createGeometry()
    locator = vtkImplicitPolyDataDistance()
    locator.SetInput(surface)
    return locator


def _smooth(locator: vtkImplicitPolyDataDistance) -> None:
    """Move every node of the current model's surface mesh to the mean of its neighbors, projected onto the surface, _SMOOTHING_SWEEPS times.

    gmsh smooths within each patch but keeps the nodes on the seams between
    patches, which left a quadrilateral of scaled Jacobian 0.04 there on the
    real femur. The seams are not edges of the bone, so the nodes on them move too.
    A gmsh compound of the patches cannot do this: a compound must map to a
    disk, and the closed surface has no boundary.
    """
    node_tags, coordinates, _ = gmsh.model.mesh.getNodes()
    index = np.zeros(int(np.max(node_tags)) + 1, dtype=np.int64)
    index[node_tags] = np.arange(len(node_tags))
    pairs = []
    for element_type in gmsh.model.mesh.getElementTypes(2):
        _, _, _, count, _, corners = gmsh.model.mesh.getElementProperties(element_type)
        _, element_nodes = gmsh.model.mesh.getElementsByType(element_type)
        elements = index[np.asarray(element_nodes)].reshape(-1, count)[:, :corners]
        pairs.extend(elements[:, [corner, (corner + 1) % corners]] for corner in range(corners))
    edges = np.unique(np.sort(np.concatenate(pairs), axis=1), axis=0)
    neighbors = np.bincount(edges.ravel(), minlength=len(node_tags))[:, None]
    positions = np.reshape(coordinates, (-1, 3))
    for _ in range(_SMOOTHING_SWEEPS):
        sums = np.zeros_like(positions)
        np.add.at(sums, edges[:, 0], positions[edges[:, 1]])
        np.add.at(sums, edges[:, 1], positions[edges[:, 0]])
        positions = _closest_points(locator, sums / np.maximum(neighbors, 1))
    for tag, position in zip(np.asarray(node_tags).tolist(), positions.tolist(), strict=True):
        gmsh.model.mesh.setNode(tag, position, [])


def _project_nodes(locator: vtkImplicitPolyDataDistance) -> None:
    """Move every node of the current model's mesh on a curve or surface onto the bone's surface.

    gmsh places the middle nodes of second-order elements through the
    patches' parametrizations, which put some of them half an edge off the
    surface, inverting elements, so they are made straight first
    (Mesh.SecondOrderLinear) and projected here.
    """
    for dimension in (1, 2):
        for _, tag in gmsh.model.getEntities(dimension):
            nodes, coordinates, _ = gmsh.model.mesh.getNodes(dimension, tag)
            for node, point in zip(np.asarray(nodes).tolist(), _closest_points(locator, np.reshape(coordinates, (-1, 3))).tolist(), strict=True):
                gmsh.model.mesh.setNode(node, point, [])


def _closest_points(locator: vtkImplicitPolyDataDistance, points: np.ndarray) -> np.ndarray:
    closest = [0.0, 0.0, 0.0]
    result = np.empty_like(points)
    for row, point in enumerate(points.tolist()):
        locator.EvaluateFunctionAndGetClosestPoint(point, closest)
        result[row] = closest
    return result


def _generate(dimension: int) -> None:
    try:
        gmsh.model.mesh.generate(dimension)
    except Exception as error:  # gmsh raises only the bare Exception type.
        raise ValueError(f"gmsh could not mesh the bone: {error}") from error
