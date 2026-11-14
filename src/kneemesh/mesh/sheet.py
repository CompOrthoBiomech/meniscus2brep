"""Hexahedral meshing of cartilage sheets: a quadrilateral mesh of one side, extruded through the thickness.

A sheet's solid (geo.sheet.sheet_to_brep) has two sides that share one
parameterization and walls ruled straight between them. One side is meshed
with quadrilaterals, and each node's column runs straight to the other side:
from interior nodes to the other side's point at the same parameters, and
from nodes on a rim along the wall's straight line through them. The columns
are divided into layers, so the elements stack through the thickness.

The quadrilaterals come from gmsh, by default from its quasi-structured
algorithm, which aligns a cross field (a 4-RoSy field) with the boundary,
places the field's singularities where the shape needs them, and meshes
along it, leaving few irregular nodes. The regions those few nodes' straight
lines of edges bound are a partition of the sheet into four-sided blocks,
which sheet_partition_brep can build.
"""

from dataclasses import dataclass
from typing import Literal

import gmsh
import numpy as np
from loguru import logger

from kneemesh.brep import BRep, add_swept_surface, clamped_interpolation, export_brep, gmsh_session
from kneemesh.tet_meshing import Mesh, gmsh_options, import_solid, require_element_order

type QuadAlgorithm = Literal["quasi_structured", "frontal_delaunay"]
"""gmsh's algorithm for the side's quadrilaterals: quasi-structured along a cross field, or Frontal-Delaunay for quads with full-quad recombination."""

_GMSH_ALGORITHMS: dict[QuadAlgorithm, int] = {"quasi_structured": 11, "frontal_delaunay": 8}
# gmsh's recombination for each algorithm: the quasi-structured one recombines on its own, and full-quad
# recombination would subdivide its mesh, halving the size. For Frontal-Delaunay, "simple full-quad" left
# triangles least often and made the best quadrilaterals.
_RECOMBINATIONS = {11: 0, 8: 2}
_BLOSSOM = 1
# Relative changes of the element size tried in turn when a mesh keeps triangles.
_SIZE_RETRIES = (0.0, 0.02, -0.02, 0.04, -0.04, 0.06, -0.06)
# Laplacian smoothing sweeps of the Frontal-Delaunay quadrilaterals.
_SMOOTHING_SWEEPS = 5
_HEXAHEDRON_TYPES = {(1, False): 5, (2, False): 12, (2, True): 17}
_QUADRANGLE_TYPES = {1: 3, 2: 10}
_SPLINE_ORDER = 3


@dataclass(frozen=True)
class _Side:
    """The quadrilaterals of one side and the far end of each node's column."""

    near: np.ndarray  # (N, 3) node positions on the meshed side
    far: np.ndarray  # (N, 3) the far end of each node's column, on the other side
    quadrangles: np.ndarray  # (Q, 4 or 9) node indices in gmsh's ordering
    rim: np.ndarray  # (R,) indices of the corner nodes on the side's rims


def hexahedral_mesh(
    brep: BRep,
    element_size: float,
    elements_through: int = 3,
    element_order: int = 1,
    second_order_incomplete: bool = False,
    quad_algorithm: QuadAlgorithm = "quasi_structured",
) -> Mesh:
    """Mesh a cartilage sheet's solid with hexahedra about element_size across and elements_through thick.

    The larger side is meshed with quadrilaterals by quad_algorithm. If gmsh
    leaves triangles, which Frontal-Delaunay recombination occasionally does,
    the element size is changed by 2, 4, then 6% each way in turn, and as a
    last resort a mesh at twice the size is subdivided, each triangle into
    three quadrilaterals and each quadrilateral into four. Each node's column
    runs straight to the other side, and the columns through the walls are
    the mesh's straight_lines, so optimization keeps the walls straight.

    Args:
        brep: A sheet's solid, such as geo.sheet.sheet_to_brep gives.
        element_size: Target length of the elements' edges along the sheet.
        elements_through: Elements through the thickness.
        element_order: Polynomial order of the elements. 2 gives 27-node hexahedra.
        second_order_incomplete: Omit face and interior nodes from high-order
            elements, giving 20-node instead of 27-node second-order hexahedra.
        quad_algorithm: gmsh's algorithm for the side's quadrilaterals.

    Raises:
        ValueError: If element_size is not positive, elements_through is below
            1, element_order is out of range, brep is not a sheet's solid, or
            no quadrilateral mesh could be made.
    """
    if element_size <= 0:
        raise ValueError(f"element_size must be positive, got {element_size}")
    if elements_through < 1:
        raise ValueError(f"elements_through must be at least 1, got {elements_through}")
    require_element_order(element_order)
    if element_order > 2:
        raise ValueError(f"Sheet hexahedra are linear or quadratic, got element_order {element_order}")

    with gmsh_session():
        import_solid(brep)
        top, bottom = _sheet_sides()
        side = _quadrangulate(top, bottom, element_size, element_order, quad_algorithm)
    return _extrude(side, elements_through, element_order, second_order_incomplete)


def _sheet_sides() -> tuple[int, int]:
    """Return the current model's two sides, the larger first, after checking that it is a sheet's solid.

    Raises:
        ValueError: If the model is not one volume whose two largest faces each
            border only walls that join them.
    """
    error = "Hexahedral meshing needs a cartilage sheet's solid, such as sheet_to_brep gives"
    if len(gmsh.model.getEntities(3)) != 1:
        raise ValueError(f"{error}; got {len(gmsh.model.getEntities(3))} volumes")
    faces = sorted((tag for _, tag in gmsh.model.getEntities(2)), key=lambda face: -gmsh.model.occ.getMass(2, face))
    if len(faces) < 3:
        raise ValueError(f"{error}; got {len(faces)} faces")
    top, bottom = faces[:2]
    for rim in _rims(top):
        if not any(bottom in _faces_of(curve) for wall in _faces_of(rim) if wall != top for curve in _rims(wall)):
            raise ValueError(f"{error}; a wall from one side's rim does not reach the other side")
    return top, bottom


def _rims(face: int) -> list[int]:
    return sorted({abs(tag) for _, tag in gmsh.model.getBoundary([(2, face)], combined=False, oriented=False)})


def _faces_of(curve: int) -> list[int]:
    return [int(face) for face in gmsh.model.getAdjacencies(1, curve)[0]]


def _quadrangulate(top: int, bottom: int, size: float, order: int, algorithm: QuadAlgorithm) -> _Side:
    """Mesh the top side with quadrilaterals and find each node's column to the bottom side."""
    gmsh.model.setVisibility(gmsh.model.getEntities(), False)
    gmsh.model.setVisibility([(2, top)], True, recursive=True)
    with gmsh_options({"Mesh.MeshOnlyVisible": 1}):
        quadrangulate([top], size, algorithm, "the sheet's side")
    if order == 2:
        with gmsh_options({"Mesh.SecondOrderIncomplete": 0}):
            gmsh.model.mesh.setOrder(2)
    return _columns(top, bottom, order)


def quadrangulate(surfaces: list[int], size: float, algorithm: QuadAlgorithm, name: str = "the surface") -> None:
    """Mesh the current model's surfaces with quadrilaterals only, about size across.

    If gmsh leaves triangles on any of surfaces, which Frontal-Delaunay
    recombination occasionally does, the size is changed by 2, 4, then 6%
    each way in turn, and as a last resort a mesh at twice the size is
    subdivided, each triangle into three quadrilaterals and each
    quadrilateral into four. Also used for the bones' surfaces (mesh.bone).

    Args:
        surfaces: Tags of the surfaces that must hold quadrilaterals only.
        size: Target length of the edges.
        algorithm: gmsh's algorithm for the quadrilaterals.
        name: What the surfaces are, for messages.

    Raises:
        ValueError: If no mesh of quadrilaterals only could be made.
    """
    base = {"Mesh.RecombineAll": 1, "Mesh.MeshSizeFromCurvature": 0, "Mesh.MeshSizeExtendFromBoundary": 0, "Mesh.MeshSizeFromPoints": 0}
    attempts = [(_GMSH_ALGORITHMS[algorithm], size * (1 + change), False) for change in _SIZE_RETRIES]
    attempts.append((_GMSH_ALGORITHMS["frontal_delaunay"], 2 * size, True))
    for number, attempt_size, subdivide in attempts:
        # Subdividing a blossom-recombined mesh, as in gmsh's own full-quad recombination, makes quadrilaterals only.
        options = base | {"Mesh.Algorithm": number, "Mesh.RecombinationAlgorithm": _BLOSSOM if subdivide else _RECOMBINATIONS[number]}
        options |= {"Mesh.MeshSizeMin": attempt_size, "Mesh.MeshSizeMax": attempt_size}
        options["Mesh.SubdivisionAlgorithm"] = 1 if subdivide else 0
        with gmsh_options(options):
            gmsh.model.mesh.clear()
            gmsh.model.mesh.generate(2)
            if number == _GMSH_ALGORITHMS["frontal_delaunay"]:
                gmsh.model.mesh.optimize("Laplace2D", niter=_SMOOTHING_SWEEPS)
        triangles = sum(
            len(tags)
            for surface in surfaces
            for element_type, tags in zip(*gmsh.model.mesh.getElements(2, surface)[:2], strict=True)
            if element_type not in _QUADRANGLE_TYPES.values()
        )
        if not triangles:
            if attempt_size != size or number != _GMSH_ALGORITHMS[algorithm]:
                logger.warning("Meshed {} with quadrilaterals only at size {:.4g}{}", name, attempt_size, ", subdivided" if subdivide else "")
            return
        logger.debug("gmsh left {} triangles at size {:.4g}", triangles, attempt_size)
    raise ValueError(f"gmsh could not mesh {name} with quadrilaterals only")


def _columns(top: int, bottom: int, order: int) -> _Side:
    """Return the top side's mesh with each node's column end on the bottom side.

    Interior nodes pair with the bottom side's point at the same parameters.
    A rim node's column is the straight line of the wall through it, from
    the rim's parameter mapped onto the wall, so columns stay on the walls.
    """
    quadrangle_type = _QUADRANGLE_TYPES[order]
    element_tags, element_nodes = gmsh.model.mesh.getElementsByType(quadrangle_type, top)
    tags = np.asarray(element_nodes, dtype=np.int64).reshape(len(element_tags), -1)
    unique = np.unique(tags)
    index = {int(tag): position for position, tag in enumerate(unique)}
    near, far = np.zeros((len(unique), 3)), np.zeros((len(unique), 3))

    node_tags, coordinates, parameters = gmsh.model.mesh.getNodes(2, top, includeBoundary=False, returnParametricCoord=True)
    rows = [index[int(tag)] for tag in node_tags]
    near[rows] = np.reshape(coordinates, (-1, 3))
    far[rows] = np.reshape(gmsh.model.getValue(2, bottom, parameters), (-1, 3))

    rim_nodes: list[int] = []
    for rim in _rims(top):
        wall = next(face for face in _faces_of(rim) if face != top)
        node_tags, coordinates, parameters = gmsh.model.mesh.getNodes(1, rim, includeBoundary=True, returnParametricCoord=True)
        on_wall = np.reshape(gmsh.model.reparametrizeOnSurface(1, rim, parameters, wall), (-1, 2))
        low, high = gmsh.model.getParametrizationBounds(2, wall)
        ends = np.where(np.abs(on_wall[:, 1] - low[1]) < np.abs(on_wall[:, 1] - high[1]), high[1], low[1])
        rows = [index[int(tag)] for tag in node_tags]
        near[rows] = np.reshape(coordinates, (-1, 3))
        far[rows] = np.reshape(gmsh.model.getValue(2, wall, np.column_stack((on_wall[:, 0], ends)).ravel().tolist()), (-1, 3))
        rim_nodes.extend(rows)
    quadrangles = np.vectorize(index.__getitem__)(tags)
    corners = np.unique(quadrangles[:, :4])
    rim = np.intersect1d(np.unique(rim_nodes), corners)
    logger.debug("Meshed the sheet's side with {} quadrilaterals and {} nodes, {} on its rims", len(quadrangles), len(unique), len(rim))
    return _Side(near, far, quadrangles, rim)


def _extrude(side: _Side, through: int, order: int, incomplete: bool) -> Mesh:
    """Stack hexahedra through the thickness on a side's quadrilaterals, along their nodes' columns."""
    layers = through * order + 1
    fractions = np.linspace(0.0, 1.0, layers)[:, np.newaxis, np.newaxis]
    nodes = ((1 - fractions) * side.near + fractions * side.far).reshape(-1, 3)
    count = len(side.near)

    hexahedron_type = _HEXAHEDRON_TYPES[order, incomplete and order == 2]
    quadrangle_type = _QUADRANGLE_TYPES[order]
    with gmsh_session():
        name = gmsh.model.mesh.getElementProperties(hexahedron_type)[0]
        hexahedron_local = np.reshape(gmsh.model.mesh.getElementProperties(hexahedron_type)[4], (-1, 3))
        quadrangle_local = np.reshape(gmsh.model.mesh.getElementProperties(quadrangle_type)[4], (-1, 2))
    # Each hexahedron node is a quadrilateral node at the same (u, v), w steps of order up its element's first layer.
    matches = np.all(np.isclose(hexahedron_local[:, np.newaxis, :2], quadrangle_local[np.newaxis]), axis=-1)
    which = np.argmax(matches, axis=1)
    rise = np.rint((hexahedron_local[:, 2] + 1) / 2 * order).astype(np.int64)
    elements = np.vstack([(layer * order + rise) * count + side.quadrangles[:, which] for layer in range(through)])

    corners = nodes[elements[:, :8]]
    determinants = np.einsum("ij,ij->i", np.cross(corners[:, 1] - corners[:, 0], corners[:, 3] - corners[:, 0]), corners[:, 4] - corners[:, 0])
    if np.median(determinants) < 0:
        # The meshed side is the higher one, so the columns run down: reverse the layers.
        nodes = nodes.reshape(layers, count, 3)[::-1].reshape(-1, 3)
    lines = np.stack([side.rim + layer * count for layer in range(0, layers, order)], axis=1)

    used, elements = np.unique(elements, return_inverse=True)
    remap = np.full(len(nodes), -1, dtype=np.int64)
    remap[used] = np.arange(len(used))
    mesh = Mesh(nodes[used], {name: elements.reshape(-1, len(hexahedron_local))}, {name: hexahedron_type}, remap[lines])
    logger.debug("Extruded {} {} through {} layers", len(mesh.elements[name]), name, through)
    return mesh


def sheet_partition_brep(brep: BRep, element_size: float, quad_algorithm: QuadAlgorithm = "quasi_structured") -> BRep:
    """Build the partition of a sheet's solid into blocks that its hexahedral mesh's irregular nodes define, one volume per block.

    The side is meshed as hexahedral_mesh meshes it. The straight lines of
    edges through the quadrilaterals' irregular nodes, interior nodes with
    other than four quadrilaterals and rim nodes with other than two, run
    until they reach a rim or another irregular node. They divide the
    quadrilaterals into four-sided patches of regular grids, the mesh's base
    complex. Each patch becomes a block: a patch on each side through its
    nodes' columns' ends, and ruled walls along its columns. The blocks are
    fragmented so neighbors share faces, as for transfinite meshing.

    The quasi-structured algorithm leaves few irregular nodes and so few
    blocks: 10 on the real lateral tibial cartilage and 32 on the femoral at
    2 mm. Frontal-Delaunay quadrilaterals leave many, and most of their
    blocks would be single elements.

    Raises:
        ValueError: If brep is not a sheet's solid, a patch is not a regular
            grid, as around a hole no line crosses, or fragmenting the blocks fails.
    """
    with gmsh_session():
        import_solid(brep)
        top, bottom = _sheet_sides()
        side = _quadrangulate(top, bottom, element_size, 1, quad_algorithm)
    grids = [_right_handed(grid, side) for grid in _base_complex(side.quadrangles, side.rim)]
    logger.debug("The side's base complex has {} patches", len(grids))
    blocks = [(side.near[grid], side.far[grid]) for grid in grids]
    return _blocks_to_brep(blocks, grids, float(np.median(np.linalg.norm(side.far - side.near, axis=1))))


def _right_handed(grid: np.ndarray, side: _Side) -> np.ndarray:
    """Return a patch's grid, transposed if needed so its rows, columns, and columns toward the far side form a right-handed frame.

    Open Cascade makes a block with the other handedness a volume of negative size, which fragmenting then mishandles.
    """
    corner = side.near[grid[0, 0]]
    frame = np.array([side.near[grid[1, 0]] - corner, side.near[grid[0, 1]] - corner, side.far[grid[0, 0]] - corner])
    return grid if np.linalg.det(frame) > 0 else grid.T


# Fuzzy tolerances of fragmenting blocks, tried in turn, as fractions of the sheet's median thickness.
_FRAGMENT_TOLERANCES = (1e-3, 3e-2, 1e-1)


def _base_complex(quadrangles: np.ndarray, rim: np.ndarray) -> list[np.ndarray]:
    """Return the patches of a quadrilateral mesh's base complex as (rows, columns) grids of node indices.

    Raises:
        ValueError: If a patch is not a regular grid.
    """
    boundary = {int(node) for node in rim}
    edge_quads: dict[frozenset[int], list[int]] = {}
    for number, quad in enumerate(quadrangles[:, :4].tolist()):
        for a, b in zip(quad, quad[1:] + quad[:1], strict=True):
            edge_quads.setdefault(frozenset((a, b)), []).append(number)
    neighbors: dict[int, set[int]] = {}
    for edge in edge_quads:
        a, b = tuple(edge)
        neighbors.setdefault(a, set()).add(b)
        neighbors.setdefault(b, set()).add(a)
    # A regular interior node has four edges, and a regular rim node three.
    irregular = [node for node, around in neighbors.items() if len(around) != (3 if node in boundary else 4)]

    def onward(previous: int, node: int) -> int | None:
        """The edge leaving a regular interior node straight on: the one sharing no quadrilateral with the edge arriving."""
        if node in boundary or len(neighbors[node]) != 4:
            return None
        arriving = set(edge_quads[frozenset((previous, node))])
        return next(other for other in neighbors[node] if other != previous and not arriving & set(edge_quads[frozenset((node, other))]))

    separatrices: set[frozenset[int]] = set()
    for start in irregular:
        for first in neighbors[start]:
            if len(edge_quads[frozenset((start, first))]) == 1:
                continue
            previous, node = start, first
            separatrices.add(frozenset((previous, node)))
            while (following := onward(previous, node)) is not None:
                previous, node = node, following
                separatrices.add(frozenset((previous, node)))

    patch = np.full(len(quadrangles), -1)
    grids = []
    for seed in range(len(quadrangles)):
        if patch[seed] >= 0:
            continue
        members, stack = [seed], [seed]
        patch[seed] = len(grids)
        while stack:
            quad = quadrangles[stack.pop(), :4].tolist()
            for a, b in zip(quad, quad[1:] + quad[:1], strict=True):
                edge = frozenset((a, b))
                if edge in separatrices:
                    continue
                for other in edge_quads[edge]:
                    if patch[other] < 0:
                        patch[other] = len(grids)
                        members.append(other)
                        stack.append(other)
        grids.append(_patch_grid(quadrangles[members, :4]))
    return grids


def _patch_grid(quads: np.ndarray) -> np.ndarray:
    """Return the (rows, columns) grid of node indices of a patch of quadrilaterals forming a regular grid.

    Grid coordinates spread from the first quadrilateral across shared edges:
    the node beyond each end of a shared edge lies as far past it as the
    node before it.

    Raises:
        ValueError: If the quadrilaterals do not form one regular grid.
    """
    error = "The base complex has a patch that is not a regular grid of quadrilaterals"
    first = quads[0].tolist()
    coordinates = {first[0]: (0, 0), first[1]: (1, 0), first[2]: (1, 1), first[3]: (0, 1)}
    edge_quads: dict[frozenset[int], list[int]] = {}
    for number, quad in enumerate(quads.tolist()):
        for a, b in zip(quad, quad[1:] + quad[:1], strict=True):
            edge_quads.setdefault(frozenset((a, b)), []).append(number)

    def beside(quad: list[int], node: int, other: int) -> int:
        """The node of quad joined to node by an edge, other than other."""
        position = quad.index(node)
        return next(candidate for candidate in (quad[position - 1], quad[(position + 1) % 4]) if candidate != other)

    done, stack = {0}, [0]
    while stack:
        quad = quads[stack.pop()].tolist()
        for a, b in zip(quad, quad[1:] + quad[:1], strict=True):
            for number in edge_quads[frozenset((a, b))]:
                if number in done:
                    continue
                neighbor = quads[number].tolist()
                for node, other in ((a, b), (b, a)):
                    inside, outside = beside(quad, node, other), beside(neighbor, node, other)
                    expected = tuple(2 * np.array(coordinates[node]) - np.array(coordinates[inside]))
                    if coordinates.setdefault(outside, expected) != expected:
                        raise ValueError(error)
                done.add(number)
                stack.append(number)
    positions = np.array(list(coordinates.values()))
    positions -= positions.min(axis=0)
    rows, columns = (int(extent) + 1 for extent in positions.max(axis=0))
    grid = np.full((rows, columns), -1, dtype=np.int64)
    grid[positions[:, 0], positions[:, 1]] = list(coordinates)
    if (grid < 0).any() or len(set(coordinates.values())) != len(coordinates) or (rows - 1) * (columns - 1) != len(quads):
        raise ValueError(error)
    return grid


def _blocks_to_brep(blocks: list[tuple[np.ndarray, np.ndarray]], grids: list[np.ndarray], thickness: float) -> BRep:
    """Build one six-sided volume per block, from its (rows, columns, 3) points on each side, fragmented so neighbors share faces.

    Each block's sides are B-spline patches through its points, and a ruled
    wall stands on each of its grid's sides, built once for sides that
    neighboring blocks share; grids give the node indices that identify them.
    Fragmenting is fuzzy, because where the sheet is thin the sides meet the
    walls nearly tangentially, and exact fragmenting left slivers where three
    blocks meet on the real femoral cartilage's rim, while neighboring
    patches there can also overlap by about 0.05 mm. Each of
    _FRAGMENT_TOLERANCES, as fractions of the thickness, is tried in turn
    until every block comes out as one volume of six four-sided faces.

    Raises:
        ValueError: If at every tolerance fragmenting splits blocks or leaves one without six four-sided faces.
    """
    problem = ""
    for tolerance in _FRAGMENT_TOLERANCES:
        with gmsh_session():
            problem = _fragment_blocks(blocks, grids, tolerance * thickness)
            if not problem:
                return export_brep()
        logger.debug("Fragmenting the blocks at tolerance {:.3g} of the thickness failed: {}", tolerance, problem)
    raise ValueError(problem)


def _fragment_blocks(blocks: list[tuple[np.ndarray, np.ndarray]], grids: list[np.ndarray], tolerance: float) -> str:
    """Add the blocks to the current gmsh model and fragment them with a fuzzy tolerance, returning what is wrong with the result, or "" if nothing."""
    walls: dict[tuple[int, ...], int] = {}
    volumes = []
    for (near, far), grid in zip(blocks, grids, strict=True):
        faces = [_add_patch(near), _add_patch(far)]
        for select in (np.s_[0], np.s_[-1], np.s_[:, 0], np.s_[:, -1]):
            nodes = tuple(grid[select].tolist())
            key = min(nodes, nodes[::-1])
            if key not in walls:
                walls[key] = _add_wall(near[select], far[select])
            faces.append(walls[key])
        volumes.append((3, gmsh.model.occ.addVolume([gmsh.model.occ.addSurfaceLoop(faces, sewing=True)])))
    # Sewing copies the shared walls into each block. They are identical
    # surfaces, so fragmenting merges them back into one each.
    with gmsh_options({"Geometry.ToleranceBoolean": tolerance}):
        gmsh.model.occ.fragment(volumes[:1], volumes[1:])
    gmsh.model.occ.synchronize()
    if len(gmsh.model.getEntities(3)) != len(blocks):
        return f"The {len(blocks)} blocks overlap; fragmenting them gave {len(gmsh.model.getEntities(3))} volumes"
    for _, volume in gmsh.model.getEntities(3):
        faces = gmsh.model.getBoundary([(3, volume)], oriented=False)
        sides = [len(gmsh.model.getBoundary([face], combined=False)) for face in faces]
        if sides != [4] * 6:
            return f"Fragmenting the blocks left a volume with {len(faces)} faces of {sides} sides instead of six four-sided faces"
    return ""


def _add_patch(points: np.ndarray) -> int:
    """Add a B-spline surface through a (rows, columns, 3) grid of points, returning its tag.

    Rows and columns are interpolated at uniform parameters, so a patch's
    edge depends only on that edge's points and matches its neighbor's and
    its wall's exactly.
    """
    degree = min(_SPLINE_ORDER, points.shape[0] - 1, points.shape[1] - 1)
    parameters = np.linspace(0.0, 1.0, points.shape[1])
    rows = [clamped_interpolation(parameters, row, min(_SPLINE_ORDER, points.shape[1] - 1)) for row in points]
    return add_swept_surface(rows, min(_SPLINE_ORDER, points.shape[1] - 1), np.linspace(0.0, 1.0, points.shape[0]), degree)


def _add_wall(near: np.ndarray, far: np.ndarray) -> int:
    """Add a ruled surface of straight lines between two (K, 3) rows of points, returning its tag."""
    degree = min(_SPLINE_ORDER, len(near) - 1)
    parameters = np.linspace(0.0, 1.0, len(near))
    rows = [clamped_interpolation(parameters, points, degree) for points in (near, far)]
    return add_swept_surface(rows, degree, np.array([0.0, 1.0]), 1)
