import numpy as np
import pytest
from vtkmodules.vtkFiltersCore import vtkImplicitPolyDataDistance, vtkMassProperties

from kneemesh.mesh.bone import surface_mesh, tetrahedral_mesh

# Corner pairs and middle node of each edge, in gmsh's node ordering.
EDGES = {
    "Triangle 6": [(0, 1, 3), (1, 2, 4), (2, 0, 5)],
    "Quadrilateral 8": [(0, 1, 4), (1, 2, 5), (2, 3, 6), (3, 0, 7)],
    "Quadrilateral 9": [(0, 1, 4), (1, 2, 5), (2, 3, 6), (3, 0, 7)],
    "Tetrahedron 10": [(0, 1, 4), (1, 2, 5), (2, 0, 6), (3, 0, 7), (3, 2, 8), (3, 1, 9)],
}


def distances(surface, points):
    distance = vtkImplicitPolyDataDistance()
    distance.SetInput(surface)
    return np.abs([distance.EvaluateFunction(*point) for point in points.tolist()])


def volume(surface):
    properties = vtkMassProperties()
    properties.SetInputData(surface)
    properties.Update()
    return properties.GetVolume()


def corner_quality(nodes, corners):
    """Each surface element's smallest corner scaled Jacobian about its mean normal."""
    after = nodes[np.roll(corners, -1, axis=1)] - nodes[corners]
    before = nodes[np.roll(corners, 1, axis=1)] - nodes[corners]
    normals = np.cross(after, before)
    mean = normals.sum(axis=1)
    mean /= np.linalg.norm(mean, axis=1, keepdims=True)
    jacobians = np.einsum("ijk,ik->ij", normals, mean) / (np.linalg.norm(after, axis=2) * np.linalg.norm(before, axis=2))
    return jacobians.min(axis=1)


def volume_ratios(nodes, corners):
    """Each tetrahedron's volume relative to a regular one's with its root mean square edge."""
    a, b, c, d = (nodes[corners[:, index]] for index in range(4))
    volumes = np.einsum("ij,ij->i", b - a, np.cross(c - a, d - a)) / 6
    edges = np.sqrt(sum(np.sum(edge**2, axis=1) for edge in (b - a, c - a, d - a, c - b, d - b, d - c)) / 6)
    return 6 * np.sqrt(2) * volumes / edges**3


def tetrahedra_volume(mesh):
    corners = mesh.nodes[mesh.elements["Tetrahedron 4"]]
    return float(np.abs(np.linalg.det(corners[:, 1:] - corners[:, :1])).sum() / 6)


def middle_offsets(mesh, name):
    """Each edge's middle node's distance from its ends' midpoint, relative to its length."""
    connectivity, nodes = mesh.elements[name], mesh.nodes
    return np.concatenate(
        [
            np.linalg.norm(nodes[connectivity[:, middle]] - (nodes[connectivity[:, first]] + nodes[connectivity[:, last]]) / 2, axis=1)
            / np.linalg.norm(nodes[connectivity[:, last]] - nodes[connectivity[:, first]], axis=1)
            for first, last, middle in EDGES[name]
        ]
    )


def test_surface_mesh_of_triangles_lies_on_the_surface(bone_surface):
    mesh = surface_mesh(bone_surface, 2.0)

    triangles = mesh.elements["Triangle 3"]
    edges = np.linalg.norm(mesh.nodes[np.roll(triangles, -1, axis=1)] - mesh.nodes[triangles], axis=2)
    assert list(mesh.elements) == ["Triangle 3"]
    assert len(np.unique(triangles)) == len(mesh.nodes)
    assert distances(bone_surface, mesh.nodes).max() < 1e-6
    assert np.median(edges) == pytest.approx(2.0, rel=0.2)
    assert corner_quality(mesh.nodes, triangles).min() > 0.5


def test_surface_mesh_of_quadrilaterals_only(bone_surface):
    mesh = surface_mesh(bone_surface, 2.0, quadrilaterals=True)

    assert list(mesh.elements) == ["Quadrilateral 4"]
    assert distances(bone_surface, mesh.nodes).max() < 1e-6
    assert corner_quality(mesh.nodes, mesh.elements["Quadrilateral 4"]).min() > 0.4


@pytest.mark.parametrize(
    ("quadrilaterals", "incomplete", "name", "nodes"),
    [(False, False, "Triangle 6", 6), (True, True, "Quadrilateral 8", 8), (True, False, "Quadrilateral 9", 9)],
)
def test_second_order_surface_mesh_puts_its_middle_nodes_on_the_surface(bone_surface, quadrilaterals, incomplete, name, nodes):
    mesh = surface_mesh(bone_surface, 2.0, 2, incomplete, quadrilaterals)

    assert list(mesh.elements) == [name]
    assert mesh.elements[name].shape[1] == nodes
    assert distances(bone_surface, mesh.nodes).max() < 1e-6
    assert middle_offsets(mesh, name).max() < 0.25


def test_tetrahedral_mesh_fills_the_bone(bone_surface):
    mesh = tetrahedral_mesh(bone_surface)

    tetrahedra = mesh.elements["Tetrahedron 4"]
    assert list(mesh.elements) == ["Tetrahedron 4"]
    assert tetrahedra_volume(mesh) == pytest.approx(volume(bone_surface), rel=0.01)
    assert volume_ratios(mesh.nodes, tetrahedra).min() > 0.2


def test_tetrahedra_are_finest_in_the_cortical_shell(bone_surface):
    mesh = tetrahedral_mesh(bone_surface, cortical_element_size=0.8, cortical_distance=1.5, cancellous_grad=1.0, max_size=3.0)

    corners = mesh.nodes[mesh.elements["Tetrahedron 4"]]
    edges = np.linalg.norm(corners[:, [1, 2, 3, 2, 3, 3]] - corners[:, [0, 0, 0, 1, 1, 2]], axis=2).mean(axis=1)
    depths = distances(bone_surface, corners.mean(axis=1))
    shell, core = edges[depths < 1.0], edges[depths > 4.0]
    assert np.median(shell) == pytest.approx(0.8, rel=0.35)
    assert np.median(core) > 2 * np.median(shell)


def test_a_gentler_grad_makes_more_tetrahedra(bone_surface):
    steep = tetrahedral_mesh(bone_surface, cancellous_grad=2.0)
    gentle = tetrahedral_mesh(bone_surface, cancellous_grad=0.25)

    assert len(gentle.elements["Tetrahedron 4"]) > len(steep.elements["Tetrahedron 4"])


def test_second_order_tetrahedra_put_their_boundary_middle_nodes_on_the_surface(bone_surface):
    mesh = tetrahedral_mesh(bone_surface, element_order=2)

    connectivity = mesh.elements["Tetrahedron 10"]
    assert connectivity.shape[1] == 10
    assert middle_offsets(mesh, "Tetrahedron 10").max() < 0.25
    assert volume_ratios(mesh.nodes, connectivity[:, :4]).min() > 0.2


def test_real_patella_meshes_with_good_elements(real_patella_surface):
    triangles = surface_mesh(real_patella_surface, 2.0)
    quadrilaterals = surface_mesh(real_patella_surface, 2.0, quadrilaterals=True)
    tetrahedra = tetrahedral_mesh(real_patella_surface)

    assert corner_quality(triangles.nodes, triangles.elements["Triangle 3"]).min() > 0.4
    assert corner_quality(quadrilaterals.nodes, quadrilaterals.elements["Quadrilateral 4"]).min() > 0.6
    assert volume_ratios(tetrahedra.nodes, tetrahedra.elements["Tetrahedron 4"]).min() > 0.25
    assert tetrahedra_volume(tetrahedra) == pytest.approx(volume(real_patella_surface), rel=0.01)


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        ({"element_size": 0.0}, "element_size"),
        ({"element_order": 0}, "element_order"),
        ({"element_order": 3}, "linear or quadratic"),
    ],
)
def test_surface_mesh_rejects_invalid_arguments(bone_surface, arguments, message):
    with pytest.raises(ValueError, match=message):
        surface_mesh(bone_surface, **arguments)


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        ({"cortical_element_size": 0.0}, "element_size"),
        ({"cortical_distance": -1.0}, "cortical_distance"),
        ({"cancellous_grad": 0.0}, "cancellous_grad"),
        ({"max_size": 0.5}, "max_size"),
    ],
)
def test_tetrahedral_mesh_rejects_invalid_arguments(bone_surface, arguments, message):
    with pytest.raises(ValueError, match=message):
        tetrahedral_mesh(bone_surface, **arguments)
