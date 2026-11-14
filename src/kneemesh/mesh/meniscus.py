"""Hexahedral meshing of the meniscus's five-block butterfly solid."""

import gmsh

from kneemesh.brep import BRep, gmsh_session
from kneemesh.tet_meshing import Mesh, gmsh_mesh, gmsh_options, import_solid, require_element_order


def hexahedral_mesh(
    brep: BRep,
    elements_per_edge: int = 4,
    elements_along: int = 20,
    element_order: int = 1,
    second_order_incomplete: bool = False,
) -> Mesh:
    """Mesh a solid of six-sided blocks with hexahedra, using transfinite surfaces and volumes.

    Every block must have six faces, such as the five blocks from
    contours_to_hex_blocks. Edges on the planar end caps get
    elements_per_edge elements, and the edges running between the caps get
    elements_along, so each block is a grid of elements_per_edge by
    elements_per_edge quadrilaterals swept elements_along times.

    Args:
        brep: A solid of six-sided blocks with planar caps.
        elements_per_edge: Elements along each edge of a cap.
        elements_along: Elements along the solid between the caps.
        element_order: Polynomial order of the elements. 2 gives 27-node hexahedra.
        second_order_incomplete: Omit face and interior nodes from high-order
            elements, giving 20-node instead of 27-node second-order hexahedra.

    Returns:
        The hexahedral mesh.

    Raises:
        ValueError: If a count is below 1, or brep is not a solid of six-sided
            blocks with planar caps.
    """
    if elements_per_edge < 1 or elements_along < 1:
        raise ValueError(f"Element counts must be at least 1, got {elements_per_edge} and {elements_along}")
    require_element_order(element_order)

    options: dict[str, float | bool] = {"Mesh.ElementOrder": element_order, "Mesh.SecondOrderIncomplete": second_order_incomplete}
    with gmsh_session(), gmsh_options(options):
        import_solid(brep)
        volumes = [tag for _, tag in gmsh.model.getEntities(3)]
        caps = [tag for _, tag in gmsh.model.getEntities(2) if gmsh.model.getType(2, tag) == "Plane"]
        six_sided = all(len(gmsh.model.getBoundary([(3, volume)], oriented=False)) == 6 for volume in volumes)
        if not six_sided or len(caps) != 2 * len(volumes):
            raise ValueError("Hexahedral meshing needs a solid of six-sided blocks with planar caps, such as contours_to_hex_blocks gives")

        cap_edges = {abs(tag) for _, tag in gmsh.model.getBoundary([(2, cap) for cap in caps], combined=False, oriented=False)}
        for _, curve in gmsh.model.getEntities(1):
            gmsh.model.mesh.setTransfiniteCurve(curve, (elements_per_edge if curve in cap_edges else elements_along) + 1)
        for _, face in gmsh.model.getEntities(2):
            gmsh.model.mesh.setTransfiniteSurface(face)
            gmsh.model.mesh.setRecombine(2, face)
        for volume in volumes:
            gmsh.model.mesh.setTransfiniteVolume(volume)
        gmsh.model.add_physical_group(3, volumes, name="solid")

        gmsh.model.mesh.generate(3)
        return gmsh_mesh()
