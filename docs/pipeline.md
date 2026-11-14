# Pipeline

For the meniscus, {py:func}`kneemesh.pipeline.run_brep` turns a LabelMap into a B-rep in these steps:

1. {py:func}`~kneemesh.geo.meniscus.centerline` fits a cubic B-spline through the centers of the label's cross-sections.
2. {py:func}`~kneemesh.vol.signed_distance_field` computes a signed distance field whose zero level lies on the label's voxel edges, then smooths it with a Gaussian.
3. {py:func}`~kneemesh.vol.extract_isocontour` extracts a closed triangle mesh at a level of that field.
4. {py:func}`~kneemesh.vol.smooth_isocontour` smooths the mesh with a windowed sinc filter.
5. {py:func}`~kneemesh.geo.meniscus.slice_isocontour` cuts the mesh with planes normal to the centerline.
6. {py:func}`~kneemesh.geo.meniscus.contours_to_brep` lofts the slices into a capped solid, or {py:func}`~kneemesh.geo.meniscus.contours_to_hex_blocks` builds the five-block butterfly solid for hexahedral meshing.

For the lateral and medial tibial and the patellar cartilage, each a sheet over its bone, it takes these steps:

1. {py:func}`~kneemesh.vol.signed_distance_field`, {py:func}`~kneemesh.vol.extract_isocontour`, and {py:func}`~kneemesh.vol.smooth_isocontour` extract the cartilage's smoothed surface as for the meniscus, and the bone's distance field too.
2. {py:func}`~kneemesh.cartilage.classify_sides` splits the surface into its articular side and the side against the bone, by the angle between each point's normal and the bone's distance gradient, dropping the steep rim between them.
3. {py:func}`~kneemesh.geo.sheet.fit_sheet` fits both sides as heights over the plane normal to the cartilage's thinnest direction, within a perimeter where the two sides overlap. Rolling a disk of radius `perimeter_radius` around the perimeter, inside and out, keeps it from curving more tightly than that radius. Regions thinner than `min_thickness` of the mean are thickened by lowering the bone side, and for tetrahedra, regions thinner than `hole_thickness`, or missing from the label, become holes.
4. {py:func}`~kneemesh.geo.sheet.sheet_to_brep` builds the solid: each side one B-spline surface over the domain, trimmed by the perimeter and any holes, and straight walls ruled between the sides' edges. The same solid serves tetrahedral and hexahedral meshing; `hex_blocks` only picks the defaults that suit hexahedra.

The femoral cartilage takes the same steps over a cylinder instead of a plane:

1. {py:func}`~kneemesh.geo.femur.fit_femoral_sheet` fits a {py:class}`~kneemesh.geo.sheet.CylinderFrame` to the side against the femur, along the direction its surface normals vary least, and fits both sides as distances from that axis over the cylinder unrolled. The unrolled domain is a U.
2. {py:func}`~kneemesh.geo.sheet.sheet_to_brep` builds its solid as for the other cartilage.

The bone under each cartilage must be in the segmentation. It tells the cartilage's sides apart, and its own surface is built too.

Both sides' surfaces take the domain coordinates as their parameters, so a point of one side and the point of the other at the same parameters lie on one straight column through the sheet, and the walls are made of such columns. {py:func}`kneemesh.mesh.sheet.hexahedral_mesh` relies on this to partition the sheet when it meshes it: it meshes one side with quadrilaterals by gmsh's Frontal-Delaunay algorithm, or its quasi-structured one, which follows a cross field aligned with the boundary, and runs each node's column to the other side, stacking hexahedra along the columns. {py:func}`kneemesh.mesh.sheet.sheet_partition_brep` builds the blocks the quadrilaterals' irregular nodes divide the sheet into, as a B-rep of one volume per block.

For the femur, tibia, and patella, {py:func}`kneemesh.pipeline.run_brep` writes a closed, triangulated surface rather than a B-rep:

1. {py:func}`~kneemesh.geo.bone.bone_mask` fills the label's cavities, and cuts shafts the scan cuts off ({py:func}`~kneemesh.geo.bone.find_shaft_ends`) shorter by `shaft_cut_fraction`, in the voxel map.
2. {py:func}`~kneemesh.vol.signed_distance_field`, {py:func}`~kneemesh.vol.extract_isocontour`, and {py:func}`~kneemesh.vol.smooth_isocontour` extract its smoothed surface, with non-manifold smoothing off.
3. {py:func}`~kneemesh.geo.bone.decimate_surface` reduces the surface to triangles about `triangle_size` across.

{py:func}`kneemesh.mesh.bone.surface_mesh` and {py:func}`kneemesh.mesh.bone.tetrahedral_mesh` load the surface into gmsh, which splits it into patches it can map to a plane and builds those maps, then mesh it with triangles or quadrilaterals, or the solid it encloses with tetrahedra graded from the cortical shell inward.

## Distance field smoothing

By default, the Gaussian's central 95% spans three voxel edges along each axis. A Gaussian's central 95% is $\pm 1.96\,\sigma$, so on an axis with voxel spacing $h$

$$
\sigma = \frac{3h}{2 \times 1.96} \approx 0.77\,h .
$$
