# Usage

`kneemesh-assemble` combines single-label LabelMaps into one multilabel LabelMap. `kneemesh-brep` builds a B-rep of every structure in a segmentation, and `kneemesh-mesh` meshes each structure's B-rep. Each takes a JSON configuration, `--option value` overrides, or both, and `--help` lists every option and its default. Every option is a field of the command's model, {py:class}`~kneemesh.config.LabelMapAssembly`, {py:class}`~kneemesh.config.LabelMapBrep`, or {py:class}`~kneemesh.config.BrepMesh`, with underscores written as dashes. Settings for one kind of structure sit in a section, such as `meniscus`, and are given on the command line as `--meniscus.points-per-slice 10`.

The structure names, keys of `labels`, `label_maps`, and `breps`, pick each structure's pipeline: `lateral_meniscus`, `medial_meniscus`, `patellar_cartilage`, `femoral_cartilage`, `lateral_tibial_cartilage`, `medial_tibial_cartilage`, `femur`, `tibia`, and `patella`. All are implemented. The bones' geometry is a triangulated surface rather than a B-rep; see [Bones](#bones).

Each structure's output is written to `<output-directory>/<structure><output>`, where `--output` is a suffix. B-rep suffixes end in `.brep`, `.brep` by default; bones' surfaces take `--bone.output`, `.stl` by default. Mesh suffixes default to `_{element_type}.msh`, with `{element_type}` replaced by `--element-type`: `hex8`, `hex20`, `hex27`, `tet4` (the default), or `tet10`, or for a bone by `--bone.element-type`. gmsh infers the mesh format from the extension.

```sh
kneemesh-brep --label-map meniscus.nii --labels '{"medial_meniscus": 1}' --meniscus.cap planar
kneemesh-brep --label-map knee.nii --labels '{"lateral_meniscus": 2, "medial_meniscus": 3}'
kneemesh-mesh --breps '{"medial_meniscus": "medial_meniscus.brep"}' --meniscus.max-size 1.0
```

The last command writes `medial_meniscus_tet4.msh`. Each run saves the configuration it used, so it can be repeated with `--config`. Without `--config` it is saved as `settings<timestamp>.json` in the working directory. With `--config` and overrides, the changed configuration is saved beside the original as `<name>_modified.json`. Section overrides change only the fields given, leaving the section's other values from `--config`.

For a hexahedral mesh, build the menisci's B-reps with `--meniscus.hex-blocks true --meniscus.points-per-slice 10`, and preferably the cartilage's with `--cartilage.hex-blocks true` (`--femoral-cartilage.hex-blocks true` for the femoral cartilage), and mesh them with `--element-type hex8`, `hex20`, or `hex27`. Add `--meniscus.optimize true` to move the nodes afterward for better scaled Jacobians with {py:func}`~kneemesh.mesh_optimization.optimize_hexahedral_mesh`. It first untangles any inverted elements. Boundary nodes may then leave the B-rep's surface by up to `--meniscus.surface-deviation` times the local element size, 0.1 by default, without changing the volume overall; `--meniscus.surface-deviation 0` keeps them on its tangent planes.

## Segmentations

When the structures are segmented in separate files, assemble them once with `kneemesh-assemble`, a pipeline configured by {py:class}`~kneemesh.config.LabelMapAssembly`:

```sh
kneemesh-assemble --label-maps '{"lateral_meniscus": "lateral.nii", "medial_meniscus": "medial.nii"}' --output knee.nrrd
```

It writes `knee.nrrd` and its labels in `knee_labels.json` beside it, logging to `assemble<timestamp>.log`. Structures get the values 1, 2, ... in the order given.

`kneemesh-brep` reads a segmentation and builds every structure named in it. Give the segmentation one of two ways:

- **A multilabel LabelMap** with `--label-map`, and `--labels` naming its values as JSON, such as `'{"lateral_meniscus": 2, "medial_meniscus": 3}'`. `--labels` is required, and values it does not name are ignored. An assembled LabelMap is read with `--label-map knee.nrrd --labels "$(cat knee_labels.json)"`.
- **Single-label LabelMaps** with `--label-maps`, assembled within the run as `kneemesh-assemble` does, one file per structure by name, such as `'{"lateral_meniscus": "lateral.nii", "medial_meniscus": "medial.nii"}'`. Every nonzero voxel of a file is its structure. {py:func}`~kneemesh.segmentation.assemble_label_maps` resamples them onto one grid, in the first file's orientation, with the smallest voxel spacing along each axis and a background margin of `--padding` voxels. `--interpolator` chooses `label_gaussian` (the default) or `nearest_neighbor`. Where structures overlap after resampling, the one listed later wins. `--assembled-label-map knee.nrrd` also writes the assembled LabelMap and `knee_labels.json`, as `kneemesh-assemble` does. {py:func}`~kneemesh.disk.read_segmentation` reads them in Python.

Every structure in the segmentation is built. A cartilage needs the bone beneath it in the segmentation too: `patella` for `patellar_cartilage`, `femur` for `femoral_cartilage`, and `tibia` for either tibial cartilage. The bone tells the cartilage's sides apart, and its surface is built as well.

```json
{
  "label_maps": {"lateral_meniscus": "lateral_meniscus.nii", "medial_meniscus": "medial_meniscus.nii"},
  "assembled_label_map": "knee.nrrd",
  "output": "_hex.brep",
  "meniscus": {"points_per_slice": 10, "hex_blocks": true}
}
```

The repository's `examples/` holds configurations for the real structures in `tests/dat`, run from the repository root. The `knee_*` ones handle all nine structures in one go:

```sh
kneemesh-assemble --config examples/knee_assemble.json   # writes examples/knee.nrrd
kneemesh-brep --config examples/knee_brep_hex.json       # or knee_brep_tet.json
kneemesh-mesh --config examples/knee_mesh_hex8.json      # or knee_mesh_tet4.json
```

## Cartilage

The tibial and patellar cartilage are built as sheets between their articular side and the side against their bone. Each B-rep is one solid: the two sides, and a straight wall around the perimeter, tilted so it meets the two sides at about equal angles, where a wall along the sheet's normal met the articular side, which curves down at the rim, at up to 130 degrees or more. Either element shape can mesh it, but the `cartilage` settings pick defaults by `hex_blocks`. For hexahedra, the perimeter curves no more tightly than a 3 mm radius, so elements can follow the real lateral tibial cartilage's notch, and regions thinner than 0.2 of the mean thickness are thickened by lowering the bone side. For tetrahedra, the perimeter follows the label more closely, with a 1 mm radius, regions thinner than 0.1 of the mean are thickened, and those thinner than 0.05 of the mean, or missing from the label, become holes. Thickened cartilage may overlap its bone.

```sh
kneemesh-brep --label-maps '{"tibia": "tibia.nii", "lateral_tibial_cartilage": "lateral.nii"}' --cartilage.hex-blocks true --output _hex.brep
kneemesh-mesh --breps '{"lateral_tibial_cartilage": "lateral_tibial_cartilage_hex.brep"}' --element-type hex8 --cartilage.optimize true
```

Hexahedral meshing partitions the sheet itself. It meshes one side with quadrilaterals about `--cartilage.element-size` across, 2 mm by default, and stacks `--cartilage.elements-through` hexahedra, 3 by default, along straight columns to the other side; the columns through the walls follow the wall. `--cartilage.quad-algorithm` picks gmsh's algorithm for the quadrilaterals: `frontal_delaunay`, the default, keeps their size closest to the target, and `quasi_structured` aligns them with a cross field, leaving few irregular nodes but sizes down to 0.3 of the target on curved sheets. If gmsh leaves triangles, the size is changed by a few percent until it does not. `--cartilage.save-partition true` also writes `<structure>_partition.brep`, the blocks the quadrilaterals' irregular nodes divide the sheet into, one volume each: about 10 to 30 with `quasi_structured`, and mostly single elements with `frontal_delaunay`.

Optimizing cartilage hexahedra also weighs their Frobenius aspect ratios, by `--cartilage.aspect-weight`, 0.5 by default: optimizing the scaled Jacobian alone squares up corners by stretching elements, to a largest aspect ratio of 8 on the real patellar cartilage against 2.3 at 0.5. Optimizing keeps each column of nodes through the walls straight. The thinnest stretches of the rim hold the worst elements; `--cartilage.perimeter-inset` cuts them off at the cost of area.

## Femoral cartilage

The femoral cartilage wraps the condyles, so it is built as a sheet over the cylinder its side against the femur wraps, unrolled into a U: the trochlea joining the two condyles around the intercondylar notch. Its `femoral_cartilage` settings extend the `cartilage` settings, with the same defaults by `hex_blocks`, and its solid and hexahedra are made the same way.

```sh
kneemesh-brep --label-maps '{"femur": "femur.nii", "femoral_cartilage": "femoral.nii"}' --femoral-cartilage.hex-blocks true --output _hex.brep
kneemesh-mesh --breps '{"femoral_cartilage": "femoral_cartilage_hex.brep"}' --element-type hex8 --femoral-cartilage.optimize true
```

Tetrahedra default to `--femoral-cartilage.element-size` too, since the default from the oriented bounding box, about 30 mm across for the wrapped cartilage, gives nearly flat tetrahedra.

## Bones

The femur, tibia, and patella are not meshed with hexahedra, and their geometry is not a B-rep. `kneemesh-brep` writes each bone's closed, triangulated surface as `.stl`:

1. {py:func}`~kneemesh.geo.bone.bone_mask` fills the label's cavities. Where the scan cuts a shaft off, as it does the femur's and tibia's, `--bone.shaft-cut-fraction` removes that fraction of the bone's length along the shaft from the cut end, square in the voxel map, to make the mesh smaller; 0, the default, keeps the shaft as scanned. A shaft is cut off where the label's last slice along an image axis is a whole cross-section, nearly the same as the slices 2 mm inside it.
2. The surface is the isocontour of the mask's smoothed distance field, as for the other structures, smoothed by a windowed sinc filter with non-manifold and feature-edge smoothing off and coordinates normalized. Smoothing rounds the flat end of a cut shaft.
3. {py:func}`~kneemesh.geo.bone.decimate_surface` reduces the triangles to `--bone.triangle-size`, 1 by default, which moves the real bones' surfaces by at most 0.02. gmsh took minutes to reparametrize the real femur's 340,000 triangles.

`kneemesh-mesh` meshes a bone given its `.stl` in `--breps`. gmsh splits the surface into patches it can map to a plane and meshes them ({py:mod}`kneemesh.mesh.bone`). `--bone.element-type` chooses the elements, separately from `--element-type`:

- `tri3` (the default) or `tri6` triangles, or `quad4`, `quad8`, or `quad9` quadrilaterals, on the surface, about `--bone.element-size` across, 2 by default. Quadrilateral meshes hold quadrilaterals only. The mesh is smoothed across the seams between gmsh's patches, and every node, second-order ones included, lies on the surface.
- `tet4` or `tet10` tetrahedra through the volume, `--bone.cortical-element-size` across, 1 by default, within `--bone.cortical-distance` of the surface, 3 by default, where the cortical bone lies. Deeper, in the cancellous bone, their edges grow by `--bone.cancellous-grad` per unit of depth, 0.5 by default, up to `--bone.max-size`, 4 by default; a smaller grad makes a smoother transition and more elements.

```sh
kneemesh-brep --label-maps '{"femur": "femur.nii"}' --bone.shaft-cut-fraction 0.3
kneemesh-mesh --breps '{"femur": "femur.stl"}' --bone.element-type quad4
kneemesh-mesh --breps '{"femur": "femur.stl"}' --bone.element-type tet10 --bone.cortical-distance 2
```
