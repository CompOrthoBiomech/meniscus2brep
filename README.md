# kneemesh

[![PyPI](https://img.shields.io/pypi/v/kneemesh)](https://pypi.org/project/kneemesh/)
[![Coverage](https://img.shields.io/endpoint?url=https://comporthobiomech.github.io/kneemesh/coverage.json)](https://github.com/CompOrthoBiomech/kneemesh/actions/workflows/release.yml)
[![Docs](https://img.shields.io/badge/docs-gh--pages-blue)](https://comporthobiomech.github.io/kneemesh/)

Generate B-rep geometry of knee structures from a segmentation LabelMap, and mesh it with high-quality elements. The menisci, the lateral and medial tibial, patellar, and femoral cartilage, and the femur, tibia, and patella are supported. For the meniscus:

- **B-rep from segmentation labels.** Reads a LabelMap (`.nii`, `.nrrd`, or `.mhd`), fits a centerline through the labeled meniscus, slices its smoothed surface into cross-section contours, and lofts them into a solid B-spline B-rep (`.brep`) with planar or smooth end caps.
- **Tetrahedral meshing** with linear 4-node or quadratic 10-node tetrahedra.
- **Hexahedral meshing** with block-structured 8-node, 20-node, or 27-node hexahedra. The solid is split into a butterfly (O-grid) of five blocks, so elements stay well shaped at the meniscus's sharp corners. Optional node optimization (`--meniscus.optimize true`) then raises the scaled Jacobians further, roughly doubling the worst element's on the real test menisci.

For the tibial, patellar, and femoral cartilage:

- **B-rep from segmentation labels.** Tells the cartilage's articular side from the side against its bone, which must be in the segmentation, and fits both as heights over a plane, or for the femoral cartilage a cylinder, within a smoothed perimeter. Thin regions are thickened on the bone side, and for tetrahedra, the thinnest regions and gaps in the label become holes.
- **Tetrahedral meshing** as for the meniscus.
- **Hexahedral meshing** of the same solid: one side is meshed with quadrilaterals of about `--cartilage.element-size`, and hexahedra stack along straight columns to the other side. Optional node optimization (`--cartilage.optimize true`) untangles any inverted elements and weighs the elements' Frobenius aspect ratios along with their scaled Jacobians, keeping the worst under about 3.5 on the real test cartilage.

For the femur, tibia, and patella:

- **Surface from segmentation labels.** Fills the label's cavities, optionally cuts the shafts the scan cuts off shorter (`--bone.shaft-cut-fraction`), and writes the smoothed, decimated surface as `.stl`; gmsh reparametrizes it for meshing.
- **Surface meshing** with 3- or 6-node triangles or 4-, 8-, or 9-node quadrilaterals (`--bone.element-type`, `tri3` by default).
- **Tetrahedral meshing** with 4- or 10-node tetrahedra, finest in the cortical shell (`--bone.cortical-element-size` within `--bone.cortical-distance` of the surface) and growing by `--bone.cancellous-grad` per unit of depth up to `--bone.max-size`.

Meshes are written in any format gmsh infers from the file extension, such as `.msh`, `.vtk`, or `.inp`.

**Documentation:** <https://comporthobiomech.github.io/kneemesh/>

## Installation

```sh
pip install kneemesh
```

## Usage

The command-line pipelines take a JSON configuration, `--option value` overrides, or both. `kneemesh-brep` builds every structure named in the segmentation, and `kneemesh-mesh` every structure named in `--breps`; the names, such as `lateral_meniscus`, pick each structure's pipeline. Outputs are written as `<structure><output>`, where `--output` is a suffix such as `_hex8.msh`. Run `kneemesh-brep --help` to list every option and its default.

```sh
kneemesh-assemble --label-maps '{"lateral_meniscus": "lateral.nii", "medial_meniscus": "medial.nii"}' --output knee.nrrd
kneemesh-brep --label-map knee.nrrd --labels "$(cat knee_labels.json)"
kneemesh-brep --label-map meniscus.nii --labels '{"medial_meniscus": 1}' --meniscus.cap planar
kneemesh-mesh --breps '{"medial_meniscus": "medial_meniscus.brep"}' --element-type tet10 --meniscus.max-size 1.0
```

A multilabel LabelMap takes `--labels` naming its values, such as `--labels '{"lateral_meniscus": 2, "medial_meniscus": 3}'`; values it does not name are ignored. `kneemesh-assemble` resamples separate single-label LabelMaps onto one grid and writes the multilabel LabelMap with its labels in `<stem>_labels.json`. `kneemesh-brep` can also take `--label-maps` directly and assemble them within the run. Example configurations are in `examples/`. See the documentation's usage page.

Each run saves the configuration it used, so it can be repeated with `--config`. Without `--config` it is saved as `settings<timestamp>.json` in the working directory. With `--config` and overrides, the changed configuration is saved beside the original as `<name>_modified.json`.

### Element types

| Elements | B-rep options | Mesh options |
|---|---|---|
| 4-node tetrahedra | | `--element-type tet4` (the default) |
| 10-node tetrahedra | | `--element-type tet10` |
| 8-node hexahedra | `--meniscus.hex-blocks true --meniscus.points-per-slice 10` | `--element-type hex8` |
| 20-node hexahedra | `--meniscus.hex-blocks true --meniscus.points-per-slice 10` | `--element-type hex20` |
| 27-node hexahedra | `--meniscus.hex-blocks true --meniscus.points-per-slice 10` | `--element-type hex27` |

For the cartilage, `--cartilage.hex-blocks true` builds the B-rep with the defaults that suit hexahedra; any cartilage B-rep can be meshed with either element shape.

## Development

Requires [uv](https://docs.astral.sh/uv/).

```sh
uv sync
uv run pytest
```

Build the documentation into `docs/_build/html`:

```sh
uv run --group docs sphinx-build -W -b html docs docs/_build/html
```

## License

Licensed under the [Apache License 2.0](LICENSE).
