"""Command-line pipelines for assembling segmentations, building B-reps of knee structures, and meshing them."""

import argparse
import json
import sys
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any, NamedTuple

import SimpleITK as sitk
from loguru import logger
from pydantic import ValidationError
from pydantic.fields import FieldInfo
from vtkmodules.vtkCommonDataModel import vtkPolyData

from kneemesh.brep import BRep
from kneemesh.cartilage import CartilageSides, classify_sides
from kneemesh.config import BONES, BaseConfig, BrepMesh, CartilageBrep, CartilageMesh, LabelMapAssembly, LabelMapBrep, Structure, StructureMesh
from kneemesh.disk import read_brep, read_label_map, read_surface, segmentation_labels_path, write_brep, write_mesh, write_segmentation, write_surface
from kneemesh.geo.bone import bone_mask, decimate_surface
from kneemesh.geo.femur import fit_femoral_sheet
from kneemesh.geo.meniscus import centerline, contours_to_brep, contours_to_hex_blocks, slice_isocontour
from kneemesh.geo.sheet import CartilageSheet, fit_sheet, sheet_to_brep
from kneemesh.mesh import bone as bone_mesh
from kneemesh.mesh import meniscus as meniscus_mesh
from kneemesh.mesh import sheet as sheet_mesh
from kneemesh.mesh_optimization import mesh_quality, optimize_hexahedral_mesh
from kneemesh.segmentation import Interpolator, Segmentation, assemble_label_maps
from kneemesh.tet_meshing import Mesh, tetrahedral_mesh
from kneemesh.vol import extract_isocontour, signed_distance_field, smooth_isocontour

type _BrepBuilder = Callable[[Segmentation, Structure, LabelMapBrep], BRep | vtkPolyData]
type _Mesher = Callable[[Path, BrepMesh, Structure], Mesh]


def run_brep(config: LabelMapBrep) -> dict[Structure, BRep | vtkPolyData]:
    """Build a B-rep of every structure in config's segmentation and write each to config.output_path(structure).

    Each structure's name picks its pipeline. A bone's geometry is its
    triangulated surface rather than a B-rep. Each output is written as soon
    as it is built.

    Returns:
        The B-rep of each structure, or the surface of each bone.

    Raises:
        ValueError: If config gives no segmentation, or it cannot be read or combined.
    """
    structures = config.structures
    if not structures:
        raise ValueError("Give a multilabel label_map with labels or single-label label_maps to read")
    segmentation = load_segmentation(config)
    breps: dict[Structure, BRep | vtkPolyData] = {}
    for structure in structures:
        geometry = _BREP_BUILDERS[structure](segmentation, structure, config)
        path = config.output_path(structure)
        path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(geometry, BRep):
            write_brep(geometry, path)
        else:
            write_surface(geometry, path)
        logger.info("Wrote {}", path)
        breps[structure] = geometry
    return breps


def run_mesh(config: BrepMesh) -> dict[Structure, Mesh]:
    """Mesh the B-rep of every structure in config.breps, or a bone's surface, and write each to config.output_path(structure).

    Each structure's name picks its pipeline, and each mesh is written as soon as it is made.

    Returns:
        The mesh of each structure.

    Raises:
        ValueError: If config gives no breps, or a file is not a B-rep or surface.
        OSError: If a file cannot be read.
    """
    if not config.breps:
        raise ValueError("Give the breps to mesh, by structure")
    meshes: dict[Structure, Mesh] = {}
    for structure, path in config.breps.items():
        logger.info("Reading {} for {}", path, structure)
        mesh = _MESHERS[structure](path, config, structure)
        counts = ", ".join(f"{len(connectivity)} {name}" for name, connectivity in mesh.elements.items())
        logger.info("Meshed {} with {} nodes and {}", structure, len(mesh.nodes), counts)
        output = config.output_path(structure)
        output.parent.mkdir(parents=True, exist_ok=True)
        write_mesh(mesh, output)
        logger.info("Wrote {}", output)
        meshes[structure] = mesh
    return meshes


def run_assembly(config: LabelMapAssembly) -> Segmentation:
    """Assemble config's single-label LabelMaps and write the result and its labels to config.output.

    Raises:
        ValueError: If config gives no label_maps, or they cannot be read or combined.
    """
    if not config.label_maps:
        raise ValueError("Give the single-label label_maps to assemble")
    return _assemble(config.label_maps, config.interpolator, config.padding, config.output)


def _assemble(label_maps: Mapping[Structure, Path], interpolator: Interpolator, padding: int, output: Path | None) -> Segmentation:
    logger.info("Assembling {} with the {} interpolator", ", ".join(label_maps), interpolator)
    images: dict[str, sitk.Image] = {}
    for name, path in label_maps.items():
        logger.info("Reading LabelMap {} for {}", path, name)
        images[name] = read_label_map(path)
    segmentation = assemble_label_maps(images, interpolator, padding)
    logger.info(
        "Assembled labels {} on a {} grid with spacing {}", segmentation.labels, segmentation.image.GetSize(), segmentation.image.GetSpacing()
    )
    if output is not None:
        write_segmentation(segmentation, output)
        logger.info("Wrote the assembled LabelMap to {} and its labels to {}", output, segmentation_labels_path(output))
    return segmentation


def load_segmentation(config: LabelMapBrep) -> Segmentation:
    """Read the multilabel LabelMap config names, or assemble its single-label LabelMaps.

    An assembled segmentation is also written to config.assembled_label_map when that is set.

    Raises:
        ValueError: If config gives neither label_map nor label_maps, or a
            LabelMap cannot be read or combined.
    """
    if config.label_maps is not None:
        return _assemble(config.label_maps, config.interpolator, config.padding, config.assembled_label_map)
    if config.label_map is None or config.labels is None:
        raise ValueError("Give a multilabel label_map with labels or single-label label_maps to read")
    logger.info("Reading LabelMap {}", config.label_map)
    labels: dict[str, int] = {name: value for name, value in config.labels.items()}
    return Segmentation(read_label_map(config.label_map), labels)


def _meniscus_brep(segmentation: Segmentation, structure: Structure, brep_config: LabelMapBrep) -> BRep:
    config = brep_config.meniscus
    label_map, label = segmentation.image, segmentation.value(structure)
    logger.info("Fitting the centerline of {} (label {})", structure, label)
    curve = centerline(label_map, label)
    logger.info("Computing the signed distance field")
    distance_field = signed_distance_field(label_map, label, config.sigma)
    logger.info("Extracting the surface at level {}", config.level)
    surface = extract_isocontour(distance_field, config.level)
    logger.info("Smoothing the surface with {} iterations and pass band {}", config.smoothing_iterations, config.smoothing_pass_band)
    surface = smooth_isocontour(surface, config.smoothing_iterations, config.smoothing_pass_band)
    logger.info("Slicing every {} along the centerline", config.spacing)
    slices = slice_isocontour(surface, curve, config.spacing, config.points_per_slice)
    if config.hex_blocks:
        logger.info("Building hexahedral blocks through {} slices", len(slices))
        return contours_to_hex_blocks(slices, config.spline_order)
    logger.info("Lofting through {} slices with {} caps", len(slices), config.cap)
    return contours_to_brep(slices, config.spline_order, config.cap, config.cap_offset)


def _cartilage_sides(segmentation: Segmentation, structure: Structure, config: CartilageBrep) -> CartilageSides:
    bone = BONES[structure]
    label_map, label, bone_label = segmentation.image, segmentation.value(structure), segmentation.value(bone)
    logger.info("Computing the signed distance fields of {} (label {}) and {} (label {})", structure, label, bone, bone_label)
    distance_field = signed_distance_field(label_map, label, config.sigma)
    bone_distance_field = signed_distance_field(label_map, bone_label, config.sigma)
    logger.info("Extracting the surface at level {}", config.level)
    surface = extract_isocontour(distance_field, config.level)
    logger.info("Smoothing the surface with {} iterations and pass band {}", config.smoothing_iterations, config.smoothing_pass_band)
    surface = smooth_isocontour(surface, config.smoothing_iterations, config.smoothing_pass_band)
    logger.info("Classifying the articular side and the side against the {} at cosine {}", bone, config.side_cosine)
    return classify_sides(surface, bone_distance_field, config.side_cosine)


def _log_sheet(sheet: CartilageSheet) -> None:
    logger.info(
        "Fitted a sheet of area {:.4g} and mean thickness {:.3g}, thickened to at least {:.3g}, with {} holes",
        sheet.area,
        sheet.mean_thickness,
        sheet.min_thickness,
        len(sheet.holes),
    )


def _cartilage_brep(segmentation: Segmentation, structure: Structure, brep_config: LabelMapBrep) -> BRep:
    config = brep_config.cartilage
    sides = _cartilage_sides(segmentation, structure, config)
    sheet = fit_sheet(sides, config.resolved_perimeter_radius, config.perimeter_inset, config.resolved_min_thickness, config.resolved_hole_thickness)
    _log_sheet(sheet)
    logger.info("Building the solid")
    return sheet_to_brep(sheet)


def _femoral_cartilage_brep(segmentation: Segmentation, structure: Structure, brep_config: LabelMapBrep) -> BRep:
    config = brep_config.femoral_cartilage
    sides = _cartilage_sides(segmentation, structure, config)
    logger.info("Fitting a sheet over the cylinder the femur's side wraps")
    sheet = fit_femoral_sheet(
        sides, config.resolved_perimeter_radius, config.perimeter_inset, config.resolved_min_thickness, config.resolved_hole_thickness
    )
    _log_sheet(sheet)
    logger.info("Building the solid")
    return sheet_to_brep(sheet)


def _bone_surface(segmentation: Segmentation, structure: Structure, brep_config: LabelMapBrep) -> vtkPolyData:
    config = brep_config.bone
    label = segmentation.value(structure)
    logger.info("Filling cavities of {} (label {}) and cutting off {} of each shaft", structure, label, config.shaft_cut_fraction)
    mask = bone_mask(segmentation.image, label, config.shaft_cut_fraction)
    logger.info("Computing the signed distance field")
    distance_field = signed_distance_field(mask, 1, config.sigma)
    logger.info("Extracting the surface at level {}", config.level)
    surface = extract_isocontour(distance_field, config.level)
    logger.info("Smoothing the surface with {} iterations and pass band {}", config.smoothing_iterations, config.smoothing_pass_band)
    surface = smooth_isocontour(surface, config.smoothing_iterations, config.smoothing_pass_band, non_manifold_smoothing=False)
    logger.info("Decimating the surface to triangles of size {}", config.triangle_size)
    return decimate_surface(surface, config.triangle_size)


def _bone_mesh(path: Path, mesh_config: BrepMesh, _structure: Structure) -> Mesh:
    config = mesh_config.bone
    surface = read_surface(path)
    if config.element_shape == "tetrahedral":
        logger.info(
            "Meshing with {} elements of size {} within {} of the surface, growing by {} per unit of depth up to {}",
            config.element_type,
            config.cortical_element_size,
            config.cortical_distance,
            config.cancellous_grad,
            config.max_size,
        )
        return bone_mesh.tetrahedral_mesh(
            surface,
            config.cortical_element_size,
            config.cortical_distance,
            config.cancellous_grad,
            config.max_size,
            config.element_order,
            config.algorithm_2d,
            config.algorithm_3d,
        )
    logger.info("Meshing the surface with {} elements of size {}", config.element_type, config.element_size)
    return bone_mesh.surface_mesh(
        surface,
        config.element_size,
        config.element_order,
        config.second_order_incomplete,
        config.element_shape == "quadrilateral",
        config.algorithm_2d,
    )


def _meniscus_mesh(path: Path, mesh_config: BrepMesh, _structure: Structure) -> Mesh:
    brep = read_brep(path)
    config = mesh_config.meniscus
    logger.info("Meshing with {} elements", mesh_config.element_type)
    if mesh_config.element_shape == "hexahedral":
        mesh = meniscus_mesh.hexahedral_mesh(
            brep, config.elements_per_edge, config.elements_along, mesh_config.element_order, mesh_config.second_order_incomplete
        )
        return _optimize(mesh, config) if config.optimize else mesh
    return _tetrahedral_mesh(brep, config, mesh_config)


def _cartilage_mesh(path: Path, mesh_config: BrepMesh, structure: Structure) -> Mesh:
    brep = read_brep(path)
    config = mesh_config.cartilage
    logger.info("Meshing with {} elements", mesh_config.element_type)
    if mesh_config.element_shape == "hexahedral":
        return _sheet_hexahedra(brep, config, mesh_config, structure)
    return _tetrahedral_mesh(brep, config, mesh_config)


def _femoral_cartilage_mesh(path: Path, mesh_config: BrepMesh, structure: Structure) -> Mesh:
    brep = read_brep(path)
    config = mesh_config.femoral_cartilage
    logger.info("Meshing with {} elements", mesh_config.element_type)
    if mesh_config.element_shape == "hexahedral":
        return _sheet_hexahedra(brep, config, mesh_config, structure)
    return _tetrahedral_mesh(brep, config, mesh_config, config.resolved_max_size)


def _sheet_hexahedra(brep: BRep, config: CartilageMesh, mesh_config: BrepMesh, structure: Structure) -> Mesh:
    logger.info(
        "Meshing a side with {} quadrilaterals of size {}, stacked {} through the thickness",
        config.quad_algorithm,
        config.element_size,
        config.elements_through,
    )
    mesh = sheet_mesh.hexahedral_mesh(
        brep,
        config.element_size,
        config.elements_through,
        mesh_config.element_order,
        mesh_config.second_order_incomplete,
        config.quad_algorithm,
    )
    if config.save_partition:
        path = mesh_config.partition_path(structure)
        logger.info("Building the partition of the sheet into blocks")
        path.parent.mkdir(parents=True, exist_ok=True)
        write_brep(sheet_mesh.sheet_partition_brep(brep, config.element_size, config.quad_algorithm), path)
        logger.info("Wrote {}", path)
    return _optimize(mesh, config) if config.optimize else mesh


def _tetrahedral_mesh(brep: BRep, config: StructureMesh, mesh_config: BrepMesh, max_size: float | None = None) -> Mesh:
    return tetrahedral_mesh(
        brep,
        config.min_size,
        config.max_size if max_size is None else max_size,
        config.elements_per_circle,
        config.algorithm_2d,
        config.algorithm_3d,
        mesh_config.element_order,
        mesh_config.second_order_incomplete,
    )


def _optimize(mesh: Mesh, config: StructureMesh) -> Mesh:
    before = mesh_quality(mesh)
    count = sum(len(connectivity) for connectivity in mesh.elements.values())
    logger.info(
        "Optimizing {} hexahedra with minimum scaled Jacobian {:.3f}, mean {:.3f}, and maximum Frobenius aspect ratio {:.3g}",
        count,
        before.minimum,
        before.mean,
        before.max_aspect,
    )
    optimized = optimize_hexahedral_mesh(
        mesh,
        config.laplacian_iterations,
        config.optimization_iterations,
        config.surface_deviation,
        config.surface_travel,
        config.feature_angle,
        config.preserve_volume,
        config.aspect_weight,
    )
    after = mesh_quality(optimized)
    logger.info(
        "Optimized minimum scaled Jacobian to {:.3f}, mean to {:.3f}, and maximum Frobenius aspect ratio to {:.3g}",
        after.minimum,
        after.mean,
        after.max_aspect,
    )
    return optimized


# The pipeline that builds each implemented structure's B-rep from a segmentation.
_BREP_BUILDERS: dict[Structure, _BrepBuilder] = {
    "lateral_meniscus": _meniscus_brep,
    "medial_meniscus": _meniscus_brep,
    "lateral_tibial_cartilage": _cartilage_brep,
    "medial_tibial_cartilage": _cartilage_brep,
    "patellar_cartilage": _cartilage_brep,
    "femoral_cartilage": _femoral_cartilage_brep,
    "femur": _bone_surface,
    "tibia": _bone_surface,
    "patella": _bone_surface,
}

# The pipeline that meshes each implemented structure's B-rep.
_MESHERS: dict[Structure, _Mesher] = {
    "lateral_meniscus": _meniscus_mesh,
    "medial_meniscus": _meniscus_mesh,
    "lateral_tibial_cartilage": _cartilage_mesh,
    "medial_tibial_cartilage": _cartilage_mesh,
    "patellar_cartilage": _cartilage_mesh,
    "femoral_cartilage": _femoral_cartilage_mesh,
    "femur": _bone_mesh,
    "tibia": _bone_mesh,
    "patella": _bone_mesh,
}


class Configuration[ConfigType: BaseConfig](NamedTuple):
    """A pipeline's configuration and the files it came from and was saved to."""

    config: ConfigType
    source: Path | None
    saved: Path | None


def configure[ConfigType: BaseConfig](
    config_type: type[ConfigType],
    description: str,
    argv: Sequence[str] | None = None,
    timestamp: str | None = None,
) -> Configuration[ConfigType]:
    """Build a configuration from command-line options, an optional JSON file, and overrides.

    Every field of the configuration is an option, such as --points-per-slice
    for points_per_slice. A field holding another configuration, a section
    such as meniscus, gives an option for each of its fields, such as
    --meniscus.points-per-slice. Values given on the command line override
    those in --config, field by field within sections. The configuration is
    saved so the run can be repeated: as settings<timestamp>.json in the
    working directory when there is no --config, or as
    <config stem>_modified.json beside the file when overrides change it. A
    config file used without overrides is not copied.

    Args:
        config_type: Configuration model of the command.
        description: Description for the command's help.
        argv: Command-line arguments, by default sys.argv[1:].
        timestamp: Timestamp for the settings file name, by default the current time.

    Returns:
        The configuration, the --config file if any, and the path the
        configuration was saved to, or None if it was not saved.
    """
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", type=Path, help="JSON configuration; options given here override its values")
    _add_config_options(parser, config_type)
    arguments = vars(parser.parse_args(argv))
    config_path: Path | None = arguments.pop("config")

    values = json.loads(config_path.read_text(encoding="utf-8")) if config_path is not None else {}
    try:
        config = config_type.model_validate(_merge_overrides(values, arguments))
    except ValidationError as error:
        parser.error(str(error))

    if config_path is None:
        saved = Path(f"settings{timestamp or _timestamp()}.json")
    elif arguments:
        saved = config_path.with_name(f"{config_path.stem}_modified.json")
    else:
        saved = None
    if saved is not None:
        saved.write_text(config.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return Configuration(config, config_path, saved)


def _section_type(annotation: object) -> type[BaseConfig] | None:
    """The configuration model a field holds, or None if it holds a plain value."""
    return annotation if isinstance(annotation, type) and issubclass(annotation, BaseConfig) else None


def _add_config_options(parser: argparse.ArgumentParser, config_type: type[BaseConfig]) -> None:
    """Add an option for every field of config_type to parser, with one argument group per section."""
    defaults = config_type().model_dump(mode="json")
    for name, field in config_type.model_fields.items():
        section = _section_type(field.annotation)
        if section is None:
            parser.add_argument(f"--{name.replace('_', '-')}", dest=name, default=argparse.SUPPRESS, help=_option_help(field, defaults[name]))
            continue
        group = parser.add_argument_group(f"{name} options", field.description)
        for option, option_field in section.model_fields.items():
            group.add_argument(
                f"--{name.replace('_', '-')}.{option.replace('_', '-')}",
                dest=f"{name}.{option}",
                metavar=option.upper(),
                default=argparse.SUPPRESS,
                help=_option_help(option_field, defaults[name][option]),
            )


def _option_help(field: FieldInfo, default: object) -> str:
    # argparse formats help with %, so a literal percent sign must be doubled.
    return f"{field.description} (default: {default})".replace("%", "%%")


def _merge_overrides(values: dict[str, Any], arguments: Mapping[str, object]) -> dict[str, Any]:
    """Layer command-line arguments over config file values, merging section.field arguments into their sections."""
    merged = dict(values)
    for key, value in arguments.items():
        section, _, name = key.rpartition(".")
        if not section:
            merged[key] = value
            continue
        current = merged.get(section)
        merged[section] = {**(current if isinstance(current, dict) else {}), name: value}
    return merged


def assemble_main(argv: Sequence[str] | None = None) -> int:
    """Run the LabelMap assembly pipeline from the command line."""
    timestamp = _timestamp()
    setup = configure(LabelMapAssembly, "Assemble single-label LabelMaps into one multilabel LabelMap.", argv, timestamp)
    return _run(lambda: run_assembly(setup.config), setup, Path(f"assemble{timestamp}.log"))


def brep_main(argv: Sequence[str] | None = None) -> int:
    """Run the B-rep pipeline from the command line."""
    timestamp = _timestamp()
    setup = configure(LabelMapBrep, "Build a B-rep of each knee structure in a segmented LabelMap.", argv, timestamp)
    return _run(lambda: run_brep(setup.config), setup, Path(f"brep{timestamp}.log"))


def mesh_main(argv: Sequence[str] | None = None) -> int:
    """Run the meshing pipeline from the command line."""
    timestamp = _timestamp()
    setup = configure(BrepMesh, "Mesh the B-rep of each knee structure.", argv, timestamp)
    return _run(lambda: run_mesh(setup.config), setup, Path(f"mesh{timestamp}.log"))


def _timestamp() -> str:
    return f"{datetime.now():%Y%m%d_%H%M%S}"


def _run[ConfigType: BaseConfig](step: Callable[[], object], setup: Configuration[ConfigType], default_log: Path) -> int:
    """Run a pipeline step, logging to stderr and to a file named for its configuration.

    The log file is <config stem>.log beside the --config file, or default_log
    in the working directory without one.
    """
    log_path = setup.source.with_suffix(".log") if setup.source is not None else default_log
    logger.remove()
    handlers = [
        logger.add(sys.stderr, level="INFO", format="<level>{level: <8}</level> {message}"),
        logger.add(log_path, level="DEBUG", format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} | {message}"),
    ]
    logger.enable("kneemesh")
    try:
        logger.info("Logging to {}", log_path)
        if setup.source is not None:
            logger.info("Read configuration from {}", setup.source)
        if setup.saved is not None:
            logger.info("Saved configuration to {}", setup.saved)
        logger.debug("Configuration: {}", setup.config.model_dump_json())
        try:
            step()
        except (ValueError, OSError) as error:
            logger.error("{}", error)
            return 1
        except Exception:
            logger.exception("Pipeline failed unexpectedly")
            raise
        return 0
    finally:
        logger.disable("kneemesh")
        for handler in handlers:
            logger.remove(handler)
