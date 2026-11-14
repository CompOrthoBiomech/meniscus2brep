import json
from pathlib import Path

import gmsh
import pytest
import SimpleITK as sitk
from loguru import logger

from kneemesh.brep import gmsh_session
from kneemesh.config import STRUCTURES, BrepMesh, LabelMapAssembly, LabelMapBrep, MeniscusBrep, MeniscusMesh
from kneemesh.disk import read_brep, read_segmentation, read_surface, write_brep, write_label_map
from kneemesh.pipeline import _BREP_BUILDERS, _MESHERS, assemble_main, brep_main, configure, mesh_main
from kneemesh.tet_meshing import Algorithm2D, Algorithm3D
from kneemesh.vol import signed_distance_field

LATERAL = '{"lateral_meniscus": 1}'


@pytest.fixture
def in_tmp_path(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    return tmp_path


def write_config(path, values):
    path.write_text(json.dumps(values))
    return path


def test_configs_have_defaults_for_every_field():
    assert MeniscusBrep().points_per_slice == 32
    assert MeniscusMesh().algorithm_2d is Algorithm2D.FRONTAL_DELAUNAY
    assert (MeniscusBrep().smoothing_iterations, MeniscusBrep().smoothing_pass_band) == (50, 0.001)
    assert (LabelMapBrep().output, BrepMesh().output, BrepMesh().element_type) == (".brep", "_{element_type}.msh", "tet4")


@pytest.mark.usefixtures("in_tmp_path")
def test_configure_overrides_section_options():
    argv = ["--meniscus.smoothing-iterations", "0", "--meniscus.smoothing-pass-band", "0.01"]

    config, _, _ = configure(LabelMapBrep, "test", argv)

    assert (config.meniscus.smoothing_iterations, config.meniscus.smoothing_pass_band) == (0, 0.01)


def test_mesh_config_saves_algorithms_by_name_and_reads_them_in_any_case():
    saved = json.loads(MeniscusMesh(algorithm_3d=Algorithm3D.HXT).model_dump_json())

    assert saved["algorithm_3d"] == "hxt"
    assert MeniscusMesh.model_validate({"algorithm_3d": "hxt"}).algorithm_3d is Algorithm3D.HXT
    assert MeniscusMesh.model_validate({"algorithm_3d": "HXT"}).algorithm_3d is Algorithm3D.HXT


def test_mesh_config_rejects_gmsh_numbers_and_unknown_algorithms():
    for value in (10, "rtree"):
        with pytest.raises(ValueError):
            MeniscusMesh.model_validate({"algorithm_3d": value})


@pytest.mark.parametrize("config_type", [LabelMapBrep, BrepMesh])
def test_configs_round_trip_through_json(tmp_path, config_type):
    path = tmp_path / "config.json"
    path.write_text(config_type().model_dump_json())

    assert config_type.from_json(path) == config_type()


def test_configure_without_config_saves_timestamped_settings(in_tmp_path):
    config, _, saved = configure(LabelMapBrep, "test", ["--meniscus.points-per-slice", "10", "--meniscus.cap", "planar", "--output", "_x.brep"])

    assert (config.meniscus.points_per_slice, config.meniscus.cap, config.output) == (10, "planar", "_x.brep")
    assert saved is not None
    assert saved.parent == type(saved)(".")
    assert saved.name.startswith("settings") and saved.suffix == ".json"
    assert LabelMapBrep.from_json(in_tmp_path / saved.name) == config


def test_configure_without_arguments_saves_defaults(in_tmp_path):
    config, _, saved = configure(BrepMesh, "test", [])

    assert config == BrepMesh()
    assert saved is not None
    assert BrepMesh.from_json(in_tmp_path / saved.name) == BrepMesh()


def test_configure_merges_section_overrides_into_the_config_file(tmp_path):
    values = {"labels": {"medial_meniscus": 2}, "meniscus": {"spacing": 2.5, "spline_order": 2}}
    path = write_config(tmp_path / "run.json", values)

    argv = ["--config", str(path), "--meniscus.spline-order", "4", "--meniscus.sigma", "0.5"]
    config, _, saved = configure(LabelMapBrep, "test", argv)

    assert (config.meniscus.spacing, config.meniscus.spline_order, config.meniscus.sigma) == (2.5, 4, 0.5)
    assert config.labels == {"medial_meniscus": 2}
    assert saved is not None
    assert saved == tmp_path / "run_modified.json"
    assert LabelMapBrep.from_json(saved) == config
    assert json.loads(path.read_text()) == values


def test_configure_with_config_and_no_overrides_saves_nothing(tmp_path):
    path = write_config(tmp_path / "run.json", {"element_type": "hex20"})

    config, _, saved = configure(BrepMesh, "test", ["--config", str(path)])

    assert config.element_type == "hex20"
    assert saved is None
    assert sorted(item.name for item in tmp_path.iterdir()) == ["run.json"]


def test_configure_reads_none_to_clear_optional_values(tmp_path):
    path = write_config(tmp_path / "run.json", {"meniscus": {"max_size": 2.0}})

    config, _, _ = configure(BrepMesh, "test", ["--config", str(path), "--meniscus.max-size", "none"])

    assert config.meniscus.max_size is None


@pytest.mark.parametrize(
    "argv",
    [
        ["--meniscus.points-per-slice", "many"],
        ["--meniscus.cap", "round"],
        ["--not-an-option", "1"],
        ["--points-per-slice", "10"],
        ["--meniscus.smoothing-iterations", "-1"],
        ["--meniscus.smoothing-pass-band", "0"],
        ["--labels", '{"meniscus": 1}'],
        ["meniscus"],
    ],
)
def test_configure_rejects_invalid_overrides(in_tmp_path, argv):
    with pytest.raises(SystemExit) as error:
        configure(LabelMapBrep, "test", argv)

    assert error.value.code == 2
    assert list(in_tmp_path.iterdir()) == []


@pytest.mark.parametrize("values", [{"spacingg": 2.0}, {"meniscus": {"spacingg": 2.0}}])
def test_configure_rejects_unknown_key_in_config_file(tmp_path, values):
    path = write_config(tmp_path / "run.json", values)

    with pytest.raises(SystemExit):
        configure(LabelMapBrep, "test", ["--config", str(path)])


def mesh_entity_counts(path):
    with gmsh_session():
        gmsh.merge(str(path))
        return {
            gmsh.model.mesh.getElementProperties(element_type)[0]: len(gmsh.model.mesh.getElementsByType(element_type)[0])
            for element_type in gmsh.model.mesh.getElementTypes(3)
        }


def test_pipelines_build_and_mesh_a_capped_solid(in_tmp_path, meniscus_label, capsys):
    write_label_map(meniscus_label, in_tmp_path / "label.nii")

    assert brep_main(["--label-map", "label.nii", "--labels", LATERAL, "--meniscus.cap", "smooth"]) == 0
    assert read_brep(in_tmp_path / "lateral_meniscus.brep").text
    breps = '{"lateral_meniscus": "lateral_meniscus.brep"}'
    assert mesh_main(["--breps", breps, "--element-type", "tet10", "--meniscus.max-size", "4"]) == 0

    assert list(mesh_entity_counts(in_tmp_path / "lateral_meniscus_tet10.msh")) == ["Tetrahedron 10"]
    assert "Wrote lateral_meniscus_tet10.msh" in capsys.readouterr().err
    assert len(list(in_tmp_path.glob("settings*.json"))) >= 1


def test_pipelines_build_and_mesh_hexahedral_blocks(in_tmp_path, meniscus_label):
    write_label_map(meniscus_label, in_tmp_path / "label.nii")
    brep_config = write_config(
        in_tmp_path / "brep.json",
        {
            "label_map": "label.nii",
            "labels": {"lateral_meniscus": 1},
            "output": "_blocks.brep",
            "meniscus": {"points_per_slice": 10, "hex_blocks": True},
        },
    )
    mesh_config = write_config(
        in_tmp_path / "mesh.json",
        {
            "breps": {"lateral_meniscus": "lateral_meniscus_blocks.brep"},
            "element_type": "hex8",
            "output": "_{element_type}.vtk",
            "output_directory": "out",
        },
    )

    assert brep_main(["--config", str(brep_config)]) == 0
    assert mesh_main(["--config", str(mesh_config), "--meniscus.elements-per-edge", "2", "--meniscus.elements-along", "10"]) == 0

    assert (in_tmp_path / "out" / "lateral_meniscus_hex8.vtk").stat().st_size > 0
    assert BrepMesh.from_json(in_tmp_path / "mesh_modified.json").meniscus.elements_per_edge == 2


def test_mesh_pipeline_makes_serendipity_hexahedra(in_tmp_path, meniscus_hex_blocks):
    write_brep(meniscus_hex_blocks, in_tmp_path / "blocks.brep")

    argv = [
        "--breps",
        '{"medial_meniscus": "blocks.brep"}',
        "--element-type",
        "hex20",
        "--meniscus.elements-per-edge",
        "2",
        "--meniscus.elements-along",
        "6",
    ]
    assert mesh_main(argv) == 0

    assert list(mesh_entity_counts(in_tmp_path / "medial_meniscus_hex20.msh")) == ["Hexahedron 20"]


@pytest.mark.usefixtures("in_tmp_path")
def test_pipeline_reports_errors_without_traceback(capsys):
    assert mesh_main(["--breps", '{"lateral_meniscus": "missing.brep"}']) == 1
    assert "No such file" in capsys.readouterr().err


def test_configure_returns_the_config_file(tmp_path):
    path = write_config(tmp_path / "run.json", {})

    assert configure(BrepMesh, "test", ["--config", str(path)]).source == path


@pytest.mark.usefixtures("in_tmp_path")
def test_configure_names_settings_with_the_given_timestamp():
    saved = configure(BrepMesh, "test", [], timestamp="20260102_030405").saved

    assert saved is not None and saved.name == "settings20260102_030405.json"


def test_package_logging_is_disabled_by_default(meniscus_label):
    messages = []
    handler = logger.add(messages.append, level="DEBUG")
    try:
        signed_distance_field(meniscus_label)
    finally:
        logger.remove(handler)

    assert messages == []


def test_pipeline_without_config_logs_to_timestamped_file_and_stderr(in_tmp_path, meniscus_label, capsys):
    write_label_map(meniscus_label, in_tmp_path / "label.nii")

    assert brep_main(["--label-map", "label.nii", "--labels", LATERAL]) == 0

    logs = list(in_tmp_path.glob("brep*.log"))
    settings = list(in_tmp_path.glob("settings*.json"))
    assert len(logs) == 1 and len(settings) == 1
    assert logs[0].stem.removeprefix("brep") == settings[0].stem.removeprefix("settings")
    log = logs[0].read_text()
    assert "INFO" in log and "Wrote lateral_meniscus.brep" in log
    assert "DEBUG" in log and "Centerline has" in log
    stderr = capsys.readouterr().err
    assert "Wrote lateral_meniscus.brep" in stderr
    assert "DEBUG" not in stderr


def test_pipeline_with_config_logs_to_file_named_for_it(in_tmp_path, meniscus_solid):
    write_brep(meniscus_solid, in_tmp_path / "solid.brep")
    config = write_config(in_tmp_path / "coarse.json", {"breps": {"medial_meniscus": "solid.brep"}, "meniscus": {"max_size": 4.0}})

    assert mesh_main(["--config", str(config)]) == 0

    assert "Meshed medial_meniscus" in (in_tmp_path / "coarse.log").read_text()
    assert (in_tmp_path / "medial_meniscus_tet4.msh").exists()
    assert list(in_tmp_path.glob("mesh*.log")) == []


def test_pipeline_logs_errors_to_file(in_tmp_path, capsys):
    assert mesh_main(["--breps", '{"lateral_meniscus": "missing.brep"}']) == 1

    assert "ERROR" in next(in_tmp_path.glob("mesh*.log")).read_text()
    assert "No such file" in capsys.readouterr().err


@pytest.mark.usefixtures("in_tmp_path")
def test_pipeline_disables_package_logging_when_done(meniscus_label):
    mesh_main(["--breps", '{"lateral_meniscus": "missing.brep"}'])
    messages = []
    handler = logger.add(messages.append, level="DEBUG")
    try:
        signed_distance_field(meniscus_label)
    finally:
        logger.remove(handler)

    assert messages == []


def test_every_structure_has_a_pipeline():
    assert set(_BREP_BUILDERS) == set(STRUCTURES)
    assert set(_MESHERS) == set(STRUCTURES)


def test_pipelines_build_and_mesh_a_bone(in_tmp_path, bone_label, capsys):
    write_label_map(bone_label, in_tmp_path / "bone.nii")
    labels = json.dumps({"femur": 1})

    assert brep_main(["--label-map", "bone.nii", "--labels", labels, "--bone.shaft-cut-fraction", "0.2", "--bone.output", "_surface.stl"]) == 0
    breps = json.dumps({"femur": "femur_surface.stl"})
    assert mesh_main(["--breps", breps, "--bone.element-type", "quad8", "--output", "_{element_type}.vtk"]) == 0
    assert mesh_main(["--breps", breps, "--element-type", "hex8", "--bone.element-type", "tet10", "--bone.max-size", "3"]) == 0

    assert read_surface(in_tmp_path / "femur_surface.stl").GetNumberOfPolys() > 0
    assert sorted(path.name for path in in_tmp_path.glob("femur_*.*sh")) == ["femur_tet10.msh"]
    assert (in_tmp_path / "femur_quad8.vtk").stat().st_size > 0
    stderr = capsys.readouterr().err
    assert "cutting off 0.2 of each shaft" in stderr
    assert "Meshing with tet10 elements of size 1.0 within 3.0 of the surface" in stderr


def test_pipeline_reads_named_labels_from_a_multilabel_image(in_tmp_path, meniscus_label):
    other = sitk.Cast(sitk.BinaryDilate(meniscus_label, [2, 2, 1]) - meniscus_label, sitk.sitkUInt8)
    write_label_map(meniscus_label * 7 + other * 2, in_tmp_path / "knee.nii")

    argv = ["--label-map", "knee.nii", "--labels", '{"lateral_meniscus": 7}', "--output-directory", "out", "--meniscus.spacing", "8"]
    assert brep_main(argv) == 0

    assert [path.name for path in (in_tmp_path / "out").iterdir()] == ["lateral_meniscus.brep"]


def test_pipeline_assembles_single_label_maps(in_tmp_path, meniscus_label):
    write_label_map(meniscus_label, in_tmp_path / "meniscus.nii")
    config = write_config(in_tmp_path / "assemble.json", {"label_maps": {"medial_meniscus": "meniscus.nii"}, "output": "_assembled.brep"})

    assert brep_main(["--config", str(config)]) == 0
    assert not list(in_tmp_path.glob("*_labels.json"))

    assert "Assembling medial_meniscus" in (in_tmp_path / "assemble.log").read_text()
    assert read_brep(in_tmp_path / "medial_meniscus_assembled.brep").text


@pytest.mark.usefixtures("in_tmp_path")
def test_pipeline_without_a_segmentation_reports_an_error(capsys):
    assert brep_main([]) == 1

    assert "label_map with labels or single-label label_maps" in capsys.readouterr().err


def test_pipeline_builds_every_meniscus_from_assembled_real_menisci(in_tmp_path, real_label, real_medial_label):
    write_label_map(real_label, in_tmp_path / "lateral.nii")
    write_label_map(real_medial_label, in_tmp_path / "medial.nii")
    label_maps = json.dumps({"lateral_meniscus": "lateral.nii", "medial_meniscus": "medial.nii"})

    argv = ["--label-maps", label_maps, "--assembled-label-map", "knee.nrrd", "--meniscus.spacing", "8"]
    assert brep_main(argv) == 0

    assembled = read_segmentation(in_tmp_path / "knee.nrrd")
    assert assembled.labels == {"lateral_meniscus": 1, "medial_meniscus": 2}
    assert json.loads((in_tmp_path / "knee_labels.json").read_text()) == assembled.labels
    assert read_brep(in_tmp_path / "lateral_meniscus.brep").text
    assert read_brep(in_tmp_path / "medial_meniscus.brep").text


def test_configure_reads_options_of_one_model(in_tmp_path):
    config, _, saved = configure(LabelMapAssembly, "test", ["--label-maps", '{"medial_meniscus": "m.nii"}', "--padding", "2"])

    assert config == LabelMapAssembly(label_maps={"medial_meniscus": Path("m.nii")}, padding=2)
    assert saved is not None
    assert LabelMapAssembly.from_json(in_tmp_path / saved.name) == config


def test_assembly_pipeline_writes_the_segmentation_and_its_labels(in_tmp_path, real_label, real_medial_label, capsys):
    write_label_map(real_label, in_tmp_path / "lateral.nii")
    write_label_map(real_medial_label, in_tmp_path / "medial.nii")
    label_maps = json.dumps({"lateral_meniscus": "lateral.nii", "medial_meniscus": "medial.nii"})

    assert assemble_main(["--label-maps", label_maps, "--interpolator", "nearest_neighbor", "--output", "knee.nii"]) == 0

    segmentation = read_segmentation(in_tmp_path / "knee.nii")
    assert segmentation.labels == {"lateral_meniscus": 1, "medial_meniscus": 2}
    assert segmentation.image.GetSpacing() == pytest.approx(real_label.GetSpacing())
    assert len(list(in_tmp_path.glob("assemble*.log"))) == 1
    assert "Wrote the assembled LabelMap to knee.nii" in capsys.readouterr().err


def test_brep_pipeline_reads_an_assembled_segmentation(in_tmp_path, meniscus_label):
    write_label_map(meniscus_label, in_tmp_path / "meniscus.nii")
    assert assemble_main(["--label-maps", '{"lateral_meniscus": "meniscus.nii"}', "--output", "knee.nrrd"]) == 0
    labels = (in_tmp_path / "knee_labels.json").read_text()

    assert brep_main(["--label-map", "knee.nrrd", "--labels", labels, "--output", "_solid.brep"]) == 0
    assert read_brep(in_tmp_path / "lateral_meniscus_solid.brep").text


@pytest.mark.usefixtures("in_tmp_path")
def test_assembly_pipeline_without_label_maps_reports_an_error(capsys):
    assert assemble_main([]) == 1

    assert "label_maps to assemble" in capsys.readouterr().err


def test_mesh_pipeline_optimizes_hexahedra(in_tmp_path, meniscus_hex_blocks, capsys):
    write_brep(meniscus_hex_blocks, in_tmp_path / "blocks.brep")
    argv = ["--breps", '{"lateral_meniscus": "blocks.brep"}', "--element-type", "hex8", "--output", ".vtk", "--meniscus.optimize", "true"]

    assert mesh_main([*argv, "--meniscus.elements-per-edge", "3", "--meniscus.elements-along", "12"]) == 0

    stderr = capsys.readouterr().err
    assert "Optimized minimum scaled Jacobian" in stderr
    assert (in_tmp_path / "lateral_meniscus.vtk").stat().st_size > 0


def test_pipeline_builds_and_meshes_cartilage_against_its_bone(in_tmp_path, cartilage_segmentation, capsys):
    write_label_map(cartilage_segmentation.image, in_tmp_path / "knee.nii")
    labels = json.dumps(cartilage_segmentation.labels)

    assert brep_main(["--label-map", "knee.nii", "--labels", labels, "--output", "_hex.brep", "--cartilage.hex-blocks", "true"]) == 0
    assert brep_main(["--label-map", "knee.nii", "--labels", labels, "--output", "_tet.brep"]) == 0
    hex_breps = json.dumps({"lateral_tibial_cartilage": "lateral_tibial_cartilage_hex.brep"})
    argv = ["--breps", hex_breps, "--element-type", "hex8", "--cartilage.element-size", "3", "--cartilage.optimize", "true"]
    argv += ["--cartilage.quad-algorithm", "quasi_structured", "--cartilage.save-partition", "true"]
    assert mesh_main(argv) == 0
    assert mesh_main(["--breps", json.dumps({"lateral_tibial_cartilage": "lateral_tibial_cartilage_tet.brep"})]) == 0

    stderr = capsys.readouterr().err
    assert "Optimized minimum scaled Jacobian" in stderr and "maximum Frobenius aspect ratio" in stderr
    assert sorted(path.name for path in in_tmp_path.glob("*.msh")) == ["lateral_tibial_cartilage_hex8.msh", "lateral_tibial_cartilage_tet4.msh"]
    assert (in_tmp_path / "lateral_tibial_cartilage_partition.brep").exists()
    # The bone is built as a surface too.
    assert not list(in_tmp_path.glob("tibia*.brep"))
    assert (in_tmp_path / "tibia.stl").stat().st_size > 0


def test_pipeline_builds_and_meshes_femoral_cartilage_against_the_femur(in_tmp_path, femoral_segmentation, capsys):
    write_label_map(femoral_segmentation.image, in_tmp_path / "knee.nii")
    labels = json.dumps(femoral_segmentation.labels)

    assert brep_main(["--label-map", "knee.nii", "--labels", labels, "--output", "_hex.brep", "--femoral-cartilage.hex-blocks", "true"]) == 0
    assert brep_main(["--label-map", "knee.nii", "--labels", labels, "--output", "_tet.brep"]) == 0
    hex_breps = json.dumps({"femoral_cartilage": "femoral_cartilage_hex.brep"})
    argv = ["--breps", hex_breps, "--element-type", "hex8", "--femoral-cartilage.element-size", "4", "--femoral-cartilage.optimize", "true"]
    assert mesh_main(argv) == 0
    assert mesh_main(["--breps", json.dumps({"femoral_cartilage": "femoral_cartilage_tet.brep"}), "--femoral-cartilage.element-size", "4"]) == 0

    stderr = capsys.readouterr().err
    assert "Optimized minimum scaled Jacobian" in stderr
    assert sorted(path.name for path in in_tmp_path.glob("*.msh")) == ["femoral_cartilage_hex8.msh", "femoral_cartilage_tet4.msh"]
