from pathlib import Path

import pydantic
import pytest

from kneemesh.config import (
    STRUCTURES,
    BaseConfig,
    BoneBrep,
    BoneMesh,
    BrepMesh,
    CartilageBrep,
    CartilageMesh,
    FemoralCartilageBrep,
    FemoralCartilageMesh,
    LabelMapBrep,
    MeniscusBrep,
    MeniscusMesh,
)


class ExampleConfig(BaseConfig):
    name: str
    label: int = 1


def test_from_json_parses_fields(tmp_path):
    path = tmp_path / "config.json"
    path.write_text('{"name": "medial", "label": 2}')

    assert ExampleConfig.from_json(path) == ExampleConfig(name="medial", label=2)


def test_from_json_rejects_unknown_field(tmp_path):
    path = tmp_path / "config.json"
    path.write_text('{"name": "medial", "lable": 2}')

    with pytest.raises(pydantic.ValidationError):
        ExampleConfig.from_json(path)


def test_from_json_rejects_malformed_json(tmp_path):
    path = tmp_path / "config.json"
    path.write_text('{"name": ')

    with pytest.raises(pydantic.ValidationError):
        ExampleConfig.from_json(path)


def test_labels_and_label_maps_parse_json_text():
    config = LabelMapBrep.model_validate({"label_map": "knee.nii", "labels": '{"tibia": 1, "medial_meniscus": 4}'})
    assembled = LabelMapBrep.model_validate({"label_maps": '{"medial_meniscus": "m.nii", "tibia": "t.nii"}'})

    assert config.labels == {"tibia": 1, "medial_meniscus": 4}
    assert assembled.label_maps == {"medial_meniscus": Path("m.nii"), "tibia": Path("t.nii")}
    assert assembled.structures == ("medial_meniscus", "tibia")
    assert LabelMapBrep.model_validate({"labels": "none"}).labels is None


def test_label_config_round_trips_through_json(tmp_path):
    config = LabelMapBrep(
        label_maps={"tibia": Path("t.nii"), "medial_tibial_cartilage": Path("m.nii")},
        interpolator="nearest_neighbor",
        meniscus=MeniscusBrep(points_per_slice=10),
    )
    path = tmp_path / "config.json"
    path.write_text(config.model_dump_json())

    assert LabelMapBrep.from_json(path) == config


@pytest.mark.parametrize(
    ("values", "message"),
    [
        ({"label_map": "knee.nii", "label_maps": {"medial_meniscus": "m.nii"}}, "not both"),
        ({"labels": {"medial_meniscus": 1}, "label_maps": {"medial_meniscus": "m.nii"}}, "named by their keys"),
        ({"label_map": "knee.nii"}, "Give labels"),
        ({"label_map": "knee.nii", "assembled_label_map": "out.nii", "labels": {"tibia": 1}}, "only when assembling"),
        ({"labels": {"meniscus": 1}}, "lateral_meniscus"),
        ({"label_maps": {"rim": "r.nii"}}, "lateral_meniscus"),
        ({"interpolator": "linear"}, "label_gaussian"),
        ({"padding": -1}, "greater than or equal"),
        ({"output": ".msh"}, "extension .brep"),
        ({"output": "/out.brep"}, "cannot hold a directory"),
    ],
)
def test_label_config_rejects_inconsistent_sources(values, message):
    with pytest.raises(pydantic.ValidationError, match=message):
        LabelMapBrep.model_validate(values)


def test_cartilage_needs_its_bone_labeled():
    assert LabelMapBrep(labels={"femur": 1, "femoral_cartilage": 2}).structures == ("femur", "femoral_cartilage")
    with pytest.raises(pydantic.ValidationError, match="needs its bone, femur"):
        LabelMapBrep(labels={"femoral_cartilage": 2})
    with pytest.raises(pydantic.ValidationError, match="needs its bone, tibia"):
        LabelMapBrep(label_maps={"lateral_tibial_cartilage": Path("l.nii")})


def test_brep_output_is_a_suffix_on_each_structure():
    config = LabelMapBrep(labels={"medial_meniscus": 1}, output="_hex.brep", output_directory=Path("out"))

    assert config.output_path("medial_meniscus") == Path("out/medial_meniscus_hex.brep")
    assert LabelMapBrep().output_path("lateral_meniscus") == Path("lateral_meniscus.brep")


@pytest.mark.parametrize(
    ("element_type", "shape", "order", "incomplete"),
    [
        ("hex8", "hexahedral", 1, False),
        ("hex20", "hexahedral", 2, True),
        ("hex27", "hexahedral", 2, False),
        ("tet4", "tetrahedral", 1, False),
        ("tet10", "tetrahedral", 2, False),
    ],
)
def test_mesh_output_defaults_to_the_element_type(element_type, shape, order, incomplete):
    config = BrepMesh(element_type=element_type)

    assert (config.element_shape, config.element_order, config.second_order_incomplete) == (shape, order, incomplete)
    assert config.output_path("lateral_meniscus") == Path(f"lateral_meniscus_{element_type}.msh")


def test_mesh_output_suffix_may_omit_or_reuse_the_element_type():
    assert BrepMesh(output=".vtk").output_path("medial_meniscus") == Path("medial_meniscus.vtk")
    assert BrepMesh(element_type="hex27", output="_{element_type}.inp").output_path("lateral_meniscus") == Path("lateral_meniscus_hex27.inp")


@pytest.mark.parametrize(("output", "message"), [("_{order}.msh", "only the {element_type}"), ("_mesh", "file extension"), ("_{", "placeholder")])
def test_mesh_config_rejects_bad_suffixes(output, message):
    with pytest.raises(pydantic.ValidationError, match=message):
        BrepMesh(output=output)


def test_mesh_config_restricts_brep_keys_and_element_types():
    assert BrepMesh.model_validate({"breps": '{"medial_meniscus": "m.brep"}'}).breps == {"medial_meniscus": Path("m.brep")}
    with pytest.raises(pydantic.ValidationError):
        BrepMesh.model_validate({"breps": {"meniscus": "m.brep"}})
    with pytest.raises(pydantic.ValidationError):
        BrepMesh.model_validate({"element_type": "hex64"})


def test_mesh_optimization_applies_only_to_hexahedra():
    assert BrepMesh(element_type="hex8", meniscus=MeniscusMesh(optimize=True)).meniscus.optimize
    with pytest.raises(pydantic.ValidationError, match="only to hexahedral"):
        BrepMesh(meniscus=MeniscusMesh(optimize=True))


def test_structures_list_every_name():
    assert STRUCTURES == (
        "lateral_meniscus",
        "medial_meniscus",
        "patellar_cartilage",
        "femoral_cartilage",
        "lateral_tibial_cartilage",
        "medial_tibial_cartilage",
        "femur",
        "tibia",
        "patella",
    )


def test_cartilage_defaults_depend_on_hex_blocks():
    tetrahedral, hexahedral = CartilageBrep(), CartilageBrep(hex_blocks=True)

    assert (tetrahedral.resolved_perimeter_radius, tetrahedral.resolved_min_thickness, tetrahedral.resolved_hole_thickness) == (1.0, 0.1, 0.05)
    assert (hexahedral.resolved_perimeter_radius, hexahedral.resolved_min_thickness, hexahedral.resolved_hole_thickness) == (3.0, 0.2, 0.0)
    assert CartilageBrep(hex_blocks=True, perimeter_radius=2.0, min_thickness=0.3).resolved_perimeter_radius == 2.0


def test_cartilage_holes_must_be_thinner_than_the_minimum():
    with pytest.raises(pydantic.ValidationError, match="must be below min_thickness"):
        CartilageBrep(min_thickness=0.1, hole_thickness=0.1)
    # Hexahedral blocks ignore hole_thickness.
    assert CartilageBrep(hex_blocks=True, min_thickness=0.1, hole_thickness=0.2).resolved_hole_thickness == 0.0


@pytest.mark.parametrize("values", [{"side_cosine": 1.0}, {"perimeter_radius": 0.0}, {"min_thickness": 1.0}, {"perimeter_inset": -1.0}])
def test_cartilage_config_rejects_out_of_range_values(values):
    with pytest.raises(pydantic.ValidationError):
        CartilageBrep.model_validate(values)


def test_cartilage_optimization_applies_only_to_hexahedra():
    assert BrepMesh(element_type="hex8", cartilage=CartilageMesh(optimize=True)).cartilage.optimize
    with pytest.raises(pydantic.ValidationError, match="only to hexahedral"):
        BrepMesh(cartilage=CartilageMesh(optimize=True))


def test_sections_share_surface_and_meshing_settings():
    assert set(MeniscusBrep.model_fields) & set(CartilageBrep.model_fields) == {
        "sigma",
        "level",
        "smoothing_iterations",
        "smoothing_pass_band",
        "hex_blocks",
    }
    assert {"min_size", "max_size", "optimize", "preserve_volume"} <= set(MeniscusMesh.model_fields) & set(CartilageMesh.model_fields)


def test_only_cartilage_optimizes_aspect_ratios_by_default():
    assert MeniscusMesh().aspect_weight == 0.0
    assert CartilageMesh().aspect_weight == 0.5
    with pytest.raises(pydantic.ValidationError):
        CartilageMesh(aspect_weight=-1.0)


def test_femoral_cartilage_sections_extend_the_cartilage_settings():
    brep = LabelMapBrep(labels={"femur": 1, "femoral_cartilage": 2}, femoral_cartilage=FemoralCartilageBrep(hex_blocks=True))

    assert brep.femoral_cartilage.resolved_perimeter_radius == 3.0
    # Its solid is built like the other sheets', so the block partition's ring width is gone.
    with pytest.raises(pydantic.ValidationError):
        FemoralCartilageBrep.model_validate({"ring_width": 0.5})


def test_femoral_tetrahedra_default_to_the_element_size():
    assert FemoralCartilageMesh().resolved_max_size == 2.0
    assert FemoralCartilageMesh(element_size=3.0).resolved_max_size == 3.0
    assert FemoralCartilageMesh(element_size=2.0, max_size=3.0).resolved_max_size == 3.0
    assert FemoralCartilageMesh().aspect_weight == 0.5
    with pytest.raises(pydantic.ValidationError, match="only to hexahedral"):
        BrepMesh(femoral_cartilage=FemoralCartilageMesh(optimize=True))


def test_cartilage_hexahedra_settings():
    config = CartilageMesh()

    assert (config.element_size, config.elements_through, config.quad_algorithm, config.save_partition) == (2.0, 3, "frontal_delaunay", False)
    assert CartilageMesh.model_validate({"quad_algorithm": "quasi_structured"}).quad_algorithm == "quasi_structured"
    with pytest.raises(pydantic.ValidationError):
        CartilageMesh.model_validate({"quad_algorithm": "butterfly"})
    with pytest.raises(pydantic.ValidationError):
        CartilageMesh.model_validate({"elements_around": 8})
    with pytest.raises(pydantic.ValidationError, match="only to hexahedral"):
        BrepMesh(cartilage=CartilageMesh(save_partition=True))
    assert BrepMesh(element_type="hex8").partition_path("patellar_cartilage") == Path("patellar_cartilage_partition.brep")


def test_bones_are_written_as_surfaces():
    config = LabelMapBrep(labels={"tibia": 1, "lateral_tibial_cartilage": 2}, output="_tet.brep", bone=BoneBrep(output="_bone.stl"))

    assert config.output_path("tibia") == Path("tibia_bone.stl")
    assert config.output_path("lateral_tibial_cartilage") == Path("lateral_tibial_cartilage_tet.brep")
    assert LabelMapBrep().output_path("patella") == Path("patella.stl")
    with pytest.raises(pydantic.ValidationError, match=".stl"):
        BoneBrep(output=".brep")


def test_bone_surface_settings():
    config = BoneBrep()

    assert (config.shaft_cut_fraction, config.triangle_size) == (0.0, 1.0)
    for values in ({"shaft_cut_fraction": 1.0}, {"shaft_cut_fraction": -0.1}, {"triangle_size": 0.0}):
        with pytest.raises(pydantic.ValidationError):
            BoneBrep.model_validate(values)


@pytest.mark.parametrize(
    ("element_type", "shape", "order", "incomplete"),
    [
        ("tri3", "triangle", 1, False),
        ("tri6", "triangle", 2, False),
        ("quad4", "quadrilateral", 1, False),
        ("quad8", "quadrilateral", 2, True),
        ("quad9", "quadrilateral", 2, False),
        ("tet4", "tetrahedral", 1, False),
        ("tet10", "tetrahedral", 2, False),
    ],
)
def test_bones_mesh_with_their_own_element_type(element_type, shape, order, incomplete):
    config = BrepMesh(element_type="hex8", bone=BoneMesh(element_type=element_type))

    assert (config.bone.element_shape, config.bone.element_order, config.bone.second_order_incomplete) == (shape, order, incomplete)
    assert config.output_path("femur") == Path(f"femur_{element_type}.msh")
    assert config.output_path("lateral_meniscus") == Path("lateral_meniscus_hex8.msh")


def test_bone_mesh_defaults_and_limits():
    config = BoneMesh()

    assert config.element_type == "tri3"
    assert (config.element_size, config.cortical_element_size, config.cortical_distance, config.cancellous_grad, config.max_size) == (
        2.0,
        1.0,
        3.0,
        0.5,
        4.0,
    )
    with pytest.raises(pydantic.ValidationError, match="must be at least cortical_element_size"):
        BoneMesh(cortical_element_size=2.0, max_size=1.0)
    for element_type in ("hex8", "tri10"):
        with pytest.raises(pydantic.ValidationError):
            BoneMesh.model_validate({"element_type": element_type})
