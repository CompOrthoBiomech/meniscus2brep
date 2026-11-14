# Configuration

Each pipeline reads one of these models from JSON with {py:meth}`~kneemesh.config.BaseConfig.from_json`. Unknown keys are rejected. {py:class}`~kneemesh.config.LabelMapBrep` and {py:class}`~kneemesh.config.BrepMesh` hold the settings every structure shares, and a section for each kind of implemented structure holds its own settings. Keys of `labels`, `label_maps`, and `breps` must be names from {py:data}`~kneemesh.config.Structure`.

```{eval-rst}
.. autopydantic_model:: kneemesh.config.LabelMapAssembly
   :noindex:

.. autopydantic_model:: kneemesh.config.LabelMapBrep
   :noindex:

.. autopydantic_model:: kneemesh.config.BrepMesh
   :noindex:
```

## Shared section settings

Every structure's B-rep section extends {py:class}`~kneemesh.config.SurfaceBrep`, and every mesh section but the bones' extends {py:class}`~kneemesh.config.StructureMesh`.

```{eval-rst}
.. autopydantic_model:: kneemesh.config.SurfaceBrep
   :noindex:

.. autopydantic_model:: kneemesh.config.StructureMesh
   :noindex:
```

## Meniscus

The `meniscus` sections apply to both `lateral_meniscus` and `medial_meniscus`.

```{eval-rst}
.. autopydantic_model:: kneemesh.config.MeniscusBrep
   :noindex:

.. autopydantic_model:: kneemesh.config.MeniscusMesh
   :noindex:
```

## Tibial and patellar cartilage

The `cartilage` sections apply to `lateral_tibial_cartilage`, `medial_tibial_cartilage`, and `patellar_cartilage`.

```{eval-rst}
.. autopydantic_model:: kneemesh.config.CartilageBrep
   :noindex:

.. autopydantic_model:: kneemesh.config.CartilageMesh
   :noindex:
```

## Femoral cartilage

The `femoral_cartilage` sections apply to `femoral_cartilage`. Its B-rep settings extend the other cartilage's.

```{eval-rst}
.. autopydantic_model:: kneemesh.config.FemoralCartilageBrep
   :noindex:

.. autopydantic_model:: kneemesh.config.FemoralCartilageMesh
   :noindex:
```

## Bones

The `bone` sections apply to `femur`, `tibia`, and `patella`. A bone's geometry is its closed, triangulated surface, written as `.stl`, rather than a B-rep, and it is meshed with its own `element_type`: triangles or quadrilaterals on its surface, or tetrahedra, never hexahedra.

```{eval-rst}
.. autopydantic_model:: kneemesh.config.BoneBrep
   :noindex:

.. autopydantic_model:: kneemesh.config.BoneMesh
   :noindex:
```
