# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import os

import omni.kit.commands
import omni.usd
from omni.isaac.core.utils.extensions import enable_extension
from omni.isaac.version import get_version
from pxr import Gf, Usd, UsdPhysics

from .asset_converter_base import AssetConverterBase
from .urdf_converter_cfg import UrdfConverterCfg

_DRIVE_TYPE = {
    "none": 0,
    "position": 1,
    "velocity": 2,
}
"""Mapping from drive type name to URDF importer drive number."""

_NORMALS_DIVISION = {
    "catmullClark": 0,
    "loop": 1,
    "bilinear": 2,
    "none": 3,
}
"""Mapping from normals division name to urdf importer normals division number."""


class UrdfConverter(AssetConverterBase):
    """Converter for a URDF description file to a USD file.

    This class wraps around the `omni.isaac.urdf_importer`_ extension to provide a lazy implementation
    for URDF to USD conversion. It stores the output USD file in an instanceable format since that is
    what is typically used in all learning related applications.

    .. caution::
        The current lazy conversion implementation does not automatically trigger USD generation if
        only the mesh files used by the URDF are modified. To force generation, either set
        :obj:`AssetConverterBaseCfg.force_usd_conversion` to True or delete the output directory.

    .. note::
        From Isaac Sim 2023.1 onwards, the extension name changed from ``omni.isaac.urdf`` to
        ``omni.importer.urdf``. This converter class automatically detects the version of Isaac Sim
        and uses the appropriate extension.

        The new extension supports a custom XML tag``"dont_collapse"`` for joints. Setting this parameter
        to true in the URDF joint tag prevents the child link from collapsing when the associated joint type
        is "fixed".

    .. _omni.isaac.urdf_importer: https://docs.omniverse.nvidia.com/isaacsim/latest/ext_omni_isaac_urdf.html
    """

    cfg: UrdfConverterCfg
    """The configuration instance for URDF to USD conversion."""

    def __init__(self, cfg: UrdfConverterCfg):
        """Initializes the class.

        Args:
            cfg: The configuration instance for URDF to USD conversion.
        """
        super().__init__(cfg=cfg)

    """
    Implementation specific methods.
    """

    def _convert_asset(self, cfg: UrdfConverterCfg):
        """Calls underlying Omniverse command to convert URDF to USD.

        Args:
            cfg: The URDF conversion configuration.
        """
        import_config = self._get_urdf_import_config(cfg)
        omni.kit.commands.execute(
            "URDFParseAndImportFile",
            urdf_path=cfg.asset_path,
            import_config=import_config,
            dest_path=self.usd_path,
        )
        # fix the issue that material paths are not relative
        if self.cfg.make_instanceable:
            instanced_usd_path = os.path.join(self.usd_dir, self.usd_instanceable_meshes_path)
            stage = Usd.Stage.Open(instanced_usd_path)
            # resolve all paths relative to layer path
            source_layer = stage.GetRootLayer()
            omni.usd.resolve_paths(source_layer.identifier, source_layer.identifier)
            stage.Save()

        # fix the issue that material paths are not relative
        # note: This issue seems to have popped up in Isaac Sim 2023.1.1
        stage = Usd.Stage.Open(self.usd_path)
        # resolve all paths relative to layer path
        source_layer = stage.GetRootLayer()
        omni.usd.resolve_paths(source_layer.identifier, source_layer.identifier)
        # Isaac Sim 5.x compatibility: post-process the generated USD to prevent
        # PhysX 5.x from hanging during articulation construction.
        self._fix_physx5_compatibility(stage)
        stage.Save()

    """
    Helper methods.
    """

    def _fix_physx5_compatibility(self, stage: Usd.Stage):
        """Post-process the generated USD to ensure PhysX 5.x compatibility.

        Isaac Sim 5.x ships with PhysX 5.x which has stricter requirements for
        reduced-coordinate articulations. Without these fixes the physics timeline
        may hang during ``commit()`` / articulation construction.

        The following issues are addressed:

        1. **Instanceable prims**: Isaac Sim 5.x's URDF importer authors instanceable
           flags on link visuals/collisions even when ``make_instanceable`` is false.
           PhysX 5.x cannot resolve instanceable physics prims during articulation
           building, causing a hang.
        2. **Degenerate mass/inertia**: Links with near-zero mass or zero diagonal
           inertia cause singular mass matrices in PhysX 5.x, which can trigger an
           infinite loop in the articulation builder.

        Args:
            stage: The USD stage to fix (modified in-place).
        """
        min_mass = 0.1
        min_inertia = 1e-4

        for prim in stage.TraverseAll():
            # -- 1. Clear instanceable flags on all prims when not intentionally instanceable --
            #    Isaac Sim 5.x's URDF importer writes instanceable=true even when
            #    make_instanceable is false. PhysX 5.x cannot build articulations
            #    that contain instanceable physics prims.
            if not self.cfg.make_instanceable and prim.IsInstanceable():
                prim.SetInstanceable(False)

            # -- 2. Fix degenerate mass properties on rigid bodies --
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                # Isaac Sim 5.x: RigidBodyAPI may not expose GetMassAttr();
                # access the "physics:mass" attribute directly for portability.
                mass_attr = prim.GetAttribute("physics:mass")
                if mass_attr.IsValid():
                    mass = mass_attr.Get()
                    if mass is None or mass < min_mass:
                        mass_attr.Set(min_mass)

                # diagonalInertia is authored via the RigidBodyAPI or PhysxRigidBodyAPI
                inertia_attr = prim.GetAttribute("physics:diagonalInertia")
                if inertia_attr.IsValid() and inertia_attr.IsAuthored():
                    inertia = inertia_attr.Get()
                    if inertia is not None:
                        new_inertia = list(inertia)
                        fixed = False
                        for i in range(3):
                            if new_inertia[i] < min_inertia:
                                new_inertia[i] = min_inertia
                                fixed = True
                        if fixed:
                            inertia_attr.Set(Gf.Vec3f(*new_inertia))

    def _get_urdf_import_config(self, cfg: UrdfConverterCfg) -> omni.importer.urdf.ImportConfig:
        """Create and fill URDF ImportConfig with desired settings

        Args:
            cfg: The URDF conversion configuration.

        Returns:
            The constructed ``ImportConfig`` object containing the desired settings.
        """
        # Enable urdf extension. Isaac Sim 5.x renamed the package from
        # ``omni.importer.urdf`` to ``isaacsim.asset.importer.urdf``.
        try:
            enable_extension("omni.importer.urdf")
            from omni.importer.urdf import _urdf as omni_urdf
        except ModuleNotFoundError:
            enable_extension("isaacsim.asset.importer.urdf")
            from isaacsim.asset.importer.urdf import _urdf as omni_urdf

        import_config = omni_urdf.ImportConfig()

        def set_import_option(name: str, value):
            setter_name = f"set_{name}"
            if hasattr(import_config, setter_name):
                getattr(import_config, setter_name)(value)
            elif hasattr(import_config, name):
                setattr(import_config, name, value)

        # set the unit scaling factor, 1.0 means meters, 100.0 means cm
        set_import_option("distance_scale", 1.0)
        # set imported robot as default prim
        set_import_option("make_default_prim", True)
        # add a physics scene to the stage on import if none exists
        set_import_option("create_physics_scene", False)

        # -- instancing settings
        # meshes will be placed in a separate usd file
        set_import_option("make_instanceable", cfg.make_instanceable)
        set_import_option("instanceable_usd_path", self.usd_instanceable_meshes_path)

        # -- asset settings
        # default density used for links, use 0 to auto-compute
        set_import_option("density", cfg.link_density)
        # import inertia tensor from urdf, if it is not specified in urdf it will import as identity
        set_import_option("import_inertia_tensor", cfg.import_inertia_tensor)
        # decompose a convex mesh into smaller pieces for a closer fit
        set_import_option("convex_decomp", cfg.convex_decompose_mesh)
        set_import_option("subdivision_scheme", _NORMALS_DIVISION["bilinear"])

        # -- physics settings
        # create fix joint for base link
        set_import_option("fix_base", cfg.fix_base)
        # consolidating links that are connected by fixed joints
        set_import_option("merge_fixed_joints", cfg.merge_fixed_joints)
        # self collisions between links in the articulation
        set_import_option("self_collision", cfg.self_collision)

        # default drive type used for joints
        set_import_option("default_drive_type", _DRIVE_TYPE[cfg.default_drive_type])
        # default proportional gains
        set_import_option("default_drive_strength", cfg.default_drive_stiffness)
        # default derivative gains
        set_import_option("default_position_drive_damping", cfg.default_drive_damping)
        if get_version()[2] == "4":
            # override joint dynamics parsed from urdf
            set_import_option("override_joint_dynamics", cfg.override_joint_dynamics)

        return import_config
