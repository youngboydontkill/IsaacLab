# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass
import torch
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import (
    LocomotionVelocityRoughEnvCfg,
)

##
# Pre-defined configs
##
# from isaaclab.robots.unitree import UNITREE_GO2_CFG  # isort: skip
from ext_template.assets.kuavo import Kuavos46_CFG

import ext_template.tasks.locomotion.velocity.mdp as mdp
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from isaaclab.managers import RewardTermCfg as RewTerm
import math
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.sensors import RayCasterCfg, patterns


from dataclasses import MISSING
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from ext_template.sensors.volume_points import (
    Grid3dPointsGeneratorCfg,
    VolumePointsCfg,
)
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from ext_template.terrains import (
    ATTEN_ROUGH_TERRAINS_CFG,
    GreedyconcatEdgeCylinderCfg,
    ROUGH_TERRAINS_CFG,
    TerrainImporterCfg,
)
from .rough_env_cfg import MySceneCfg

@configclass
class AttentionSceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with a legged robot."""




    # ground terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=ATTEN_ROUGH_TERRAINS_CFG,
        max_init_terrain_level=0,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="average",
            restitution_combine_mode="average",
            static_friction=0.4,
            dynamic_friction=0.4,
            restitution=0.5,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        # virtual_obstacles={
        #     "edges": GreedyconcatEdgeCylinderCfg(
        #         cylinder_radius=0.05,
        #         min_points=2,
        #     ),
        # },
        debug_vis=False,
    )
    # robots
    robot: ArticulationCfg = MISSING
    # sensors
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base_link",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment='yaw',
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True
    )
    # leg_volume_points = VolumePointsCfg(
    #     prim_path="{ENV_REGEX_NS}/Robot/leg_[l,r]6_link",
    #     points_generator=Grid3dPointsGeneratorCfg(
    #         x_min=-0.025,
    #         x_max=0.12,
    #         x_num=10,
    #         y_min=-0.03,
    #         y_max=0.03,
    #         y_num=5,
    #         z_min=-0.04,
    #         z_max=0.0,
    #         z_num=2,
    #     ),
    #     debug_vis=False,
    # )
    # lights
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )
    Feet_L_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/leg_l6_link",
        offset=RayCasterCfg.OffsetCfg(pos=(0.05, 0.0, 20.0)),
        ray_alignment='yaw',
        pattern_cfg=patterns.GridPatternCfg(resolution=0.05, size=[0.2, 0.05]),
        debug_vis=True,
        mesh_prim_paths=["/World/ground"],
    )
    Feet_R_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/leg_r6_link",
        offset=RayCasterCfg.OffsetCfg(pos=(0.05, 0.0, 20.0)),
        ray_alignment='yaw',
        pattern_cfg=patterns.GridPatternCfg(resolution=0.05, size=[0.2, 0.05]),
        debug_vis=True,
        mesh_prim_paths=["/World/ground"],
    )

@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    # base_velocity = mdp.UniformVelocityCommandCfg(
    #     asset_name="robot",
    #     resampling_time_range=(5.0, 5.0),
    #     rel_standing_envs=0.1,
    #     rel_heading_envs=1.0,
    #     heading_command=True,
    #     heading_control_stiffness=0.5,
    #     debug_vis=True,
    #     ranges=mdp.UniformVelocityCommandCfg.Ranges(
    #         lin_vel_x=(-1.0, 1.0),
    #         lin_vel_y=(-0.5, 0.5),
    #         ang_vel_z=(-1.0, 1.0),
    #         heading=(-math.pi, math.pi),
    #     ), 
    # )
    base_velocity = mdp.UniformSteppingVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(5.0, 5.0),
        rel_standing_envs=0.1,
        rel_heading_envs=1.0,
        heading_command=True,
        heading_control_stiffness=0.5,
        debug_vis=True,
        ranges=mdp.UniformSteppingVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.0, 1.0),
            lin_vel_y=(-0.5, 0.5),
            ang_vel_z=(-1.0, 1.0),
            heading=(-math.pi, math.pi),
        ),
        rel_stepping_envs=0.5,    
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PerceptionCfg(ObsGroup):
        map_scan = ObsTerm(
            func=mdp.map_scan_base,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            # noise=Unoise(n_min=-0.1, n_max=0.1),
            clip=(-3.0, 2.0),
        )
        flatten_history_dim = False  # [B,H,D,...]
        history_length = 1
    
    @configclass
    class CommandCfg(ObsGroup):
        velocity_commands = ObsTerm(
            func=mdp.generated_commands, params={"command_name": "base_velocity"}
        )
        flatten_history_dim = False
        history_length = 5
        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""
        base_lin_vel = ObsTerm(
            func=mdp.base_lin_vel, 
            # noise=Unoise(n_min=-0.8, n_max=0.8)
        )
        # base_lin_vel = ObsTerm(
        # func=mdp.fixed_zero_vel,
        # # noise=Unoise(n_min=-0.1, n_max=0.1)
        # )

        # observation terms (order preserved)
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel, 
            # noise=Unoise(n_min=-0.2, n_max=0.2)
        )
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            # noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel, 
            # noise=Unoise(n_min=-0.05, n_max=0.05)
        )
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, 
                            # noise=Unoise(n_min=-1.5, n_max=1.5)
                            )
        actions = ObsTerm(func=mdp.last_action)
        # feet_contact_force = ObsTerm(
        #     func=mdp.feet_contact_force,
        #     params={
        #         "sensor_cfg": SceneEntityCfg(
        #             "contact_forces", body_names=["leg_[l,r]6_link"]
        #         )
        #     },
        # )
        # feet_heights = ObsTerm(
        #     func=mdp.feet_heights_bipeds,
        #     params={
        #         "sensor_cfg1": SceneEntityCfg("Feet_L_scanner"),
        #         "sensor_cfg2": SceneEntityCfg("Feet_R_scanner"),
        #     },
        # )
        # joint_torques = ObsTerm(func=mdp.joint_torques)
        # joint_accs = ObsTerm(func=mdp.joint_accs)
        # feet_lin_vel = ObsTerm(
        #     func=mdp.feet_lin_vel,
        #     params={
        #         "asset_cfg": SceneEntityCfg("robot", body_names=["leg_[l,r]6_link"])
        #     },
        # )
        flatten_history_dim = False
        history_length = 5

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
    
    @configclass
    class PrivilegedCfg(ObsGroup):
        """Observations for policy group."""
        base_lin_vel = ObsTerm(
            func=mdp.base_lin_vel, 
        )
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel, 
        )
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
        )
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel, 
        )
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, 
                            )
        actions = ObsTerm(func=mdp.last_action)
        # feet_contact_force = ObsTerm(
        #     func=mdp.feet_contact_force,
        #     params={
        #         "sensor_cfg": SceneEntityCfg(
        #             "contact_forces", body_names=["leg_[l,r]6_link"]
        #         )
        #     },
        # )
        # feet_heights = ObsTerm(
        #     func=mdp.feet_heights_bipeds,
        #     params={
        #         "sensor_cfg1": SceneEntityCfg("Feet_L_scanner"),
        #         "sensor_cfg2": SceneEntityCfg("Feet_R_scanner"),
        #     },
        # )
        # joint_torques = ObsTerm(func=mdp.joint_torques)
        # joint_accs = ObsTerm(func=mdp.joint_accs)
        # feet_lin_vel = ObsTerm(
        #     func=mdp.feet_lin_vel,
        #     params={
        #         "asset_cfg": SceneEntityCfg("robot", body_names=["leg_[l,r]6_link"])
        #     },
        # )
        
        # feet_air_times = ObsTerm(
        #     func=mdp.feet_air_time_obs,
        #     params={
        #         "sensor_cfg": SceneEntityCfg(
        #             "contact_forces", body_names="leg_[l,r]6_link"
        #         ),
        #     },
        # )
        flatten_history_dim = False
        history_length = 1
        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # @configclass
    # class PrivilegedCfg(ObsGroup):
    #     joint_torques = ObsTerm(func=mdp.joint_torques)
    #     joint_accs = ObsTerm(func=mdp.joint_accs)
    #     feet_lin_vel = ObsTerm(
    #         func=mdp.feet_lin_vel,
    #         params={
    #             "asset_cfg": SceneEntityCfg("robot", body_names=["leg_[l,r]6_link"])
    #         },
    #     )
    #     feet_contact_force = ObsTerm(
    #         func=mdp.feet_contact_force,
    #         params={
    #             "sensor_cfg": SceneEntityCfg(
    #                 "contact_forces", body_names=["leg_[l,r]6_link"]
    #             )
    #         },
    #     )
    #     base_mass_rel = ObsTerm(
    #         func=mdp.rigid_body_masses,
    #         params={"asset_cfg": SceneEntityCfg("robot", body_names="base_link")},
    #     )
    #     rigid_body_material = ObsTerm(
    #         func=mdp.rigid_body_material,
    #         params={
    #             "asset_cfg": SceneEntityCfg("robot", body_names=["leg_[l,r]6_link"])
    #         },
    #     )
    #     base_com = ObsTerm(
    #         func=mdp.base_com,
    #         params={"asset_cfg": SceneEntityCfg("robot", body_names="base_link")},
    #     )
    #     action_delay = ObsTerm(
    #         func=mdp.action_delay, params={"actuators_names": "motor"}
    #     )
    #     push_force = ObsTerm(
    #         func=mdp.push_force,
    #         params={"asset_cfg": SceneEntityCfg("robot", body_names="base_link")},
    #     )
    #     push_torque = ObsTerm(
    #         func=mdp.push_torque,
    #         params={"asset_cfg": SceneEntityCfg("robot", body_names="base_link")},
    #     )

    #     feet_heights = ObsTerm(
    #         func=mdp.feet_heights_bipeds,
    #         params={
    #             "sensor_cfg1": SceneEntityCfg("Feet_L_scanner"),
    #             "sensor_cfg2": SceneEntityCfg("Feet_R_scanner"),
    #         },
    #     )
    #     feet_air_times = ObsTerm(
    #         func=mdp.feet_air_time_obs,
    #         params={
    #             "sensor_cfg": SceneEntityCfg(
    #                 "contact_forces", body_names="leg_[l,r]6_link"
    #             ),
    #         },
    #     )
    #     history_length = 1
    #     flatten_history_dim = False
    #     def __post_init__(self):
    #         self.enable_corruption = False
    #         self.concatenate_terms = True

    # observation groups

    # @configclass
    # class AMPPolicyCfg(ObsGroup):
    #     # 64：joint_pos joint_vel end_eff_pos
    #     joint_pos = ObsTerm(
    #         func=mdp.joint_pos_rel, 
    #         noise=Unoise(n_min=-0.05, n_max=0.05)
    #     )
    #     joint_vel = ObsTerm(func=mdp.joint_vel_rel, 
    #                         )
    #     end_eff_pos = ObsTerm(func=mdp.end_eff_pos_amp, 
    #                           params={
    #                             "asset_cfg": SceneEntityCfg("robot"),
    #                             "end_effector_body_id": "zarm_[l,r]7_link,leg_[l,r]6_link",
    #                         },
    #     )
    command: CommandCfg = CommandCfg()
    policy: PolicyCfg = PolicyCfg()
    privileged: PrivilegedCfg = PrivilegedCfg()
    perception: PerceptionCfg = PerceptionCfg()
    #amp_policy: AMPPolicyCfg = AMPPolicyCfg()


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # # # -- task
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_yaw_frame_exp,
        # weight=5.0,
        weight=4.0,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_world_exp,
        weight=4.0,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )
    # -- penalties
    # lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-0.2)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    dof_power_l2 = RewTerm(func=mdp.joint_power_l2, weight=-2.0e-5)
    dof_torques_l2 = RewTerm(
        func=mdp.joint_torques_l2,
        weight=-1.0e-5,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=["leg_[l,r][1-5]_joint", "zarm_.*_joint"]
            )
        },
    )
    dof_torques_ankle_l2 = RewTerm(
        func=mdp.joint_torques_l2,
        weight=-1.0e-5,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["leg_[l,r]6_joint"])},
    )
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.005)
    action_smoothness_l2 = RewTerm(func=mdp.action_smoothness_l2, weight=-0.01)
    # volume_points_penetration = RewTerm(
    #     func=mdp.volume_points_penetration,
    #     weight=-4.0,
    #     params={
    #         "sensor_cfg": SceneEntityCfg("leg_volume_points"),
    #     },
    # )

    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=["leg_[l,r][1-5]_link", "base_link", "zarm_.*_link"],
            ),
            "threshold": 1.0,
        },
    )
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-10.0)
    # yysy,感觉得加
    is_terminated = RewTerm(func=mdp.is_terminated, weight=-200.0)

    # feet_air_time = RewTerm(
    #     func=mdp.feet_air_time_clip,
    #     weight=10.0,
    #     params={
    #         "command_name": "base_velocity",
    #         "sensor_cfg": SceneEntityCfg(
    #             "contact_forces", body_names="leg_[l,r]6_link"
    #         ),
    #         "threshold_min": 0.2,
    #         "threshold_max": 0.5,
    #         "use_stance_mask": False,
    #     },
    # )
    feet_air_time = RewTerm(
        func=mdp.feet_air_time_positive_biped,
        weight=1,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names="leg_[l,r]6_link"
            ),
            "threshold": 0.5,
            "use_stance_mask": True,
        },
    )


    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.1,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names="leg_[l,r]6_link"
            ),
            "asset_cfg": SceneEntityCfg("robot", body_names="leg_[l,r]6_link"),
        },
    )

    track_default_arm_pos = RewTerm(
        func=mdp.track_default_arm_pos,
        weight=6.0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    "zarm_l2_joint", "zarm_l3_joint", "zarm_l5_joint",
                    "zarm_l6_joint", "zarm_l7_joint",
                    "zarm_r2_joint", "zarm_r3_joint", "zarm_r5_joint",
                    "zarm_r6_joint", "zarm_r7_joint",
                ]
            ),
            "alpha": 5.0
        }
    )

    feet_contact_without_cmd = RewTerm(
        func=mdp.feet_contact_without_cmd,
        weight=0.4,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["leg_[l,r]6_link"]),
            "use_stance_mask": False,
        },
    )

    joint_deviation_hip = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.1,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["leg_[l,r][1,2]_joint"])
        },
    )
    joint_deviation_arms = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.1,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    "zarm_.*",
                ],
            )
        },
    )
    flat_orientation_l2 = RewTerm(
        func=mdp.flat_orientation_l2,
        weight=-3.0,
    )

    contact_force = RewTerm(
        func=mdp.contact_forces,
        weight=-0.001,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names="leg_[l,r]6_link"
            ),
            "threshold": 900,
            "violation_max": 300,
        },
    )

    stand_still_without_cmd = RewTerm(
        func=mdp.stand_still_without_cmd,
        weight=-10.0,
        params={
            "command_name": "base_velocity",
            "use_stance_mask": False,
        },
    )
    # 加入feet contact的惩罚
    feet_stumble = RewTerm(
        func=mdp.feet_stumble,
        weight=-1.0,
        params={"sensor_cfg": SceneEntityCfg(
            "contact_forces", body_names="leg_[l,r]6_link")},
    )
    no_feet_contact = RewTerm(
        func=mdp.no_feet_contact,
        weight=-0.1,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names="leg_[l,r]6_link"),
            "use_stance_mask": False,
        },
    )
    illegal_dof_barrier = RewTerm(
        func=mdp.illegal_dof_pos_barrier,
        weight=-0.1,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["leg_l[5,6]_joint", "leg_r[5,6]_joint"])
        },
    )
    feet_too_near = RewTerm(
        func=mdp.feet_too_near_humanoid,
        weight=-5.0,
    )
    fly = RewTerm(
        func=mdp.fly,
        weight=-10.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="leg_[l,r]6_link"), "threshold": 1.0},
    )
    # 单加这个，看见台阶直接跪了
    feet_solid_contact = RewTerm(
        func=mdp.feet_solid_contact,
        weight=-0.05,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names="leg_[l,r]6_link"), 
            "sensor_cfg1": SceneEntityCfg("Feet_L_scanner"),
            "sensor_cfg2": SceneEntityCfg("Feet_R_scanner"),
            "threshold": 5.0,
            "feet_height_threshold": 0.13},
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names="base_link"),
            "threshold": 1.0,
        },
    )

    dof_pos_illegal = DoneTerm(
        func=mdp.dof_pos_illegal,
        params={
            "actuators_names": "motor"
        },
    )


@configclass
class EventCfg:
    """Configuration for events."""

    # startup
    # physics_material = EventTerm(
    #     func=mdp.randomize_rigid_body_material,
    #     mode="startup",
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
    #         "static_friction_range": (0.2, 1.0),
    #         "dynamic_friction_range": (0.1, 0.9),
    #         "restitution_range": (0.0, 0.5),
    #         "num_buckets": 64,
    #         "make_consistent": True,
    #     },
    # )

    # add_base_mass = EventTerm(
    #     func=mdp.randomize_rigid_body_mass,
    #     mode="startup",
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
    #         "mass_distribution_params": (-2.0, 2.0),
    #         "operation": "add",
    #     },
    # )

    # scale_link_mass = EventTerm(
    #     func=mdp.randomize_rigid_body_mass,
    #     mode="startup",
    #     params={
    #         "asset_cfg": SceneEntityCfg(
    #             "robot", body_names=["leg_.*_link", "zarm_.*_link"]
    #         ),
    #         "mass_distribution_params": (0.8, 1.2),
    #         "operation": "scale",
    #     },
    # )

    # randomize_rigid_body_com = EventTerm(
    #     func=mdp.randomize_base_body_com,
    #     mode="startup",
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
    #         "com_range": {"x": (-0.1, 0.1), "y": (-0.1, 0.1), "z": (-0.1, 0.1)},
    #     },
    # )

    # scale_actuator_gains = EventTerm(
    #     func=mdp.randomize_actuator_gains,
    #     mode="startup",
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", joint_names=".*_joint"),
    #         "stiffness_distribution_params": (0.8, 1.2),
    #         "damping_distribution_params": (0.8, 1.2),
    #         "operation": "scale",
    #     },
    # )

    # scale_joint_parameters = EventTerm(
    #     func=mdp.randomize_joint_parameters,
    #     mode="startup",
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", joint_names=".*_joint"),
    #         "friction_distribution_params": (1.0, 1.0),
    #         "armature_distribution_params": (0.5, 1.5),
    #         "operation": "scale",
    #     },
    # )

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.7, 0.7), "y": (-0.7, 0.7), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (-0.3, 0.3),
                "y": (-0.3, 0.3),
                "z": (-0.3, 0.3),
                "roll": (-0.3, 0.3),
                "pitch": (-0.3, 0.3),
                "yaw": (-0.3, 0.3),
            },
        },
    )

    # register_virtual_obstacles = EventTerm(
    #     func=mdp.register_virtual_obstacle_to_sensor,
    #     mode="startup",
    #     params={
    #         "sensor_cfgs": SceneEntityCfg("leg_volume_points"),
    #     },
    # )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (0.5, 1.5),
            "velocity_range": (0.0, 0.0),
        },
    )

    # base_external_force_torque = EventTerm(
    #     func=mdp.apply_external_force_torque_stochastic,
    #     mode="interval",
    #     interval_range_s=(0.0, 0.0),
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
    #         "force_range": {
    #             "x": (-100.0, 100.0),
    #             "y": (-100.0, 100.0),
    #             "z": (-100.0, 100.0),
    #         },  # force = mass * dv / dt
    #         "torque_range": {"x": (-0.0, 0.0), "y": (-0.0, 0.0), "z": (-0.0, 0.0)},
    #         "probability": 0.002,  # Expect step = 1 / probability
    #     },
    # )


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)


@configclass
class KuavoAttention2RoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    # scene: MySceneCfg = MySceneCfg(num_envs=4096, env_spacing=2.5)
    scene: AttentionSceneCfg = AttentionSceneCfg(num_envs=4096, env_spacing=2.5)
    commands: CommandsCfg = CommandsCfg()
    observations: ObservationsCfg = ObservationsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        self.scene.robot = Kuavos46_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/base_link"
        # reduce action scale
        self.actions.joint_pos.scale = 0.25

@configclass
class ObservationsPlayCfg(ObservationsCfg):
    """Observations for the Play."""
    @configclass
    class VisualizeCfg(ObsGroup):
        root_pos = ObsTerm(
            func=mdp.root_pos_world,
        )
        root_quat = ObsTerm(
            func=mdp.root_quat_w,
        )
        flatten_history_dim = False  # [B,H,D,...]
        history_length = 1

    visualize: VisualizeCfg = VisualizeCfg()

@configclass
class KuavoAttention2RoughEnvCfg_PLAY(KuavoAttention2RoughEnvCfg):
    observations: ObservationsPlayCfg = ObservationsPlayCfg()
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # spawn the robot randomly in the grid (instead of their terrain levels)
        self.scene.terrain.max_init_terrain_level = None
        # reduce the number of terrains to save memory
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False

        # remove random pushing event
        # self.events.base_external_force_torque = None

        self.commands.base_velocity.ranges.lin_vel_x = (0, 0.8)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0, 0.0)
        # self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        # self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)