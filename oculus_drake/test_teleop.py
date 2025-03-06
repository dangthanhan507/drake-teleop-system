import numpy as np
from pydrake.geometry import StartMeshcat
from pydrake.multibody.inverse_kinematics import (
    DifferentialInverseKinematicsParameters,
    DifferentialInverseKinematicsStatus,
    DoDifferentialInverseKinematics,
)
from pydrake.systems.analysis import Simulator
from pydrake.systems.framework import DiagramBuilder, EventStatus, LeafSystem
from pydrake.visualization import MeshcatPoseSliders

from manipulation import running_as_notebook
from manipulation.meshcat_utils import WsgButton
from manipulation.station import load_scenario, MakeHardwareStation, AddIiwa, AddWsg, AddPlanarIiwa
from manipulation.scenarios import AddIiwaDifferentialIK
from pydrake.all import (
    MultibodyPlant,
    LeafSystem,
    Body,
    AbstractValue,
    RigidTransform,
    AddDefaultVisualization
)
class MultibodyPositionToBodyPose(LeafSystem):
    """A system that computes a body pose from a MultibodyPlant position vector. The
    output port calls `plant.SetPositions()` and then `plant.EvalBodyPoseInWorld()`.

    Args:
        plant: The MultibodyPlant.
        body: A body in the plant whose pose we want to compute (e.g. `plant.
            GetBodyByName("body")`).
    """

    def __init__(self, plant: MultibodyPlant, body: Body):
        LeafSystem.__init__(self)
        self.plant = plant
        self.body = body
        self.plant_context = plant.CreateDefaultContext()
        self.DeclareVectorInputPort("position", plant.num_positions())
        self.DeclareAbstractOutputPort(
            "pose",
            lambda: AbstractValue.Make(RigidTransform()),
            self._CalcOutput,
        )

    def _CalcOutput(self, context, output):
        position = self.get_input_port().Eval(context)
        self.plant.SetPositions(self.plant_context, position)
        pose = self.plant.EvalBodyPoseInWorld(self.plant_context, self.body)
        output.get_mutable_value().set(pose.rotation(), pose.translation())

meshcat = StartMeshcat()
scenario_data = """
directives:
- add_model:
    name: iiwa
    file: package://manipulation/planar_iiwa14_no_collision.urdf
    default_joint_positions:
        iiwa_joint_2: [0.1]
        iiwa_joint_4: [-1.2]
        iiwa_joint_6: [1.6]
- add_weld:
    parent: world
    child: iiwa::iiwa_link_0
- add_model:
    name: wsg
    file: package://manipulation/hydro/schunk_wsg_50_with_tip.sdf
- add_weld:
    parent: iiwa::iiwa_link_7
    child: wsg::body
    X_PC:
        translation: [0, 0, 0.09]
        rotation: !Rpy { deg: [90, 0, 90]}
- add_model:
    name: foam_brick
    file: package://manipulation/hydro/061_foam_brick.sdf
- add_model:
    name: robot_table
    file: package://manipulation/hydro/extra_heavy_duty_table_surface_only_collision.sdf
- add_weld:
    parent: world
    child: robot_table::link
    X_PC:
        translation: [0, 0, -0.7645]
- add_model:
    name: work_table
    file: package://manipulation/hydro/extra_heavy_duty_table_surface_only_collision.sdf
- add_weld:
    parent: world
    child: work_table::link
    X_PC:
        translation: [0.75, 0, -0.7645]
# Restrict the brick to move only in the x-z plane
- add_model:
    name: planar_joint
    file: package://manipulation/planar_joint.sdf
    default_joint_positions:
        planar_joint: [0.6, 0, 0]
- add_weld:
    parent: world
    child: planar_joint::parent
    X_PC:
        rotation: !Rpy { deg: [90, 0, 0]}
- add_weld:
    parent: planar_joint::child
    child: foam_brick::base_link
model_drivers:
    iiwa: !IiwaDriver
      control_mode: position_only
      hand_model_name: wsg
    wsg: !SchunkWsgDriver {}
"""


def teleop_2d():
    scenario = load_scenario(data=scenario_data)
    meshcat.Set2dRenderMode(xmin=-0.25, xmax=1.5, ymin=-0.1, ymax=1.3)

    builder = DiagramBuilder()

    station = builder.AddSystem(MakeHardwareStation(scenario, meshcat))

    # Set up differential inverse kinematics.
    diff_ik_plant = MultibodyPlant(time_step=1e-3)
    controller_iiwa = AddPlanarIiwa(diff_ik_plant)
    AddWsg(diff_ik_plant, controller_iiwa, welded=True)
    diff_ik_plant.Finalize()
    
    differential_ik = AddIiwaDifferentialIK(builder, diff_ik_plant, diff_ik_plant.GetFrameByName("iiwa_link_7"))
    builder.Connect(
        differential_ik.get_output_port(),
        station.GetInputPort("iiwa.position"),
    )
    builder.Connect(
        station.GetOutputPort("iiwa.state_estimated"),
        differential_ik.GetInputPort("robot_state"),
    )
    
    # AddDefaultVisualization(builder, meshcat)

    # Set up teleop widgets.
    meshcat.DeleteAddedControls()
    teleop = builder.AddSystem(
        MeshcatPoseSliders(
            meshcat,
            lower_limit=[0, -np.pi, -np.pi, -0.6, -1, 0],
            upper_limit=[2 * np.pi, np.pi, np.pi, 0.8, 1, 1.1],
            # Only roll, x, and z are used in this example:
            visible=[True, False, False, True, False, True],
            decrement_keycodes=["KeyQ", "", "", "ArrowLeft", "", "ArrowDown"],
            increment_keycodes=["KeyE", "", "", "ArrowRight", "", "ArrowUp"],
        )
    )
    builder.Connect(
        teleop.get_output_port(), differential_ik.GetInputPort("X_WE_desired")
    )
    ee_pose = builder.AddSystem(
        MultibodyPositionToBodyPose(
            diff_ik_plant, diff_ik_plant.GetBodyByName("iiwa_link_7")
        )
    )
    builder.Connect(
        station.GetOutputPort("iiwa.position_measured"), ee_pose.get_input_port()
    )
    builder.Connect(ee_pose.get_output_port(), teleop.get_input_port())
    wsg_teleop = builder.AddSystem(WsgButton(meshcat))
    builder.Connect(wsg_teleop.get_output_port(0), station.GetInputPort("wsg.position"))
    # builder.AddSystem(StopButton(meshcat))

    # Simulate.
    diagram = builder.Build()
    simulator = Simulator(diagram)

    simulator.set_target_realtime_rate(1.0)
    simulator.AdvanceTo(np.inf)


teleop_2d()
input()