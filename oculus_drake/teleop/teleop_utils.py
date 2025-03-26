from pydrake.systems.framework import DiagramBuilder, LeafSystem
from manipulation.station import (
    MakeHardwareStationInterface,
    Scenario,
    ConfigureParser,
    ProcessModelDirectives,
    ModelDirectives,
    _ApplyDriverConfigsSim,
    _ApplyCameraConfigSim
)
import typing
from pydrake.all import (
    MultibodyPlant,
    Meshcat,
    Parser,
    AddMultibodyPlant,
    LeafSystem,
    Body,
    AbstractValue,
    RigidTransform,
    AddDefaultVisualization,
    DifferentialInverseKinematicsIntegrator,
    DifferentialInverseKinematicsParameters,
    DifferentialInverseKinematicsStatus,
    DoDifferentialInverseKinematics,
    EventStatus
)
from pydrake.all import AddDefaultVisualization
import numpy as np

def MakeFakeStation(
    scenario: Scenario,
    meshcat: Meshcat = None,
    *,
    package_xmls: typing.List[str] = [],    
):
    builder = DiagramBuilder()

    # Create the multibody plant and scene graph.
    sim_plant, scene_graph = AddMultibodyPlant(
        config=scenario.plant_config, builder=builder
    )
    

    parser = Parser(sim_plant)
    for p in package_xmls:
        parser.package_map().AddPackageXml(p)
    ConfigureParser(parser)

    # Add model directives.
    added_models = ProcessModelDirectives(
        directives=ModelDirectives(directives=scenario.directives),
        parser=parser,
    )

    # Now the plant is complete.
    sim_plant.Finalize()
    if meshcat:
        AddDefaultVisualization(builder, meshcat)
    
    diagram = builder.Build()
    return diagram

def MakeHardwareStation(
    scenario: Scenario,
    meshcat: Meshcat = None,
    *,
    package_xmls: typing.List[str] = [],
    hardware: bool = False,
    parser_preload_callback: typing.Callable[[Parser], None] = None,
    parser_prefinalize_callback: typing.Callable[[Parser], None] = None,
):
    """
    If `hardware=False`, (the default) returns a HardwareStation diagram containing:
      - A MultibodyPlant with populated via the directives in `scenario`.
      - A SceneGraph
      - The default Drake visualizers
      - Any robot / sensors drivers specified in the YAML description.

    If `hardware=True`, returns a HardwareStationInterface diagram containing the network interfaces to communicate directly with the hardware drivers.

    Args:
        scenario: A Scenario structure, populated using the `load_scenario` method.

        meshcat: If not None, then AddDefaultVisualization will be added to the subdiagram using this meshcat instance.

        package_xmls: A list of package.xml file paths that will be passed to the parser, using Parser.AddPackageXml().
    """
    if hardware:
        return MakeHardwareStationInterface(
            scenario=scenario, meshcat=meshcat, package_xmls=package_xmls
        )

    builder = DiagramBuilder()

    # Create the multibody plant and scene graph.
    sim_plant, scene_graph = AddMultibodyPlant(
        config=scenario.plant_config, builder=builder
    )
    

    parser = Parser(sim_plant)
    for p in package_xmls:
        parser.package_map().AddPackageXml(p)
    ConfigureParser(parser)
    if parser_preload_callback:
        parser_preload_callback(parser)

    # Add model directives.
    added_models = ProcessModelDirectives(
        directives=ModelDirectives(directives=scenario.directives),
        parser=parser,
    )

    if parser_prefinalize_callback:
        parser_prefinalize_callback(parser)

    # Now the plant is complete.
    sim_plant.Finalize()
    AddDefaultVisualization(builder, meshcat)

    # Add drivers.
    _ApplyDriverConfigsSim(
        driver_configs=scenario.model_drivers,
        sim_plant=sim_plant,
        directives=scenario.directives,
        models_from_directives=added_models,
        package_xmls=package_xmls,
        builder=builder,
    )

    # Add scene cameras.
    for _, camera in scenario.cameras.items():
        _ApplyCameraConfigSim(config=camera, builder=builder)

    # Add visualization.
    # ApplyVisualizationConfig(scenario.visualization, builder, meshcat=meshcat)

    # Export "cheat" ports.
    builder.ExportOutput(scene_graph.get_query_output_port(), "query_object")
    builder.ExportOutput(
        sim_plant.get_contact_results_output_port(), "contact_results"
    )
    builder.ExportOutput(
        sim_plant.get_state_output_port(), "plant_continuous_state"
    )
    builder.ExportOutput(sim_plant.get_body_poses_output_port(), "body_poses")

    diagram = builder.Build()
    diagram.set_name("station")
    return diagram

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

def DiffIKParams(plant, xyz_speed_limit = 0.03, time_step=1e-4):
    params = DifferentialInverseKinematicsParameters(
        plant.num_positions(), plant.num_velocities()
    )
    q0 = plant.GetPositions(plant.CreateDefaultContext())
    params.set_nominal_joint_position(q0)
    # params.set_joint_acceleration_limits()
    params.set_end_effector_angular_speed_limit(20.0 * np.pi / 180) # 20 deg/s
    params.set_end_effector_translational_velocity_limits(
        [-xyz_speed_limit, -xyz_speed_limit, -xyz_speed_limit], [xyz_speed_limit, xyz_speed_limit, xyz_speed_limit]
    )
    
    iiwa14_velocity_limits = np.array([1.4, 1.4, 1.7, 1.3, 2.2, 2.3, 2.3])
    params.set_joint_velocity_limits(
        (-iiwa14_velocity_limits, iiwa14_velocity_limits)
    )
    params.set_joint_centering_gain(0 * np.eye(7)) # no nullspace crap
    params.set_time_step(time_step)
    return params

def AddIiwaDifferentialIK(builder, plant, frame=None, xyz_speed_limit = 0.03, time_step=1e-4):
    params = DifferentialInverseKinematicsParameters(
        plant.num_positions(), plant.num_velocities()
    )
    q0 = plant.GetPositions(plant.CreateDefaultContext())
    params.set_nominal_joint_position(q0)
    # params.set_joint_acceleration_limits()
    params.set_end_effector_angular_speed_limit(20.0 * np.pi / 180) # 20 deg/s
    params.set_end_effector_translational_velocity_limits(
        [-xyz_speed_limit, -xyz_speed_limit, -xyz_speed_limit], [xyz_speed_limit, xyz_speed_limit, xyz_speed_limit]
    )
    
    iiwa14_velocity_limits = np.array([1.4, 1.4, 1.7, 1.3, 2.2, 2.3, 2.3])
    params.set_joint_velocity_limits(
        (-iiwa14_velocity_limits, iiwa14_velocity_limits)
    )
    params.set_joint_centering_gain(0 * np.eye(7)) # no nullspace crap
    
    if frame is None:
        frame = plant.GetFrameByName("body")
    differential_ik = builder.AddSystem(
        DifferentialInverseKinematicsIntegrator(
            plant,
            frame,
            time_step,
            params,
            log_only_when_result_state_changes=True,
        )
    )
    return differential_ik

class DiffIKSystem(LeafSystem):
    def __init__(self, plant, frame_E, parameters: DifferentialInverseKinematicsParameters, time_step=1e-4):
        LeafSystem.__init__(self)
        self._diff_ik_params = parameters
        self._frame_E = frame_E
        self._plant = plant
        self._plant_context = plant.CreateDefaultContext()
        
        self.V_WE_port = self.DeclareVectorInputPort("V_WE", 6)
        self.robot_state_port = self.DeclareVectorInputPort("robot_state", plant.num_multibody_states())
        self.DeclareInitializationDiscreteUpdateEvent(self.Initialize)
        
        self.out_port = self.DeclareVectorOutputPort(
            "iiwa.position", plant.num_positions(), self.OutputJointPosition
        )
        self.out_port.disable_caching_by_default()
        self._time_step = time_step
        self.DeclareDiscreteState(plant.num_positions())
        self.DeclarePeriodicDiscreteUpdateEvent(self._time_step, 0, self.Integrate)
        
    def Initialize(self, context, discrete_state):
        discrete_state.set_value(
            0,
            self.robot_state_port.Eval(context)[:self._plant.num_positions()]
        )
    def Integrate(self, context, discrete_state):
        V_WE = np.copy(self.V_WE_port.Eval(context) / self._time_step)
        # preprocess V_WE
        if np.linalg.norm(V_WE[:3]) > self._diff_ik_params.get_end_effector_angular_speed_limit():
            V_WE[:3] = V_WE[:3] / np.linalg.norm(V_WE[:3])
            V_WE[:3] = V_WE[:3] * self._diff_ik_params.get_end_effector_angular_speed_limit()
        
        lower_vel_limit, upper_vel_limit = self._diff_ik_params.get_end_effector_translational_velocity_limits()
        lower_vel_limit = np.array(lower_vel_limit)
        upper_vel_limit = np.array(upper_vel_limit)
        V_WE[3:] = np.clip(V_WE[3:], lower_vel_limit, upper_vel_limit)
        
        
        assert np.all(V_WE[3:] >= lower_vel_limit) and np.all(V_WE[3:] <= upper_vel_limit)
        assert np.linalg.norm(V_WE[:3]) <= self._diff_ik_params.get_end_effector_angular_speed_limit() + 1e-3
        
        q = np.copy(context.get_discrete_state(0).get_value())
        self._plant.SetPositions(self._plant_context, q)
        result = DoDifferentialInverseKinematics(
            self._plant,
            self._plant_context,
            V_WE,
            self._frame_E,
            self._diff_ik_params
        )
        if result.status != DifferentialInverseKinematicsStatus.kNoSolutionFound:
            discrete_state.set_value(0, q + self._time_step * result.joint_velocities)
        else:
            discrete_state.set_value(0, q)
        return EventStatus.Succeeded()
    
    def OutputJointPosition(self, context, output):
        output.set_value(context.get_discrete_state(0).get_value())