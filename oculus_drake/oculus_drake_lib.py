from pydrake.all import (
    LeafSystem,
    RigidTransform,
    Value,
    DiagramBuilder,
    MultibodyPlant,
    DifferentialInverseKinematicsIntegrator,
    DoDifferentialInverseKinematics,
    DifferentialInverseKinematicsStatus,
    EventStatus,
    Meshcat,
    ModelDirectives,
    Parser,
    SceneGraph,
    AddMultibodyPlant,
    ProcessModelDirectives,
    AddDefaultVisualization,
    Cylinder,
    UnitInertia,
    SpatialInertia,
    RotationMatrix,
    MultibodyPlantConfig,
    Multiplexer,
    ConstantVectorSource,
    InverseDynamicsController,
    FixedOffsetFrame,
    ApplyCameraConfig,
    CameraInfo,
    Diagram,
    ImageRgba8U,
    ImageDepth16U,
    ImageLabel16I,
    ColorizeLabelImage,
    PackageMap,
	IllustrationProperties,
	GeometrySet,
	Role,
	RoleAssign,
    InverseKinematics,
    RollPitchYaw,
    SnoptSolver,
    Cylinder,
    Box,
    UnitInertia,
    SpatialInertia,
    RotationMatrix,
    GeometryInstance,
    MakePhongIllustrationProperties,
    Shape,
    SourceId,
    ApplyMultibodyPlantConfig,
    ConstantValueSource
)
from manipulation.scenarios import AddMultibodyTriad
from manipulation.meshcat_utils import WsgButton
from teleop_utils import MakeHardwareStation, MultibodyPositionToBodyPose, AddIiwaDifferentialIK, MakeFakeStation
from manipulation import ConfigureParser
from oculus_drake import PACKAGE_XML, SCENARIO_FILEPATH, FAKE_SCENARIO_FILEPATH
from manipulation.station import load_scenario, MakeHardwareStationInterface, AddIiwa, AddWsg
from oculus_reader.reader import OculusReader
import numpy as np
import time
class OculusSystem(LeafSystem):
    def __init__(self, sensor_read_hz = 60.0):
        LeafSystem.__init__(self)
        self.reader = OculusReader()
        time.sleep(0.3)
        
        '''
            NOTE: (Structure of self.reader.get_transformations_and_buttons())
                -> First Dict:
                    -> 'l': 4x4 numpy array (R,t)
                    -> 'r': 4x4 numpy array (R,t)
                -> Second Dict:
                    -> 'A': bool
                    -> 'B': bool
                    -> 'X': bool
                    -> 'Y': bool
                    
                    -> 'LJ': bool
                    -> 'RJ': bool
                    -> 'LG': bool
                    -> 'RG': bool
                    
                    -> 'leftTrig': [0.0, 1.0]
                    -> 'rightTrig': [0.0, 1.0]
                    
                    -> 'leftGrip': [0.0, 1.0]
                    -> 'rightGrip': [0.0, 1.0]
                    
                    -> 'leftJS': [0.0, 1.0] x [0.0, 1.0]
                    -> 'rightJS': [0.0, 1.0] x [0.0, 1.0]
        '''
        self.sensor_read_hz = sensor_read_hz
        # periodic publish system buttons / outputs
        
        # buttons
        self.buttonA = False
        self.buttonB = False
        self.buttonX = False
        self.buttonY = False
        
        self.buttonLJ = False
        self.buttonRJ = False
        self.buttonLG = False
        self.buttonRG = False
        
        # triggers
        self.leftTrig = 0.0
        self.rightTrig = 0.0
        self.leftGrip = 0.0
        self.rightGrip = 0.0
        
        self.leftJS = [0.0, 0.0]
        self.rightJS = [0.0, 0.0]
        
        # poses
        self.left_pose = RigidTransform()
        self.right_pose = RigidTransform()
        self.oculus_read()
            
        self.DeclarePeriodicPublishEvent(period_sec=1.0/sensor_read_hz, offset_sec=0.0, publish=self.OculusRead)
    def OculusRead(self, context):
        self.oculus_read()
    def oculus_read(self):
        transform_dict, buttons_dict = self.reader.get_transformations_and_buttons()
        # print(transform_dict, buttons_dict)
        self.left_pose = RigidTransform(transform_dict['l']) # +z points towards where controller faces
        self.right_pose = RigidTransform(transform_dict['r']) # +z points towards where controller faces
        
        # buttons
        self.buttonA = buttons_dict['A']
        self.buttonB = buttons_dict['B']
        self.buttonX = buttons_dict['X']
        self.buttonY = buttons_dict['Y']
        
        self.buttonLJ = buttons_dict['LJ']
        self.buttonRJ = buttons_dict['RJ']
        self.buttonLG = buttons_dict['LG']
        self.buttonRG = buttons_dict['RG']
        
        # triggers
        self.leftTrig = buttons_dict['leftTrig']
        self.rightTrig = buttons_dict['rightTrig']
        self.leftGrip = buttons_dict['leftGrip']
        self.rightGrip = buttons_dict['rightGrip']
        
        # triggers
        self.leftTrig = buttons_dict['leftTrig']
        self.rightTrig = buttons_dict['rightTrig']
        self.leftGrip = buttons_dict['leftGrip']
        self.rightGrip = buttons_dict['rightGrip']
        
        # joysticks
        self.leftJS = buttons_dict['leftJS']
        self.rightJS = buttons_dict['rightJS']

class OculusTeleopSystem(LeafSystem):
    def __init__(self, oculus: OculusSystem, plant: MultibodyPlant, use_iiwa=False):
        LeafSystem.__init__(self)
        self.oculus = oculus
        
        
        self.gripper_close = 0.01 # close gripper width
        self.gripper_open  = 0.1
        self.move_kuka_mode = False
        self.base_pose = None
        self.base_controller_pose = None
        self._plant = plant
        self._plant_context = plant.CreateDefaultContext()
        
        self.trigger = False
        self.prev_trigger = False
        
        self.commanded_pose = None
        self.prev_commanded_pose = None
        
        self.current_controller_pose = None
        self.prev_controller_pose = None
        
        if use_iiwa:
            self._iiwa_instance = self._plant.GetModelInstanceByName("iiwa")
            self.kuka_q_port  = self.DeclareVectorInputPort("iiwa_position", 7)
            self.DeclareAbstractOutputPort(
                "controller_pose",
                lambda: Value(RigidTransform()),
                self.GetLeftControllerPose,
            )
            
        self.DeclareVectorOutputPort(
            "gripper_out",
            1,
            self.GetGripperOut
        )
    
    def GetGripperOut(self, context, output):
        trigger = self.oculus.rightTrig[0]
        gripper_out = self.gripper_close if trigger > 0.5 else self.gripper_open
        output.set_value(np.array([gripper_out]))
    def GetLeftControllerPose(self, context, output):
        q = self.kuka_q_port.Eval(context) # (7,)
        
        # get pose
        self._plant.SetPositions(self._plant_context, self._iiwa_instance, q)
        ee_pose = self._plant.GetFrameByName("grasp_frame").CalcPoseInWorld(self._plant_context)
        
        if self.base_pose is None: # only runs once (when no trigger is pressed)
            self.base_pose = ee_pose
        if self.base_controller_pose is None:
            self.base_controller_pose = self.oculus.right_pose
                
        trigger = self.oculus.rightGrip[0] > 0.5
        self.commanded_pose: RigidTransform = self.base_pose.__copy__()
        

        flip_axis = RigidTransform(RotationMatrix.MakeZRotation(np.pi/2)) # flip the frame that the oculus lives in such that it aligns with kuka.
        if trigger:
            
            self.current_controller_pose = self.oculus.right_pose.__copy__() @ flip_axis
            if self.prev_controller_pose is None:
                self.prev_controller_pose = self.current_controller_pose
            
            # alpha = 0.01
            # current_controller_quat = self.current_controller_pose.rotation().ToQuaternion()
            # prev_controller_quat = self.prev_controller_pose.rotation().ToQuaternion()
            # quat_filter = prev_controller_quat.slerp(alpha, current_controller_quat)
            # self.current_controller_pose.set_rotation(quat_filter)
            
            rel_current_controller2controller = self.base_controller_pose.inverse() @ self.current_controller_pose # relative
            movement_transform = rel_current_controller2controller # how much you moved relative to before trigger
            commanded_pose = self.base_pose @ movement_transform
            
            # # compare with previous commanded pose (deg error and translation error)
            # R_comm = commanded_pose.rotation().matrix()
            # R_prev = self.prev_commanded_pose.rotation().matrix()
            # angle_diff = np.rad2deg( np.arccos( (np.trace(R_prev.T @ R_comm) - 1) / 2.0 ) )
            # trans_diff = np.linalg.norm(commanded_pose.translation() - self.prev_commanded_pose.translation())
            # print("Angle Diff:", angle_diff)
            # print("Trans Diff:", trans_diff)
            # within_err_tol = (np.abs(angle_diff) < 1.0 and trans_diff < 1e-2)
            # if (not np.isnan(angle_diff)) and (not within_err_tol):
            #     print("Running")
            #     self.commanded_pose = commanded_pose
            # else:
            #     print("Ignoring")
            #     self.commanded_pose = self.prev_commanded_pose
            # print()
            
            # do recursive filter for only rotation
            # NOTE: rotation is causing our issues
            alpha = 1e-3
            quat_commanded = commanded_pose.rotation().ToQuaternion()
            quat_prev      = self.prev_commanded_pose.rotation().ToQuaternion()
            quat_filter = quat_prev.slerp(alpha, quat_commanded)
            self.commanded_pose.set_rotation(quat_filter)
            self.commanded_pose.set_translation(commanded_pose.translation())
            
            # self.commanded_pose.set_rotation(commanded_pose.rotation())
            # self.commanded_pose.set_translation(commanded_pose.translation())
            
            # self.commanded_pose = commanded_pose
            self.prev_controller_pose = self.current_controller_pose
            
        elif self.prev_trigger and not trigger: # trigger was released
            self.base_pose = ee_pose
            self.base_controller_pose = self.oculus.right_pose.__copy__() @ flip_axis
            self.prev_controller_pose = None
        else:
            self.base_controller_pose = self.oculus.right_pose.__copy__() @ flip_axis
            self.prev_controller_pose = None
            
        self.prev_trigger = trigger
        self.prev_commanded_pose = self.commanded_pose
        
        output.set_value(self.commanded_pose)

# setup pose from oculus
def setup_sim_teleop_diagram(meshcat: Meshcat):
    builder = DiagramBuilder()
    
    scenario_filename = SCENARIO_FILEPATH
    scenario = load_scenario(filename=scenario_filename, scenario_name='Demo')
    station = builder.AddNamedSystem("station", MakeHardwareStation(scenario, meshcat))
    
    plant = station.GetSubsystemByName("plant")
    scene_graph = station.GetSubsystemByName("scene_graph")
    qs = np.array([0, 40, 0, -40, 0, 100, 0]) * np.pi / 180
    
    oculus_sys = builder.AddSystem(OculusSystem())
    teleop_sys = builder.AddSystem(OculusTeleopSystem(oculus_sys, plant, use_iiwa=True))
    
    
    fake_scenario = load_scenario(filename=FAKE_SCENARIO_FILEPATH, scenario_name='Demo')
    fake_station = MakeFakeStation(fake_scenario)
    diff_ik_plant = fake_station.GetSubsystemByName("plant")
    
    differential_ik = AddIiwaDifferentialIK(builder, diff_ik_plant, diff_ik_plant.GetFrameByName("grasp_frame"))
    builder.Connect(
        differential_ik.get_output_port(),
        station.GetInputPort("iiwa.position"),
    )
    
    iiwa_state = builder.AddSystem(Multiplexer([7,7]))
    builder.Connect(
        station.GetOutputPort("iiwa.position_measured"),
        iiwa_state.get_input_port(0)
    )
    builder.Connect(
        station.GetOutputPort("iiwa.velocity_estimated"),
        iiwa_state.get_input_port(1)
    )
    builder.Connect(
        iiwa_state.get_output_port(),
        differential_ik.GetInputPort("robot_state"),
    )
    
    builder.Connect(
        station.GetOutputPort("iiwa.position_measured"),
        teleop_sys.GetInputPort("iiwa_position")
    )
    builder.Connect(
        teleop_sys.GetOutputPort("controller_pose"), differential_ik.GetInputPort("X_WE_desired")
    )
    
    builder.Connect(teleop_sys.GetOutputPort("gripper_out"), station.GetInputPort("wsg.position"))
    AddMultibodyTriad(plant.GetFrameByName("grasp_frame"), scene_graph)
    
    diagram = builder.Build()
    return diagram


def setup_teleop_diagram(meshcat):
    builder = DiagramBuilder()
    
    scenario = load_scenario(filename=SCENARIO_FILEPATH, scenario_name='Demo')
    station = builder.AddNamedSystem("station", MakeHardwareStationInterface(scenario, meshcat=meshcat))
    
    
    fake_scenario = load_scenario(filename=FAKE_SCENARIO_FILEPATH, scenario_name='Demo')
    fake_station = MakeFakeStation(fake_scenario)
    diff_ik_plant = fake_station.GetSubsystemByName("plant")
    xyz_speed_limit = 0.1
    sensor_hz = 70
    diffik_period = 1e-3
    
    differential_ik = AddIiwaDifferentialIK(
        builder,
        diff_ik_plant,
        diff_ik_plant.GetFrameByName("grasp_frame"),
        xyz_speed_limit=xyz_speed_limit,
        time_step=diffik_period
    )
    
    oculus_sys = builder.AddSystem(OculusSystem(sensor_read_hz=sensor_hz))
    teleop_sys = builder.AddSystem(OculusTeleopSystem(oculus_sys, diff_ik_plant, use_iiwa=True))
    
    builder.Connect(
        differential_ik.get_output_port(),
        station.GetInputPort("iiwa.position"),
    )
    
    # connect
    use_state = builder.AddSystem(ConstantValueSource(Value(False)))
    builder.Connect(
        use_state.get_output_port(),
        differential_ik.GetInputPort("use_robot_state"),
    )
    
    iiwa_state = builder.AddSystem(Multiplexer([7,7]))
    builder.Connect(
        station.GetOutputPort("iiwa.position_measured"),
        iiwa_state.get_input_port(0)
    )
    builder.Connect(
        station.GetOutputPort("iiwa.velocity_estimated"),
        iiwa_state.get_input_port(1)
    )
    builder.Connect(
        iiwa_state.get_output_port(),
        differential_ik.GetInputPort("robot_state"),
    )
    
    builder.Connect(
        station.GetOutputPort("iiwa.position_measured"),
        teleop_sys.GetInputPort("iiwa_position")
    )
    builder.Connect(
        teleop_sys.GetOutputPort("controller_pose"), differential_ik.GetInputPort("X_WE_desired")
    )
    builder.Connect(teleop_sys.GetOutputPort("gripper_out"), station.GetInputPort("wsg.position"))
    diagram = builder.Build()
    return diagram