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
)
from manipulation import ConfigureParser
from oculus_drake import PACKAGE_XML, SCENARIO_FILEPATH
from manipulation.station import load_scenario, MakeHardwareStationInterface, MakeHardwareStation
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
        self.left_pose = RigidTransform(transform_dict['l'])
        self.right_pose = RigidTransform(transform_dict['r'])
        
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
    def __init__(self, oculus: OculusSystem):
        LeafSystem.__init__(self)
        self.oculus = oculus
        
        
        self.gripper_close = 0.03 # close gripper width
        self.gripper_open  = 0.1
        self.DeclareVectorOutputPort("gripper_out", 1, self.GetGripperOut)
    def GetGripperOut(self, context, output):
        trigger = self.oculus.rightTrig[0]
        gripper_out = self.gripper_close if trigger > 0.5 else self.gripper_open
        output.set_value(np.array([gripper_out]))

# setup pose from oculus
def setup_sim_teleop_diagram(meshcat: Meshcat):
    builder = DiagramBuilder()
    oculus = builder.AddSystem(OculusSystem())
    
    multibody_config = MultibodyPlantConfig()
    plant_scene_graph_tuple = AddMultibodyPlant(multibody_config, builder)
    plant : MultibodyPlant = plant_scene_graph_tuple[0]
    scene_graph : SceneGraph = plant_scene_graph_tuple[1]
    
    parser = Parser(plant, scene_graph)
    ConfigureParser(parser)
    parser.package_map().AddPackageXml(PACKAGE_XML)
    
    scenario_filename = SCENARIO_FILEPATH
    scenario = load_scenario(filename=scenario_filename, scenario_name='Demo')
    
    directives = ModelDirectives(directives=scenario.directives)
    added_models = ProcessModelDirectives(directives=directives, parser=parser) # list of model instances
    
    plant.Finalize()
    
    controller_block = builder.AddSystem(InverseDynamicsController(plant, kp=100*np.ones(9), ki=1*np.ones(9), kd=20*np.ones(9), has_reference_acceleration=False))
    
    qs_zero = builder.AddSystem(ConstantVectorSource(np.zeros(9) + 45 * np.pi / 180.0))
    zero_vel_block   = builder.AddSystem(ConstantVectorSource(np.zeros(9)))
    multiplexer_block = builder.AddSystem(Multiplexer([9,9]))
    
    builder.Connect(qs_zero.get_output_port(), multiplexer_block.get_input_port(0))
    builder.Connect(zero_vel_block.get_output_port(), multiplexer_block.get_input_port(1))
    
    iiwa_instance = plant.GetModelInstanceByName('iiwa')
    builder.Connect(multiplexer_block.get_output_port(), controller_block.get_input_port_desired_state())
    builder.Connect(plant.get_state_output_port(), controller_block.get_input_port_estimated_state())
    builder.Connect(controller_block.get_output_port(0), plant.get_actuation_input_port())
    
    AddDefaultVisualization(builder, meshcat)
    
    oculus_sys = builder.AddSystem(OculusSystem())
    teleop_sys = builder.AddSystem(OculusTeleopSystem(oculus_sys))
    
    
    
    diagram = builder.Build()
    return diagram



def setup_teleop_diagram(meshcat):
    builder = DiagramBuilder()
    
    scenario_data = """
    directives:
      - add_model:
          name: wsg
          file: package://manipulation/hydro/schunk_wsg_50_with_tip.sdf
      - add_weld:
          parent: world
          child: wsg::body
          X_PC:
              translation: [0, 0, 0.09]
              rotation: !Rpy { deg: [90, 0, 90]}
    model_drivers:
        wsg: !SchunkWsgDriver {}
    """
    scenario = load_scenario(data=scenario_data)
    station = MakeHardwareStationInterface(scenario, meshcat=meshcat)
    builder.AddSystem(station)
    oculus_sys = builder.AddSystem(OculusSystem())
    teleop_sys = builder.AddSystem(OculusTeleopSystem(oculus_sys))
    
    builder.Connect(teleop_sys.GetOutputPort("gripper_out"), station.GetInputPort("wsg.position"))
    diagram = builder.Build()
    return diagram