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
    SourceId
)
from manipulation.scenarios import AddMultibodyTriad
from oculus_reader import OculusReader

class OculusSystem(LeafSystem):
    def __init__(self, sensor_read_hz = 60.0):
        LeafSystem.__init__(self)
        self.reader = OculusReader(print_FPS=False)
        
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
            
        self.DeclarePeriodicPublishEvent(period_sec=1.0/sensor_read_hz, offset_sec=0.0, publish=self.oculus_read)
        self.DeclareAbstractOutputPort("left_hand_pose", lambda: Value(RigidTransform()), self.StreamLeftPose)
    def oculus_read(self):
        transform_dict, buttons_dict = self.reader.get_transformations_and_buttons()
        
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
    def StreamLeftPose(self, context, output):
        output.set_value(self.left_pose)
    def StreamRightPose(self, context, output):
        output.set_value(self.right_pose)
        

def AddAnchoredTriad(
    source_id: SourceId,
    scene_graph: SceneGraph,
    length=0.25,
    radius=0.01,
    opacity=1.0,
    X_FT=RigidTransform(),
    name="frame",
):
    X_TG = RigidTransform(RotationMatrix.MakeYRotation(np.pi / 2), [length / 2.0, 0, 0])
    geom = GeometryInstance(
        X_FT.multiply(X_TG), Cylinder(radius, length), name + " x-axis"
    )
    geom.set_illustration_properties(
        MakePhongIllustrationProperties([1, 0, 0, opacity])
    )
    scene_graph.RegisterAnchoredGeometry(source_id, geom)

    # y-axis
    X_TG = RigidTransform(RotationMatrix.MakeXRotation(np.pi / 2), [0, length / 2.0, 0])
    geom = GeometryInstance(
        X_FT.multiply(X_TG), Cylinder(radius, length), name + " y-axis"
    )
    geom.set_illustration_properties(
        MakePhongIllustrationProperties([0, 1, 0, opacity])
    )
    scene_graph.RegisterAnchoredGeometry(source_id, geom)

    # z-axis
    X_TG = RigidTransform([0, 0, length / 2.0])
    geom = GeometryInstance(
        X_FT.multiply(X_TG), Cylinder(radius, length), name + " z-axis"
    )
    geom.set_illustration_properties(
        MakePhongIllustrationProperties([0, 0, 1, opacity])
    )
    scene_graph.RegisterAnchoredGeometry(source_id, geom)
        

# setup pose from oculus
def setup_sim_teleop_diagram():
    builder = DiagramBuilder()
    oculus = builder.AddSystem(OculusSystem())
    
    
    diagram = builder.Build()
    return diagram