from pydrake.all import (
    LeafSystem,
    RigidTransform,
    Value,
    DiagramBuilder,
    MultibodyPlant,
    Meshcat,
    RotationMatrix,
    Multiplexer,
    RotationMatrix,
    ConstantValueSource,
    PiecewisePolynomial,
    TrajectorySource,
    PiecewisePose,
    ConstantVectorSource,
    Simulator
)
from manipulation.scenarios import AddMultibodyTriad
from teleop_utils import MakeHardwareStation, AddIiwaDifferentialIK, MakeFakeStation
from oculus_drake import SCENARIO_FILEPATH, FAKE_SCENARIO_FILEPATH, SCENARIO_NO_WSG_FILEPATH, FAKE_CALIB_SCENARIO_FILEPATH, CALIB_SCENARIO_FILEPATH
from manipulation.station import load_scenario, MakeHardwareStationInterface
from oculus_reader.reader import OculusReader
from enum import Enum
import numpy as np
import time

from oculus_drake.realsense.cameras import Cameras
import os
import multiprocessing as mp
import cv2

class OculusSystem(LeafSystem):
    def __init__(self, sensor_read_hz = 60.0, save_cams = False):
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
        
        self.save_cams = save_cams        
            
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
        
        
        self.gripper_close = 0.0 # close gripper width
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
    
    scenario = load_scenario(filename=SCENARIO_FILEPATH, scenario_name='Demo')
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

class KukaEndEffectorPose(LeafSystem):
    def __init__(self, hardware_plant: MultibodyPlant, kuka_frame_name = 'iiwa_link_7'):
        LeafSystem.__init__(self)
        self._plant = hardware_plant
        self._plant_context = hardware_plant.CreateDefaultContext()
        self._frame_name = kuka_frame_name
        
        self.DeclareVectorInputPort("kuka_q", 7)
        #make abstract port for kuka pose
        self.DeclareAbstractOutputPort("kuka_pose", lambda: Value(RigidTransform()), self.CalcOutput)
        
    def CalcOutput(self, context, output):
        q = self.get_input_port().Eval(context)
        self._plant.SetPositions(self._plant_context, q)
        pose = self._plant.GetFrameByName(self._frame_name).CalcPoseInWorld(self._plant_context)
        output.set_value(pose)

def setup_teleop_diagram(meshcat):
    builder = DiagramBuilder()
    
    scenario = load_scenario(filename=SCENARIO_FILEPATH, scenario_name='Demo')
    station = builder.AddNamedSystem("station", MakeHardwareStationInterface(scenario, meshcat=meshcat))
    
    
    fake_scenario = load_scenario(filename=FAKE_SCENARIO_FILEPATH, scenario_name='Demo')
    fake_station = MakeFakeStation(fake_scenario)
    diff_ik_plant = fake_station.GetSubsystemByName("plant")
    xyz_speed_limit = 0.1
    sensor_hz = 70
    diffik_period = 1e-4
    
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
    
    # export things we want to record
    builder.ExportOutput( station.GetOutputPort("iiwa.position_measured"), "iiwa.position_measured" )
    builder.ExportOutput( teleop_sys.GetOutputPort("controller_pose"), "X_WE_desired")
    builder.ExportOutput( teleop_sys.GetOutputPort("gripper_out"), "gripper_out")
    builder.ExportOutput( station.GetOutputPort("wsg.state_measured"), "wsg.state_measured")
    builder.ExportOutput( station.GetOutputPort("iiwa.position_commanded"), "iiwa.position_commanded")
    
    diagram = builder.Build()
    return diagram

class ReplayType(Enum):
    JOINT_COMMANDS       = 0
    EE_POSE_COMMANDS     = 1
    EE_VELOCITY_COMMANDS = 2

class TeleopSequenceDataset:
    def __init__(self, dataset_dir: str):
        self.dataset_dir = dataset_dir
        
        self.joints = np.load(os.path.join(self.dataset_dir, 'joints.npy')) # (N, 7)
        self.joints_commanded = np.load(os.path.join(self.dataset_dir, 'joints_commanded.npy')) # (N, 7)
        self.commanded_gripper_pos = np.load(os.path.join(self.dataset_dir, 'gripper_out.npy')) # (N, 1)
        self.gripper_pos = np.load(os.path.join(self.dataset_dir, 'gripper_pos.npy')) # (N, 1)
        self.diffik_commanded_pose = np.load(os.path.join(self.dataset_dir, 'diffik_out.npy')) # (N, 4, 4)
        self.ts = np.load(os.path.join(self.dataset_dir, 'ts.npy')) # (N,)
        
    def __getitem__(self, index):
        data = {
            "q": self.joints[index],
            "q_command": self.joints_commanded[index],
            "g_command": self.commanded_gripper_pos[index],
            "g": self.gripper_pos[index],
            "X_WE_command": self.diffik_commanded_pose[index],
            "t": self.ts[index]
        }
        return data
    def __len__(self):
        return len(self.ts)

class PoseTrajectorySource(LeafSystem):
    def __init__(self, piecewise_pose: PiecewisePose):
        LeafSystem.__init__(self)
        self.piecewise_pose = piecewise_pose
        self.DeclareAbstractOutputPort("X_WE_desired", lambda: Value(RigidTransform()), self.CalcOutput)
    def CalcOutput(self, context, output):
        time = context.get_time()
        output.set_value(self.piecewise_pose.GetPose(time))

def setup_replay_diagram(meshcat, datapath: str, replay_type: ReplayType = ReplayType.JOINT_COMMANDS):
    # load data
    data_sequence = TeleopSequenceDataset(datapath)
    
    builder = DiagramBuilder()
    
    scenario = load_scenario(filename=SCENARIO_FILEPATH, scenario_name='Demo')
    station = builder.AddNamedSystem("station", MakeHardwareStationInterface(scenario, meshcat=meshcat))
    ts = data_sequence.ts
    if replay_type == ReplayType.JOINT_COMMANDS:
        qs_commanded = data_sequence.joints_commanded
        traj = PiecewisePolynomial.FirstOrderHold(ts, qs_commanded.T)
        traj_block = builder.AddSystem(TrajectorySource(traj))
        
        builder.Connect(
            traj_block.get_output_port(),
            station.GetInputPort("iiwa.position")
        )
        
        commanded_pose = data_sequence.diffik_commanded_pose
    elif replay_type == ReplayType.EE_POSE_COMMANDS:
        commanded_pose = data_sequence.diffik_commanded_pose
        
        fake_scenario = load_scenario(filename=FAKE_SCENARIO_FILEPATH, scenario_name='Demo')
        fake_station = MakeFakeStation(fake_scenario)
        diff_ik_plant = fake_station.GetSubsystemByName("plant")
        xyz_speed_limit = 0.1
        diffik_period = 1e-4
        
        differential_ik = AddIiwaDifferentialIK(
            builder,
            diff_ik_plant,
            diff_ik_plant.GetFrameByName("grasp_frame"),
            xyz_speed_limit=xyz_speed_limit,
            time_step=diffik_period
        )
        seq_rigid_transform = [RigidTransform(commanded_pose[i]) for i in range(len(commanded_pose))]
        X_WE_traj = PiecewisePose.MakeLinear(data_sequence.ts, seq_rigid_transform)
        X_WE_traj_block = builder.AddSystem(PoseTrajectorySource(X_WE_traj))
        
        builder.Connect(
            X_WE_traj_block.get_output_port(),
            differential_ik.GetInputPort("X_WE_desired")
        )
        
        # don't use robot state
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
            differential_ik.get_output_port(),
            station.GetInputPort("iiwa.position"),
        )
        
    else:
        raise NotImplementedError("Only joint commands are supported for now.")
    
    # connect gripper
    gripper_commanded = data_sequence.commanded_gripper_pos
    assert np.all(gripper_commanded >= 0.0), f"Gripper command must be strictly non-negative.\n {gripper_commanded}"
    traj_gripper = PiecewisePolynomial.FirstOrderHold(ts, gripper_commanded.T)
    traj_gripper_block = builder.AddSystem(TrajectorySource(traj_gripper))
    
    builder.Connect(
        traj_gripper_block.get_output_port(),
        station.GetInputPort("wsg.position")
    )
    
    diagram = builder.Build()
    end_time = data_sequence.ts[-1]
    return diagram, end_time

def set_kuka_joints(goal_q: np.ndarray, endtime = 10.0, joint_speed = None, pad_time = 2.0, use_mp = False):
    def set_kuka_joints_fn(goal_q: np.ndarray, endtime = 10.0, joint_speed = None, pad_time = 2.0):
        builder = DiagramBuilder()
        scenario = load_scenario(filename=SCENARIO_NO_WSG_FILEPATH, scenario_name='Demo')
        
        station = MakeHardwareStationInterface(scenario)
        # get curr_q
        station_context = station.CreateDefaultContext()
        station.ExecuteInitializationEvents(station_context)
        curr_q = station.GetOutputPort("iiwa.position_measured").Eval(station_context)
        
        if joint_speed is not None:
            max_dq = np.max(np.abs(goal_q - curr_q))
            endtime_from_speed = max_dq / joint_speed # joint_speed is in rad/s
            endtime = max(endtime_from_speed, 5.0) # at least 5 seconds
        
        ts = np.array([0.0, endtime])
        qs = np.array([curr_q, goal_q])
        traj = PiecewisePolynomial.FirstOrderHold(ts, qs.T)
        traj_block = builder.AddSystem(TrajectorySource(traj))
        
        station_block = builder.AddNamedSystem("station", station)
        builder.Connect(
            traj_block.get_output_port(),
            station_block.GetInputPort("iiwa.position")
        )
        diagram = builder.Build()
        
        simulator = Simulator(diagram)
        simulator.set_target_realtime_rate(1.0)
        simulator.AdvanceTo(endtime + pad_time)
    if use_mp:
        proc = mp.Process(target=set_kuka_joints_fn, args=(goal_q, endtime, joint_speed, pad_time))
        proc.start()
        proc.join()
    else:
        set_kuka_joints_fn(goal_q, endtime, joint_speed, pad_time)

def get_kuka_pose(scenario_filepath=SCENARIO_FILEPATH, fake_scenario_filepath=FAKE_CALIB_SCENARIO_FILEPATH, frame_name='iiwa_link_7', use_mp=False):
    def get_kuka_pose_fn(scenario_filepath=SCENARIO_FILEPATH, fake_scenario_filepath=FAKE_CALIB_SCENARIO_FILEPATH, frame_name='iiwa_link_7'):
        scenario = load_scenario(filename=scenario_filepath, scenario_name='Demo')
        station = MakeHardwareStationInterface(scenario)
        station_context = station.CreateDefaultContext()
        station.ExecuteInitializationEvents(station_context)
        curr_q = station.GetOutputPort("iiwa.position_measured").Eval(station_context)
        
        fake_scenario = load_scenario(filename=fake_scenario_filepath, scenario_name='Demo')
        fake_station = MakeFakeStation(fake_scenario)
        fake_plant = fake_station.GetSubsystemByName("plant")
        fake_plant_context = fake_plant.CreateDefaultContext()
        fake_plant.SetPositions(fake_plant_context, fake_plant.GetModelInstanceByName("iiwa"), curr_q)
        # get pose
        pose = fake_plant.GetFrameByName(frame_name).CalcPoseInWorld(fake_plant_context)
        return pose
    def fn(q: mp.Queue, scenario_filepath=SCENARIO_FILEPATH, fake_scenario_filepath=FAKE_CALIB_SCENARIO_FILEPATH, frame_name='iiwa_link_7'):
        q.put(get_kuka_pose_fn(scenario_filepath, fake_scenario_filepath, frame_name))
    if use_mp:
        q = mp.Queue()
        proc = mp.Process(target=fn, args=(q, scenario_filepath, fake_scenario_filepath, frame_name))
        proc.start()
        proc.join()
        return q.get()
    else:
        return get_kuka_pose_fn(scenario_filepath, fake_scenario_filepath, frame_name)
########### RECORD CODE ###########

# AsyncWriter
class CameraRecorder(LeafSystem):
    _obs_queue = mp.Queue()
    
    def __init__(self, save_folder, fps=15.0):
        LeafSystem.__init__(self)
        self.fps = fps
        self.save_folder = save_folder
        self.DeclarePeriodicPublishEvent(period_sec=1.0/self.fps, offset_sec=0.0, publish=self.write)
        self.start()
        
    def start(self):    
        self.process_save = mp.Process(target=CameraRecorder.camera_async_write, args=(self.save_folder,))
        self.process_save.start()
        time.sleep(5.0)
    def end(self):
        _obs_queue = CameraRecorder._obs_queue
        _obs_queue.put(None)
        time.sleep(5.0)
        # self.process_save.join() # for some reason this hangs. apprently a mp.Queue() issue
    def write(self, context):
        _obs_queue = CameraRecorder._obs_queue
        _obs_queue.put(5)
    
    @staticmethod
    def camera_async_write(save_folder):
        _obs_queue = CameraRecorder._obs_queue
        cameras = Cameras(
            WH=[640, 480],
            capture_fps=15,
            obs_fps=30,
            n_obs_steps=1,
            enable_color=True,
            enable_depth=True,
            process_depth=True,
        )
        cameras.start(exposure_time=10)
        for i in range(cameras.n_fixed_cameras):
            os.makedirs(f'{save_folder}/camera_{i}/', exist_ok=True)
        trigger = 5
        index = 0
        while trigger is not None:
            if not _obs_queue.empty():
                trigger = _obs_queue.get()
                recent_obs = cameras.get_obs(get_color=True, get_depth=True)
                if trigger is None:
                    break
                for i in range(cameras.n_fixed_cameras):
                    cv2.imwrite(f'{save_folder}/camera_{i}/color_{ "{:04d}".format(index) }.png', recent_obs[f'color_{i}'][-1])
                    cv2.imwrite(f'{save_folder}/camera_{i}/depth_{ "{:04d}".format(index) }.png', (recent_obs[f'depth_{i}'][-1]* 1000.0).astype(np.uint16) )
                index += 1
        
        while not _obs_queue.empty():
            _obs_queue.get()
        print("Thread done!")

class KukaRecorder(LeafSystem):
    def __init__(self, save_folder, hz=15.0):
        LeafSystem.__init__(self)
        self.save_folder = save_folder
        self.q_port = self.DeclareVectorInputPort("joints", 7)
        self.diffik_out_port = self.DeclareAbstractInputPort("X_WE_desired", Value(RigidTransform()))
        self.gripper_out_port = self.DeclareVectorInputPort("gripper_command", 1)
        self.gripper_pos_port = self.DeclareVectorInputPort("gripper_state", 2)
        self.joint_commanded_port = self.DeclareVectorInputPort("iiwa.position_commanded", 7)
        self.DeclarePeriodicPublishEvent(period_sec=1.0/hz, offset_sec=0.0, publish=self.record)
        self.ts = []
        self.joints_list = []
        self.diffik_out = []
        self.gripper_list = []
        self.gripper_pos_list = []
        self.joints_commanded_list = []
    def record(self, context):
        q = self.q_port.Eval(context)
        q_commanded = self.joint_commanded_port.Eval(context)
        X_WE_desired = self.diffik_out_port.Eval(context)
        gripper_command = self.gripper_out_port.Eval(context)
        gripper_state = self.gripper_pos_port.Eval(context)
        self.joints_list.append(q)
        self.diffik_out.append(X_WE_desired.GetAsMatrix4())
        self.gripper_list.append(gripper_command)
        self.gripper_pos_list.append(gripper_state[0])
        self.ts.append(context.get_time())
        self.joints_commanded_list.append(q_commanded)
    def save(self):
        np.save(os.path.join(self.save_folder, 'joints.npy'), np.array(self.joints_list))
        np.save(os.path.join(self.save_folder, 'diffik_out.npy'), np.array(self.diffik_out))
        np.save(os.path.join(self.save_folder, 'gripper_out.npy'), np.array(self.gripper_list))
        np.save(os.path.join(self.save_folder, 'ts.npy'), np.array(self.ts))
        np.save(os.path.join(self.save_folder, 'gripper_pos.npy'), np.array(self.gripper_pos_list))
        np.save(os.path.join(self.save_folder, 'joints_commanded.npy'), np.array(self.joints_commanded_list))
    
def record_teleop_diagram(meshcat, save_folder, fps):
    builder = DiagramBuilder()
    teleop = setup_teleop_diagram(meshcat)
    teleop_block = builder.AddSystem(teleop)
    camera_recorder = CameraRecorder(save_folder, fps)
    camera_recorder_block = builder.AddSystem(camera_recorder)
    kuka_recorder   = KukaRecorder(save_folder, fps)
    kuka_recorder_block   = builder.AddSystem(kuka_recorder)
    
    builder.Connect(
        teleop_block.GetOutputPort("iiwa.position_measured"), kuka_recorder.GetInputPort("joints")
    )
    builder.Connect(
        teleop_block.GetOutputPort("X_WE_desired"), kuka_recorder_block.GetInputPort("X_WE_desired")
    )
    builder.Connect(
        teleop_block.GetOutputPort("gripper_out"), kuka_recorder.GetInputPort("gripper_command")
    )
    builder.Connect(
        teleop_block.GetOutputPort("wsg.state_measured"), kuka_recorder.GetInputPort("gripper_state")
    )
    builder.Connect(
        teleop_block.GetOutputPort("iiwa.position_commanded"), kuka_recorder.GetInputPort("iiwa.position_commanded")
    )
    
    
    diagram = builder.Build()
    
    def _end_record_fn():
        kuka_recorder.save()
        camera_recorder.end()
        print("Recording done!")
    
    return diagram, _end_record_fn