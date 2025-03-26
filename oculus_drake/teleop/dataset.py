from pydrake.all import (
    RigidTransform,
)
from oculus_drake.teleop.teleop_utils import MakeFakeStation
from oculus_drake import FAKE_SCENARIO_FILEPATH
from manipulation.station import load_scenario
import numpy as np
import os

# I messed up and didn't record diffik velocity, but it's calculated implicitly by DiffIK anyways.
# I'll just offload the computation to the TeleopSequenceDataset
class TeleopSequenceDataset:
    def __init__(self, dataset_dir: str, get_V_WE = False):
        self.dataset_dir = dataset_dir
        
        self.joints = np.load(os.path.join(self.dataset_dir, 'joints.npy')) # (N, 7)
        self.joints_commanded = np.load(os.path.join(self.dataset_dir, 'joints_commanded.npy')) # (N, 7)
        self.commanded_gripper_pos = np.load(os.path.join(self.dataset_dir, 'gripper_out.npy')) # (N, 1)
        self.gripper_pos = np.load(os.path.join(self.dataset_dir, 'gripper_pos.npy')) # (N, 1)
        self.diffik_commanded_pose = np.load(os.path.join(self.dataset_dir, 'diffik_out.npy')) # (N, 4, 4)
        self.ts = np.load(os.path.join(self.dataset_dir, 'ts.npy')) # (N,)
        
        if get_V_WE:
            self.V_WEs = np.zeros((len(self), 6))
            fake_scenario = load_scenario(filename=FAKE_SCENARIO_FILEPATH, scenario_name='Demo')
            fake_station = MakeFakeStation(fake_scenario)
            fake_plant = fake_station.GetSubsystemByName("plant")
            fake_plant_context = fake_plant.CreateDefaultContext()
            for i in range(len(self)):
                X_WE_des_i = RigidTransform(self.diffik_commanded_pose[i])
                
                fake_plant.SetPositions(fake_plant_context, fake_plant.GetModelInstanceByName("iiwa"), self.joints_commanded[i])
                X_WE_curr  = fake_plant.GetFrameByName("grasp_frame").CalcPoseInWorld(fake_plant_context)
                
                diff_translation = (X_WE_des_i.translation() - X_WE_curr.translation())
                diff_rotation    = (X_WE_des_i.rotation() @ X_WE_curr.rotation().transpose()).ToAngleAxis()
                diff_rotation = diff_rotation.axis() * diff_rotation.angle()
                V_WE = np.concatenate([diff_rotation, diff_translation])
                self.V_WEs[i] = V_WE
        
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