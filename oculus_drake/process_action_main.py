import numpy as np
from oculus_drake.oculus_drake_lib import TeleopSequenceDataset
import os
from pydrake.all import (
    RigidTransform,
    AngleAxis,
    Quaternion
)
from oculus_drake import FAKE_SCENARIO_FILEPATH
from oculus_drake.teleop_utils import MakeFakeStation
from manipulation.station import load_scenario
from tqdm import tqdm
def processaction(
            teleop_data_dir: str,
        ):
        teleop_dataset = TeleopSequenceDataset(teleop_data_dir, get_V_WE=False)
        qs_command = teleop_dataset.joints_commanded
        ts = teleop_dataset.ts
        
        scenario = load_scenario(filename=FAKE_SCENARIO_FILEPATH, scenario_name='Demo')
        station = MakeFakeStation(scenario)
        plant = station.GetSubsystemByName("plant")
        plant_context = plant.CreateDefaultContext()
        
        os.makedirs(os.path.join(teleop_data_dir, 'action'), exist_ok=True)
        for i in tqdm(range(len(teleop_dataset)-1)):
            plant.SetPositions(plant_context, plant.GetModelInstanceByName("iiwa"), qs_command[i])
            grasp_frame_commanded_pose = plant.GetFrameByName("grasp_frame").CalcPoseInWorld(plant_context)
            plant.SetPositions(plant_context, plant.GetModelInstanceByName("iiwa"), qs_command[i+1])
            grasp_frame_next_commanded_pose = plant.GetFrameByName("grasp_frame").CalcPoseInWorld(plant_context)
            
            delta_commanded_grasp_frame: RigidTransform = grasp_frame_next_commanded_pose @ grasp_frame_commanded_pose.inverse()
            
            # convert this to axis-angle
            quaternion = Quaternion(delta_commanded_grasp_frame.rotation().matrix())
            axis_angle = AngleAxis(quaternion)
            tvec = delta_commanded_grasp_frame.translation()
            
            # concatenate the axis-angle and translation
            command = np.concatenate([axis_angle.angle() * axis_angle.axis(), tvec])
            np.save(os.path.join(teleop_data_dir, 'action', f'dpose_{i:04d}.npy'), command)
            

if __name__ == '__main__':
    import argparse 
    parser = argparse.ArgumentParser()
    parser.add_argument('--teleop_data_dir', type=str, default=None)
    args = parser.parse_args()
    
    processaction(
        args.teleop_data_dir,
    )