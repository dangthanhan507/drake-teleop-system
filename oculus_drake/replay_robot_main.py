import numpy as np
from pydrake.geometry import StartMeshcat
from pydrake.systems.analysis import Simulator
from pydrake.systems.framework import DiagramBuilder
from oculus_drake.oculus_drake_lib import setup_replay_diagram, set_kuka_joints
import argparse
from oculus_drake.oculus_drake_lib import ReplayType
from oculus_drake import HOME_Q
#NOTE: run this to record joint positions of robot for calibration

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--save_file", type=str, required=True)
    args = argparser.parse_args()
    
    input("Press Enter to set Kuka to home position...")
    MAX_JOINT_SPEED = 20.0 * np.pi / 180
    set_kuka_joints(HOME_Q, endtime = 5.0, joint_speed=MAX_JOINT_SPEED, use_mp=True)
    
    input("Press Enter to start replay!")
    meshcat = StartMeshcat()
    replay_diagram, end_time = setup_replay_diagram(meshcat, args.save_file, ReplayType.JOINT_COMMANDS)
    simulator = Simulator(replay_diagram)
    simulator_context = simulator.get_mutable_context()
    simulator.set_target_realtime_rate(1.0)
    
    simulator.AdvanceTo(end_time + 2.0)
    input("Done")