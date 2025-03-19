from oculus_drake.oculus_drake_lib import record_teleop_diagram, set_kuka_joints
import argparse
from pydrake.all import (
    Simulator,
    StartMeshcat
)
import numpy as np
from oculus_drake import HOME_Q
if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--save_folder', type=str, default='dataset_bottle_flip/test')
    args = argparser.parse_args()
    meshcat = StartMeshcat()
    
    input("Press Enter to set Kuka to home position...")
    MAX_JOINT_SPEED = 20.0 * np.pi / 180
    set_kuka_joints(HOME_Q, endtime = 5.0, joint_speed=MAX_JOINT_SPEED, use_mp=True)
    
    input("Press Enter to start Demo!")
    record_diagram, exit_fn = record_teleop_diagram(meshcat, args.save_folder, fps=30)
    simulator = Simulator(record_diagram)
    simulator_context = simulator.get_mutable_context()
    diagram_context = record_diagram.GetMutableSubsystemContext(record_diagram, simulator_context)
    simulator.set_target_realtime_rate(1.0)
    meshcat.AddButton("Stop Simulation", "Escape")
    try:
        while meshcat.GetButtonClicks("Stop Simulation") < 1:
            simulator.AdvanceTo(simulator.get_context().get_time() + 2.0)
    except KeyboardInterrupt:
        pass
    
    # print "demo is done" in green color using escape chars
    print("\033[92mDemo is done\033[0m")
    exit_fn() # saves everything
    # print writing is done in green color using escape chars
    print("\033[92mWriting is done\033[0m")