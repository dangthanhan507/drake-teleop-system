import numpy as np
from pydrake.geometry import StartMeshcat
from pydrake.systems.analysis import Simulator
from pydrake.systems.framework import DiagramBuilder
from oculus_drake.calib_scripts.calib_utils import CameraCalibrateVisSystem, CameraCalibrateVisSystemAsync
from oculus_drake.realsense.cameras import Cameras
from oculus_drake.teleop.oculus_drake_lib import setup_teleop_diagram
import argparse
#NOTE: run this to record joint positions of robot for calibration

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--camera", type=bool, default=False)
    argparser.add_argument("--save_file", type=str, required=True)
    args = argparser.parse_args()
    
    meshcat = StartMeshcat()
    meshcat.ResetRenderMode()
    builder = DiagramBuilder()
    if args.camera:
        camera_tag_pub = CameraCalibrateVisSystemAsync(fps=30)
        cam_block = builder.AddSystem(camera_tag_pub)
    
    teleop_diagram = setup_teleop_diagram(meshcat)
    teleop_block = builder.AddSystem(teleop_diagram)
    builder.ExportOutput(teleop_block.GetOutputPort("iiwa.position_measured"), "iiwa.position_measured")
    
    diagram = builder.Build()
    simulator = Simulator(diagram)
    simulator_context = simulator.get_mutable_context()
    diagram_context = diagram.GetMutableSubsystemContext(diagram, simulator_context)
    
    simulator.set_target_realtime_rate(1.0)
    
    meshcat.AddButton("Stop Simulation", "Escape")
    meshcat.AddButton("Record Joint", "KeyC")
    current_record_button_ctr = 0
    
    joints = []
    while meshcat.GetButtonClicks("Stop Simulation") < 1:
        simulator.AdvanceTo(simulator.get_context().get_time() + 2.0)
        if meshcat.GetButtonClicks("Record Joint") > current_record_button_ctr:
            current_record_button_ctr += 1
            print("Recording joint position...")
            joint_pos = diagram.GetOutputPort("iiwa.position_measured").Eval(diagram_context)
            print(joint_pos)
            joints.append(joint_pos)
            print(f"Recorded joint position: {joint_pos}.")
            print()
    meshcat.DeleteButton("Stop Simulation")
    meshcat.DeleteButton("Record Joint")
    if len(joints) > 0:
        joints = np.array(joints)
        np.save(args.save_file, joints)
    if args.camera:
        camera_tag_pub.end()
    input("Done")