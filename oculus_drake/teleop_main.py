from oculus_drake_lib import setup_teleop_diagram, setup_sim_teleop_diagram, set_kuka_joints
from pydrake.all import (
    Simulator,
    StartMeshcat
)
import numpy as np

if __name__ == '__main__':
    sim = False
    
    if not sim:
        input("Press Enter to set Kuka to home position...")
        MAX_JOINT_SPEED = 20.0 * np.pi / 180
        home_q = np.array([-90.0, 30.0, 0.0, -70.0, 0.0, 75.0, 0.0]) * np.pi / 180
        set_kuka_joints(home_q, endtime = 5.0, joint_speed=MAX_JOINT_SPEED, use_mp=True)
    
    meshcat = StartMeshcat()
    diagram = setup_sim_teleop_diagram(meshcat) if sim else setup_teleop_diagram(meshcat)
    simulator = Simulator(diagram)
    
    if sim:
        # set initial joints
        context = simulator.get_mutable_context()
        station = diagram.GetSubsystemByName("station")
        plant   = station.GetSubsystemByName("plant")
        iiwa_instance = plant.GetModelInstanceByName("iiwa")
        plant_context = plant.GetMyMutableContextFromRoot(context)
        q0 = np.array([0, 40, 0, -40, 0, 100, 0]) * np.pi / 180
        plant.SetPositions(plant_context, iiwa_instance, q0)
    
    input("Press Enter to start teleop...")
    simulator.Initialize()
    simulator.set_target_realtime_rate(1.0)
    simulator.AdvanceTo(np.inf)