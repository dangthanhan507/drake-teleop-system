from oculus_drake_lib import setup_teleop_diagram, setup_sim_teleop_diagram
from pydrake.all import (
    Simulator,
    StartMeshcat
)
import numpy as np

if __name__ == '__main__':
    sim = False
    
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
        
    
    simulator.Initialize()
    simulator.set_target_realtime_rate(1.0)
    simulator.AdvanceTo(np.inf)