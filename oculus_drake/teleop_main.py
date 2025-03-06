from oculus_drake_lib import setup_teleop_diagram, setup_sim_teleop_diagram
from pydrake.all import (
    Simulator,
    StartMeshcat
)
import numpy as np

if __name__ == '__main__':
    meshcat = StartMeshcat()
    diagram = setup_sim_teleop_diagram(meshcat)
    simulator = Simulator(diagram)
    simulator.Initialize()
    simulator.set_target_realtime_rate(1.0)
    simulator.AdvanceTo(np.inf)