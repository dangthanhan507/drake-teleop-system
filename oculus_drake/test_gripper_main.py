import numpy as np
from pydrake.all import (
    StartMeshcat,
    DiagramBuilder,
    Simulator,
    ConstantVectorSource,
    ConstantValueSource,
    LeafSystem,
    Value,
    PiecewisePolynomial,
    TrajectorySource
)
from manipulation.station import load_scenario, MakeHardwareStationInterface
import argparse

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--gripper_width", type=float, default=0.05)
    args = argparser.parse_args()
    gripper_width = args.gripper_width
    
    meshcat = StartMeshcat()
    scenario_data = """
    directives:
      - add_model:
          name: wsg
          file: package://manipulation/hydro/schunk_wsg_50_with_tip.sdf
      - add_weld:
          parent: world
          child: wsg::body
          X_PC:
              translation: [0, 0, 0.09]
              rotation: !Rpy { deg: [90, 0, 90]}
    model_drivers:
        wsg: !SchunkWsgDriver {}
    """
    builder = DiagramBuilder()
    scenario = load_scenario(data=scenario_data)
    station = MakeHardwareStationInterface(scenario, meshcat=meshcat)
    builder.AddSystem(station)
    
    endtime = 10.0
    ts = np.linspace(0, endtime, 100)
    widths1 = np.linspace(0.1, 0.01, 50)[np.newaxis, :]
    widths2 = np.linspace(0.01, 0.1, 50)[np.newaxis, :]
    widths = np.concatenate([widths1, widths2], axis=1)
    traj = builder.AddSystem(TrajectorySource(PiecewisePolynomial.ZeroOrderHold(ts.tolist(), widths.tolist())))
    builder.Connect(traj.get_output_port(), station.GetInputPort("wsg.position"))
    
    # fixed_gripper_pos = builder.AddSystem(ConstantVectorSource(np.zeros(1) + gripper_width))
    # builder.Connect(fixed_gripper_pos.get_output_port(), station.GetInputPort("wsg.position"))
    
    force_limit = builder.AddSystem(ConstantVectorSource(np.zeros(1) + 10.0))
    
    builder.Connect(force_limit.get_output_port(), station.GetInputPort("wsg.force_limit"))
    
    builder.ExportOutput(station.GetOutputPort("wsg.state_measured"), "wsg.state_measured")
    diagram = builder.Build()
    
    diagram_context = diagram.CreateDefaultContext()
    diagram.ExecuteInitializationEvents(diagram_context)
    wsg_state = diagram.GetOutputPort("wsg.state_measured").Eval(diagram_context)
    print("wsg state:", wsg_state)
    
    simulator = Simulator(diagram)
    # simulator.Initialize()
    simulator.set_target_realtime_rate(1.0)
    
    input("Press [Enter] to start the simulation.")
    simulator.AdvanceTo(20.0)
    input("Done")