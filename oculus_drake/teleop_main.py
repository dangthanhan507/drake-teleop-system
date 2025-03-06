import numpy as np
from pydrake.all import (
    StartMeshcat,
    DiagramBuilder,
    Simulator,
    ConstantVectorSource,
    ConstantValueSource,
    Value
)

from manipulation.exercises.grader import Grader
from manipulation.exercises.robot.test_hardware_station_io import TestHardwareStationIO
from manipulation.station import load_scenario, MakeHardwareStationInterface

if __name__ == '__main__':
    meshcat = StartMeshcat()
    scenario_data = """
    directives:
      - add_model:
          name: iiwa
          file: package://drake/manipulation/models/iiwa_description/sdf/iiwa14_no_collision.sdf
      - add_weld:
          parent: world
          child: iiwa::iiwa_link_0
      - add_model:
          name: wsg
          file: package://manipulation/hydro/schunk_wsg_50_with_tip.sdf
      - add_weld:
          parent: iiwa::iiwa_link_7
          child: wsg::body
          X_PC:
              translation: [0, 0, 0.09]
              rotation: !Rpy { deg: [90, 0, 90]}
    model_drivers:
        iiwa: !IiwaDriver
            control_mode: position_and_torque
        wsg: !SchunkWsgDriver {}
    """
    builder = DiagramBuilder()
    scenario = load_scenario(data=scenario_data)
    station = MakeHardwareStationInterface(scenario, meshcat=meshcat)
    builder.AddSystem(station)
    
    q_fixed = np.array([0, 0, 0, 0, 0, 0, 0.0 * np.pi / 180.0])
    fixed_pos    = builder.AddSystem(ConstantVectorSource(q_fixed))
    fixed_torque = builder.AddSystem(ConstantVectorSource(np.zeros(7)))
    fixed_gripper_pos = builder.AddSystem(ConstantVectorSource(np.zeros(1) + 0.05))
    builder.Connect(fixed_pos.get_output_port(), station.GetInputPort("iiwa.position"))
    builder.Connect(fixed_torque.get_output_port(), station.GetInputPort("iiwa.feedforward_torque"))
    builder.Connect(fixed_gripper_pos.get_output_port(), station.GetInputPort("wsg.position"))
    
    builder.ExportOutput(station.GetOutputPort("wsg.state_measured"), "wsg.state_measured")
    builder.ExportOutput(station.GetOutputPort("iiwa.position_measured"), "iiwa.position_measured")
    diagram = builder.Build()
    
    diagram_context = diagram.CreateDefaultContext()
    diagram.ExecuteInitializationEvents(diagram_context)
    wsg_state = diagram.GetOutputPort("wsg.state_measured").Eval(diagram_context)
    q = diagram.GetOutputPort("iiwa.position_measured").Eval(diagram_context)
    print("wsg state:", wsg_state)
    print("q:", q * 180.0 / np.pi)
    
    simulator = Simulator(diagram)
    simulator.Initialize()
    simulator.set_target_realtime_rate(1.0)
    
    input("Press [Enter] to start the simulation.")
    simulator.AdvanceTo(10.0)
    input("Done")