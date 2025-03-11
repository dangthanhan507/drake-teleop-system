from pydrake.all import (
    DiagramBuilder,
    Simulator
)
import numpy as np
from oculus_drake.realsense.cameras import Cameras
from oculus_drake.calib_utils import CameraCalibrateVisSystem
if __name__ == '__main__':
    
    cameras = Cameras(
        WH=[640, 480],
        capture_fps=15,
        obs_fps=30,
        n_obs_steps=2,
        enable_color=True,
        enable_depth=True,
        process_depth=True
    )
    cameras.start(exposure_time=10)
    Ks = cameras.get_intrinsics()
    
    
    builder = DiagramBuilder()
    cam_calib = CameraCalibrateVisSystem(cameras, Ks, tag_width=0.04, save_data=False)
    builder.AddSystem(cam_calib)
    diagram = builder.Build()
    
    simulator = Simulator(diagram)
    simulator.Initialize()
    simulator.AdvanceTo(np.inf)