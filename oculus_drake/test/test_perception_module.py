import sys
sys.path.append('./')
import time
import numpy as np
from perception.perception3d_module import Perception3DModule
import open3d as o3d
from oculus_drake.realsense.cameras import Cameras, load_extrinsics, load_intrinsics
import torch

if __name__ == '__main__':
    cameras = Cameras(
        WH=[640, 480],
        capture_fps=15,
        obs_fps=30,
        n_obs_steps=2,
        enable_color=True,
        enable_depth=True,
        process_depth=False,
    )
    cameras.start(exposure_time=10)
    
    time.sleep(5)
    
    obs = cameras.get_obs(get_depth=True, get_color=True)
    perception = Perception3DModule()
    
    n_cams = cameras.n_fixed_cameras
    
    colors = [obs[f'color_{i}'][-1] for i in range(n_cams)]
    depths = [obs[f'depth_{i}'][-1] for i in range(n_cams)]
    # intrinsics = cameras.get_intrinsics()
    intrinsics = load_intrinsics('calibration/camera_intrinsics.json')
    intrinsics = np.array([intrinsics[f'cam{i}'] for i in range(n_cams)])
    extrinsics = load_extrinsics('calibration/camera_extrinsics.json')
    extrinsics = np.array([extrinsics[f'cam{i}'] for i in range(n_cams)])
    pts3d, ptsrgb = perception.get_pcd(colors, depths, intrinsics, extrinsics, object_names=['red mug'])
    # pts3d, ptsrgb = perception.get_scene_pcd(colors, depths, intrinsics, extrinsics)
    # o3d vis
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts3d)
    pcd.colors = o3d.utility.Vector3dVector(ptsrgb[:,::-1] / 255.0)
    
    # crop pcd by bbox
    bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound=(-1.0, -1.0, -0.5), max_bound=(1.0, 1.0, 0.5))
    pcd = pcd.crop(bbox)
    
    o3d.visualization.draw_geometries([pcd])
    cameras.stop()