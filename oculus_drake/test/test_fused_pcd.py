from oculus_drake.realsense.cameras import Cameras, depth2pcd, load_extrinsics
from pydrake.all import (
    RigidTransform
)
import open3d as o3d
import argparse
import numpy as np
if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--extrinsics_path', type=str, default='calibration/camera_extrinsics.json')
    args = argparser.parse_args()
    cameras = Cameras(
        WH=[640, 480],
        capture_fps=15,
        obs_fps=30,
        n_obs_steps=2,
        enable_color=True,
        enable_depth=True,
        process_depth=True,
    )
    cameras.start(exposure_time=10)
    n_cam = cameras.n_fixed_cameras
    device = 'cuda:0' if o3d.core.cuda.is_available() else 'cpu:0'
    o3d_device = o3d.core.Device(device)
    
    #convert intrinsics to hmap
    intrinsics_arr = cameras.get_intrinsics()
    intrinsics_o3d = {}
    for i in range(intrinsics_arr.shape[0]):
        # fx = intrinsics_arr[i][0,0]
        # fy = intrinsics_arr[i][1,1]
        # cx = intrinsics_arr[i][0,2]
        # cy = intrinsics_arr[i][1,2]
        # intrinsics[f'cam{i}'] = o3d.camera.PinholeCameraIntrinsic(width=640, height=480, fx=fx, fy=fy, cx=cx, cy=cy)
        intrinsics_o3d[f'cam{i}'] = o3d.core.Tensor(intrinsics_arr[i], dtype=o3d.core.Dtype.Float32, device=o3d_device)
    
    extrinsics = load_extrinsics(args.extrinsics_path)
    extrinsics_o3d = {}
    for key in extrinsics.keys():
        extrinsics_o3d[key] = o3d.core.Tensor(extrinsics[key], dtype=o3d.core.Dtype.Float32, device=o3d_device)
    
    global_pcd = o3d.geometry.PointCloud()
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name='Fused PCL')
    vis.add_geometry(global_pcd)
    
    world_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
    
    for key in extrinsics.keys():
        extrinsic_mat = extrinsics[key]
        cam_pose = np.linalg.inv(extrinsic_mat)
        cam_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        cam_frame.transform(cam_pose)
        vis.add_geometry(cam_frame)
    
    while 1:
        try:
            obs = cameras.get_obs(get_color=True, get_depth=True)
            new_pcd = o3d.geometry.PointCloud()
            for i in range(n_cam):
                color = obs[f'color_{i}'][-1]
                depth = obs[f'depth_{i}'][-1]
                pts3d, ptscolor = depth2pcd(depth, intrinsics_arr[i], rgb=color)
                
                cam_pose = np.linalg.inv(extrinsics[f'cam{i}'])
                R = cam_pose[:3,:3]
                t = cam_pose[:3,3]
                
                pts3d = np.dot(R, pts3d.T).T + t
                
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(pts3d)
                pcd.colors = o3d.utility.Vector3dVector(ptscolor / 255)
                new_pcd += pcd
            global_pcd.points = new_pcd.points
            global_pcd.colors = new_pcd.colors
            
            global_pcd = global_pcd.voxel_down_sample(voxel_size=0.005)
            
            vis.update_geometry(global_pcd)
            vis.poll_events()
            vis.update_renderer()
            
        except KeyboardInterrupt:
            vis.destroy_window()
            cameras.stop()
            break