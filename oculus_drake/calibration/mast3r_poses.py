import numpy as np
import json

if __name__ == '__main__':
    
    with open('camera_extrinsics_kuka.json', 'r') as f:
        extrinsics_kuka_data = json.load(f)
    kuka2kuka_cam = np.array(extrinsics_kuka_data['extrinsics'])
    kuka_cam2kuka = np.linalg.inv(kuka2kuka_cam)
    
    # these are outputs from mast3r
    cam2worlds = np.array([[[-0.6007,  0.4052, -0.6891,  1.5499],
                            [-0.3520, -0.9080, -0.2271,  0.7765],
                            [-0.7178,  0.1061,  0.6881, -0.6722],
                            [ 0.0000,  0.0000,  0.0000,  1.0000]],

                            [[ 0.6144,  0.4446, -0.6518,  1.5862],
                            [-0.0919, -0.7802, -0.6188,  1.1116],
                            [-0.7836,  0.4401, -0.4384,  0.4998],
                            [ 0.0000,  0.0000,  0.0000,  1.0000]],

                            [[ 0.0569, -0.3021,  0.9516,  0.3320],
                            [ 0.2895, -0.9072, -0.3053,  0.6950],
                            [ 0.9555,  0.2928,  0.0359, -0.0373],
                            [ 0.0000,  0.0000,  0.0000,  1.0000]]])
    focals = np.array([489.5253, 483.5385, 456.4479])
    principal_points = np.array([[276.9317, 183.4841],
                                 [252.7770, 188.1569],
                                 [249.6548, 190.1449]])
    intrinsics = np.zeros((3,3,3))
    for i in range(3):
        intrinsics[i] = np.array([[focals[i], 0, principal_points[i,0]],
                                  [0, focals[i], principal_points[i,1]],
                                  [0, 0, 1]])
    cam02world = cam2worlds[0]
    cam12world = cam2worlds[1]
    cam22world = cam2worlds[2]
    
    cam2kuka_cams = np.zeros_like(cam2worlds)
    
    kuka_cam_idx = 2
    for idx in range(3):
        cam2kuka_cams[idx] = np.linalg.inv(cam2worlds[kuka_cam_idx]) @ cam2worlds[idx]
    
    cam2kuka = np.zeros_like(cam2worlds)
    for idx in range(3):
        cam2kuka[idx] = kuka_cam2kuka @ cam2kuka_cams[idx]
    
    # cam2kuka = cam2worlds
    
    import open3d as o3d
    
    extrinsics_dict = {}
    intrinsics_dict = {}
    for i in range(3):
        extrinsics_dict[f'cam{i}'] = np.linalg.inv(cam2kuka[i]).tolist()
        intrinsics_dict[f'cam{i}'] = intrinsics[i].tolist()
    # save
    with open('mast3r_camera_extrinsics.json', 'w') as f:
        json.dump(extrinsics_dict, f)
    with open('mast3r_camera_intrinsics.json', 'w') as f:
        json.dump(intrinsics_dict, f)
    
    # visualize origin frame and 3 camera frames
    origin_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
    origin_frame.compute_vertex_normals()
    # origin_frame.paint_uniform_color([0.5, 0.5, 0.5])
    
    cam0_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
    cam0_frame.compute_vertex_normals()
    cam0_frame.paint_uniform_color([1, 0, 0])
    cam0_frame.transform(cam2kuka[0])
    
    cam1_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
    cam1_frame.compute_vertex_normals()
    cam1_frame.paint_uniform_color([0, 1, 0])
    cam1_frame.transform(cam2kuka[1])
    
    cam2_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
    cam2_frame.compute_vertex_normals()
    cam2_frame.paint_uniform_color([0, 0, 1])
    cam2_frame.transform(cam2kuka[2])
    
    o3d.visualization.draw_geometries([origin_frame, cam0_frame, cam1_frame, cam2_frame])