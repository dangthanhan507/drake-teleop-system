from pydrake.all import (
    MultibodyPlant,
    StartMeshcat,
    DiagramBuilder,
    LeafSystem,
    PointCloud,
    Fields,
    BaseField,
    Value,
    MeshcatPointCloudVisualizer
)
from oculus_drake import SCENARIO_FILEPATH
from manipulation.station import load_scenario, MakeHardwareStationInterface
from teleop_utils import MakeHardwareStation, AddIiwaDifferentialIK, MakeFakeStation
from oculus_drake.realsense.cameras import load_extrinsics, load_intrinsics, depth2pcd
import argparse
import os
import numpy as np
import cv2
import time

class ManualPublishPCL(LeafSystem):
    def __init__(self, load_folder, extrinsics_folder, intrinsics_folder):
        LeafSystem.__init__(self)
        self.load_folder = load_folder
        self.extrinsic_folder = extrinsics_folder
        self.intrinsic_folder = intrinsics_folder
        self.extrinsics = load_extrinsics(extrinsics_folder)
        self.intrinsics = load_intrinsics(intrinsics_folder)
        self.num_cams = sum([1 for item in os.listdir(self.load_folder) if item.startswith('camera_')])
        self.DeclareAbstractOutputPort("pcl", lambda: Value(PointCloud()), self.PublishPCL)
        self.idx = 0
        
    def PublishPCL(self, context, output):
        pts3d_total = []
        ptsrgb_total = []
        for i in range(self.num_cams):
            # get intr/extr
            extr = self.extrinsics[f'cam{i}']
            intr = self.intrinsics[f'cam{i}']
            # get rgb
            rgb_path = os.path.join(self.load_folder, f'camera_{i}', f'color_{ "{:04d}".format(self.idx) }.png')
            rgb = cv2.imread(rgb_path, cv2.IMREAD_COLOR_RGB)
            depth_path = os.path.join(self.load_folder, f'camera_{i}', f'depth_{ "{:04d}".format(self.idx) }.png')
            depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED) / 1000.0
            
            pts3d, ptsrgb = depth2pcd(depth, intr, rgb)
            # transform pts3d
            cam_pose = np.linalg.inv(extr)
            R = cam_pose[:3, :3]
            t = cam_pose[:3, 3]
            pts3d = np.dot(R, pts3d.T).T + t
            pts3d_total.append(pts3d)
            ptsrgb_total.append(ptsrgb)
        pts3d_total = np.concatenate(pts3d_total, axis=0)
        ptsrgb_total = np.concatenate(ptsrgb_total, axis=0)
        pcl = PointCloud(new_size = pts3d_total.shape[0], fields= Fields(BaseField.kXYZs | BaseField.kRGBs))
        pcl.mutable_rgbs()[:] = ptsrgb_total.T
        pcl.mutable_xyzs()[:] = pts3d_total.T
        pcl = pcl.VoxelizedDownSample(voxel_size=1e-2, parallelize=True)
        self.idx += 1
        output.set_value(pcl)
        
if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--load_folder', type=str, default='dataset_bottle_flip/test')
    argparser.add_argument('--extr_folder', type=str, default='calibration/camera_extrinsics.json')
    argparser.add_argument('--intr_folder', type=str, default='calibration/camera_intrinsics.json')
    args = argparser.parse_args()
    meshcat = StartMeshcat()
    
    builder = DiagramBuilder()
    scenario = load_scenario(filename=SCENARIO_FILEPATH, scenario_name='Demo')
    station = MakeFakeStation(scenario, meshcat)
    plant: MultibodyPlant = station.GetSubsystemByName("plant")
    
    station_block = builder.AddSystem(station)
    pcd_block = builder.AddSystem(ManualPublishPCL(args.load_folder, args.extr_folder, args.intr_folder))
    meshcat_pcl_vis = builder.AddSystem(MeshcatPointCloudVisualizer(meshcat, path="/rgb", publish_period=1e-4))
    
    builder.Connect(
        pcd_block.get_output_port(),
        meshcat_pcl_vis.get_input_port(0)
    )
    root_diagram = builder.Build()
    
    root_diagram_context = root_diagram.CreateDefaultContext()
    diagram_context = station.GetMyMutableContextFromRoot(root_diagram_context)
    plant_context = station.GetMutableSubsystemContext(plant,diagram_context)
    
    ts = np.load(os.path.join(args.load_folder, "ts.npy"))
    joints = np.load(os.path.join(args.load_folder, "joints.npy"))
    gripper_pos = np.load(os.path.join(args.load_folder, "gripper_out.npy"))
    
    # convert gripper_pos to acceptable in drake sim format
    gripper_pos = np.concatenate([-gripper_pos, gripper_pos], axis=1) / 2
    
    meshcat.StartRecording()
    for i in range(len(ts)):
        if i % 10 == 0:
            meshcat_pcl_vis.Delete()
        root_diagram_context.SetTime(ts[i])
        plant.SetPositions(plant_context, plant.GetModelInstanceByName("iiwa"), joints[i])
        plant.SetPositions(plant_context, plant.GetModelInstanceByName("wsg"), gripper_pos[i])
        root_diagram.ForcedPublish(root_diagram_context)
    meshcat.StopRecording()
    meshcat.PublishRecording()
    input("Done")