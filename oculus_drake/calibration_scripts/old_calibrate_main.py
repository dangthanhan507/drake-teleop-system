import numpy as np
import argparse
import pupil_apriltags
from tqdm import tqdm
from oculus_drake.realsense.cameras import Cameras
from oculus_drake.teleop.oculus_drake_lib import set_kuka_joints, get_kuka_pose
from oculus_drake import CALIB_SCENARIO_FILEPATH, FAKE_CALIB_SCENARIO_FILEPATH
# given list of joints, follow each joint and take 10 seconds to go to each joint
from collections import defaultdict
import cv2
import json
def save_extrinsics(json_dict, filename):
    with open(filename, 'w') as f:
        json.dump(json_dict, f)

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--file", type=str, default="calibration/calibration_joints.npy")
    argparser.add_argument('--out_file', type=str, default='calibration/camera_extrinsics.json')
    args = argparser.parse_args()
    
    joints = np.load(args.file) #(N, 7)
    
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
    detector = pupil_apriltags.Detector(families='tagStandard41h12')
    camera_datapoints = defaultdict(list)
    
    print(joints.shape)
    
    # print in blue starting joints run
    print("\033[94mStarting collection\033[0m")
    for i in tqdm(range(joints.shape[0])):
        joint = joints[i]
        set_kuka_joints(joint, endtime=30.0, joint_speed=5.0 * np.pi / 180.0, pad_time=2.0)
        pose = get_kuka_pose(CALIB_SCENARIO_FILEPATH, FAKE_CALIB_SCENARIO_FILEPATH, "tag_frame", use_mp=True)
        pt3d = pose.translation()
        obs = cameras.get_obs()
        
        for i in range(cameras.n_fixed_cameras):
            color = obs[f'color_{i}'][-1]
            detections = detector.detect(cv2.cvtColor(color, cv2.COLOR_BGR2GRAY))
            for detect in detections:
                if detect.tag_id == 0:
                    pt2d = detect.center
                    camera_datapoints[f'cam{i}'].append((pt2d, pt3d))
    # print in red finished joints data collection
    print("\033[91mFinished collection\033[0m")
    for i in range(cameras.n_fixed_cameras):
        print(f"cam{i} has {len(camera_datapoints[f'cam{i}'])} points")
    
    # print in blue starting camera calibration
    print("\033[94mStarting calibration\033[0m")
    Ks = cameras.get_intrinsics()
    camera_json = dict()
    for i in range(cameras.n_fixed_cameras):
        pts2d = np.zeros((len(camera_datapoints[f'cam{i}']), 2))
        pts3d = np.zeros((len(camera_datapoints[f'cam{i}']), 3))
        for j in range(len(camera_datapoints[f'cam{i}'])):
            pt2d, pt3d = camera_datapoints[f'cam{i}'][j]
            pts2d[j, :] = pt2d
            pts3d[j, :] = pt3d
        K = Ks[i]
        ret, rvec, tvec = cv2.solvePnP(pts3d, pts2d, K, distCoeffs=np.zeros(5)) # get kuka2cam pts3d is in kuka frame
        
        rotm = cv2.Rodrigues(rvec)[0]
        H = np.eye(4)
        H[:3, :3] = rotm
        H[:3, 3] = tvec.flatten()
        camera_json[f'cam{i}'] = H.tolist()
        print(np.linalg.inv(H))
    save_extrinsics(camera_json, args.out_file)
    print("\033[91mFinished calibration\033[0m")
    print("Done")