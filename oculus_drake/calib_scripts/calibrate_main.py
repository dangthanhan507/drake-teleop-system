import numpy as np
import argparse
from tqdm import tqdm
from oculus_drake.realsense.cameras import load_intrinsics
# given list of joints, follow each joint and take 10 seconds to go to each joint
import cv2
import json
import os
if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--data_folder", type=str, default="calibrate_images/")
    argparser.add_argument('--intrinsics_file', type=str, default='calibration/camera_intrinsics.json') 
    argparser.add_argument('--kuka_extrinsics_file', type=str, default='calibration/camera_extrinsics_kuka.json') #make this identity if no kuka extrinsic
    argparser.add_argument('--out_file', type=str, default='calibration/camera_extrinsics.json')
    args = argparser.parse_args()
    
    intrinsics_dict = load_intrinsics(args.intrinsics_file)
    num_cameras = len(intrinsics_dict)
    
    with open(args.kuka_extrinsics_file, 'r') as f:
        kuka_extrinsics_data = json.load(f)
    kuka_cam_id = kuka_extrinsics_data['cam_id']
    kuka2kuka_cam = np.array(kuka_extrinsics_data['extrinsics'])
    kuka_cam2kuka = np.linalg.inv(kuka2kuka_cam)
    
    MARKER_LENGTH = 0.0485
    MARKER_SEPARATION = 0.016
    arucoDict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_1000)
    # arucoDict = cv2.aruco.Dictionary(cv2.aruco.DICT_4X4_1000)
    arucoParams = cv2.aruco.DetectorParameters()
    arucoParams.adaptiveThreshWinSizeStep=40
    arucoParams.adaptiveThreshWinSizeMax=100
    arucoParams.aprilTagMinClusterPixels=1000
    aruco_grid = cv2.aruco.GridBoard((3,3), MARKER_LENGTH, MARKER_SEPARATION, arucoDict)
    # aruco_grid = cv2.aruco.GridBoard_create(3,3, MARKER_LENGTH, MARKER_SEPARATION, arucoDict)
    
    cam_paths = {}
    for i in range(num_cameras):
        cam_paths[f'cam{i}'] = os.path.join(args.data_folder, f'cam{i}')
    
    num_frames = len(os.listdir(cam_paths['cam0']))
    for frame_idx in range(num_frames):
        
        camera_idx = kuka_cam_id
        img = cv2.imread(os.path.join(cam_paths[f'cam{camera_idx}'], f'{frame_idx:04d}.png'))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = cv2.aruco.detectMarkers(gray, arucoDict, parameters=arucoParams)
        _, rvec, tvec = cv2.aruco.estimatePoseBoard(corners, ids, aruco_grid, intrinsics_dict[f'cam{camera_idx}'], np.zeros(5), None, None)
        Rot = cv2.Rodrigues(rvec)[0]
        t = tvec.flatten()                
        board2kuka_cam = np.eye(4)
        board2kuka_cam[:3,:3] = Rot
        board2kuka_cam[:3,3] = t
        
        objpoints2kuka_cam_dict = {}
        imgpoints_dict = {}
        for camera_idx in range(num_cameras):
            if camera_idx != kuka_cam_id:
                objpoints2kuka_cam_dict[f'cam{camera_idx}'] = []
                imgpoints_dict[f'cam{camera_idx}'] = []
        for camera_idx in range(num_cameras):
            if camera_idx != kuka_cam_id:
                img = cv2.imread(os.path.join(cam_paths[f'cam{camera_idx}'], f'{frame_idx:04d}.png'))
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                corners, ids, _ = cv2.aruco.detectMarkers(gray, arucoDict, parameters=arucoParams)
            
            
                objpoints2board, im_points = aruco_grid.matchImagePoints(corners, ids)
                objpoints2board = objpoints2board.squeeze()
                im_points = im_points.squeeze()
                
                # obj_points are 3d (N,3) and represent objpoints2board
                R = board2kuka_cam[:3,:3]
                t = board2kuka_cam[:3,3]
                
                objpoints2kuka_cam = np.einsum('...ij,...j->...i', R, objpoints2board) + t
                objpoints2kuka_cam_dict[f'cam{camera_idx}'].append(objpoints2kuka_cam)
                imgpoints_dict[f'cam{camera_idx}'].append(im_points)
                
        # calibrate each camera
        extrinsics_dict = {'base_cam_id': kuka_cam_id}
        for camera_idx in range(num_cameras):
            if camera_idx == kuka_cam_id:
                kuka_cam2cam = np.eye(4)
            else:
                pts2d = np.concatenate(imgpoints_dict[f'cam{camera_idx}'], axis=0) # pix in camera_i
                pts3d = np.concatenate(objpoints2kuka_cam_dict[f'cam{camera_idx}'], axis=0) # pts3d in kuka_cam frame
                K = intrinsics_dict[f'cam{camera_idx}']
                ret, rvec, tvec = cv2.solvePnP(pts3d, pts2d, K, distCoeffs=np.zeros(5)) # get camera_i to kuka_cam
                rotm = cv2.Rodrigues(rvec)[0]
                kuka_cam2cam = np.eye(4)
                kuka_cam2cam[:3, :3] = rotm
                kuka_cam2cam[:3, 3] = tvec.flatten()
            cam2kuka_cam = np.linalg.inv(kuka_cam2cam)
            cam2kuka = kuka_cam2kuka @ cam2kuka_cam
            kuka2cam = np.linalg.inv(cam2kuka)
            E = kuka2cam.tolist()
            extrinsics_dict[f'cam{camera_idx}'] = E
        # save json
        with open(args.out_file, 'w') as f:
            json.dump(extrinsics_dict, f)
        # make every camera relative to camera kuka