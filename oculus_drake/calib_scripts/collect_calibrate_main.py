import numpy as np
import cv2
from oculus_drake.realsense.cameras import Cameras
import argparse
import os
'''
NOTE: previous calibration collection sucks. Instead, we will choose 1 camera to be in world-frame.
Then we will use an apriltag board to move around and detect the apriltags in the world-frame camera.
'''

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--out_folder", type=str, default="calibrate_images")
    args = argparser.parse_args()
    cameras = Cameras(
        WH=[640, 480],
        capture_fps=15,
        obs_fps=30,
        n_obs_steps=1,
        enable_color=True,
        enable_depth=True,
        process_depth=True,
    )
    cameras.start(exposure_time=10)
    # 4.85cm
    arucoDict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_1000)
    arucoParams = cv2.aruco.DetectorParameters()
    arucoParams.adaptiveThreshWinSizeStep=40
    arucoParams.adaptiveThreshWinSizeMax=100
    arucoParams.aprilTagMinClusterPixels=1000
    index = 0
    if os.path.exists(os.path.join(args.out_folder, "cam0")):
        index = len(os.listdir(os.path.join(args.out_folder, "cam0")))
    
    MARKER_LENGTH = 0.0485
    MARKER_SEPARATION = 0.016
    aruco_grid = cv2.aruco.GridBoard((3,3), MARKER_LENGTH, MARKER_SEPARATION, arucoDict)
    intrinsics = cameras.get_intrinsics()
    while 1:
        try:
            # input("Press Enter to save image")
            # save rgb to out_folder
            recent_obs = cameras.get_obs(get_color=True, get_depth=False)
            
            rgbs = []
            fail = False
            for i in range(cameras.n_fixed_cameras):
                rgb = recent_obs[f'color_{i}'][-1]
                gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
                corners, ids, _ = cv2.aruco.detectMarkers(gray, arucoDict, parameters=arucoParams)
                if len(corners) > 3:
                    retval, rvec, tvec = cv2.aruco.estimatePoseBoard(corners, ids, aruco_grid, intrinsics[i], np.zeros(5), None, None)
                    rgb_draw = cv2.drawFrameAxes(rgb.copy(), intrinsics[i], np.zeros(5), rvec, tvec, 0.1)
                else:
                    rgb_draw = rgb
                rgbs.append(rgb_draw)
            
            rgbs_vis = np.hstack(rgbs)
            cv2.imshow("rgbs", rgbs_vis)
            if ord('s') == cv2.waitKey(1):            
                if not os.path.exists(args.out_folder):
                    os.makedirs(args.out_folder)
                    for i in range(cameras.n_fixed_cameras):
                        os.makedirs(f"{args.out_folder}/cam{i}")
                for i in range(cameras.n_fixed_cameras):
                    cv2.imwrite(f"{args.out_folder}/cam{i}/{index:04d}.png", recent_obs[f'color_{i}'][-1])
                print("Saved images to ", args.out_folder)
                print("Index: ", index)
                index += 1
        except KeyboardInterrupt:
            break
    cameras.stop()
    cv2.destroyAllWindows()
    print("Done")