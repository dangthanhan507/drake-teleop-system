from oculus_drake.realsense.cameras import Cameras, save_intrinsics
import argparse
if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--save_file', type=str, required=True)
    args = argparser.parse_args()
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
    save_intrinsics(Ks, args.save_file)