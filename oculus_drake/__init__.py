import os
import numpy as np
PROJECT_PATH = os.path.dirname(__file__)
PACKAGE_XML = f'{os.path.dirname(__file__)}/../package.xml'
SCENARIO_FILEPATH = f'{os.path.dirname(__file__)}/config/teleop_iiwa.yaml'
SCENARIO_NO_WSG_FILEPATH = f'{os.path.dirname(__file__)}/config/teleop_iiwa_no_wsg.yaml'
FAKE_SCENARIO_FILEPATH = f'{os.path.dirname(__file__)}/config/fake_teleop_iiwa.yaml'
CALIB_SCENARIO_FILEPATH = f'{os.path.dirname(__file__)}/config/calib_iiwa.yaml'
FAKE_CALIB_SCENARIO_FILEPATH = f'{os.path.dirname(__file__)}/config/fake_calib_iiwa.yaml'

CALIBRATION_PATH = f'{os.path.dirname(__file__)}/calibration'
CALIBRATION_MARKER_LENGTH = 0.033

REPO_DIR = f'{os.path.dirname(__file__)}/../'
WEIGHT_DIR = os.path.join(REPO_DIR, 'weights')
THIRD_PARTY_DIR = os.path.join(REPO_DIR, 'third-party')

HOME_Q = np.array([-90.0, 10.0, 0.0, -90.0, 0.0, 80.0, -90.0]) * np.pi / 180