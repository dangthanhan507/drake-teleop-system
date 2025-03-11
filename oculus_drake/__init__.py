import os
PROJECT_PATH = os.path.dirname(__file__)
PACKAGE_XML = f'{os.path.dirname(__file__)}/../package.xml'
SCENARIO_FILEPATH = f'{os.path.dirname(__file__)}/config/teleop_iiwa.yaml'
SCENARIO_NO_WSG_FILEPATH = f'{os.path.dirname(__file__)}/config/teleop_iiwa_no_wsg.yaml'
FAKE_SCENARIO_FILEPATH = f'{os.path.dirname(__file__)}/config/fake_teleop_iiwa.yaml'