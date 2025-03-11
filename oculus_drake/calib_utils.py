import numpy as np
from pydrake.multibody.inverse_kinematics import DifferentialInverseKinematicsParameters
from pydrake.systems.framework import DiagramBuilder, LeafSystem
from pydrake.visualization import MeshcatPoseSliders

from pydrake.all import (
    LeafSystem,
    RigidTransform,
    Value,
    ValueProducer,
    AbstractValue,
    MultibodyPlant,
    Multiplexer,
    DifferentialInverseKinematicsIntegrator,
    ConstantValueSource,
    DoDifferentialInverseKinematics,
    DifferentialInverseKinematicsStatus,
    EventStatus,
    Meshcat
)
from manipulation.scenarios import AddMultibodyTriad

from oculus_drake.realsense.cameras import Cameras
import apriltag
import cv2
from collections import defaultdict
import pupil_apriltags

LINE_LENGTH = 5
CENTER_COLOR = (0, 255, 0)
CORNER_COLOR = (255, 0, 255)

def plotPoint(image, center, color):
    center = (int(center[0]), int(center[1]))
    image = cv2.line(image,
                     (center[0] - LINE_LENGTH, center[1]),
                     (center[0] + LINE_LENGTH, center[1]),
                     color,
                     3)
    image = cv2.line(image,
                     (center[0], center[1] - LINE_LENGTH),
                     (center[0], center[1] + LINE_LENGTH),
                     color,
                     3)
    return image

def plotText(image, center, color, text):
    center = (int(center[0]) + 4, int(center[1]) - 4)
    return cv2.putText(image, str(text), center, cv2.FONT_HERSHEY_SIMPLEX,
                       1, color, 3)

#given 4x4 pose, visualize frame on camera (RGB)
def plotPose(image, pose, length=0.1, thickness=10, K = np.eye(3), dist=np.zeros(5)):
    #NOTE: apriltag x-axis, points out of the tag plane
    rvec = cv2.Rodrigues(pose[:3,:3])[0]
    tvec = pose[:3,3]
    
    axis = np.float32([[length,0,0], [0,length,0], [0,0,length]]).reshape(-1,3)
    imgpts, _ = cv2.projectPoints(axis, rvec, tvec, K, dist)
    imgpts = imgpts.astype(int)
    center, _ = cv2.projectPoints(np.float32([[0,0,0]]), rvec, tvec, K, dist)
    center = tuple(center[0].ravel())
    center = (int(center[0]), int(center[1]))
    
    image = cv2.line(image, center, tuple(imgpts[0].ravel()), (0,0,255), thickness=thickness) #red
    image = cv2.line(image, center, tuple(imgpts[1].ravel()), (0,255,0), thickness=thickness)
    image = cv2.line(image, center, tuple(imgpts[2].ravel()), (255,0,0), thickness=thickness)
    return image

def visualize_detections(image, detections):
    for detect in detections:
        image = plotPoint(image, detect.center, CENTER_COLOR)
        image = plotText(image, detect.center, CENTER_COLOR, detect.tag_id)
        for corner in detect.corners:
            image = plotPoint(image, corner, CORNER_COLOR)
    return image

class CameraCalibrateSystem(LeafSystem):
    def __init__(self, cameras: Cameras, Ks, tag_width: float = 0.056, save_data=False):
        LeafSystem.__init__(self)
        self.save_data = save_data
        self.tag_width = tag_width
        self.Ks = Ks
        self.cameras = cameras
        self.n_cam = cameras.n_fixed_cameras
        # detector_options = apriltag.DetectorOptions(families='tagstandard41h12')
        # self.detector = apriltag.Detector(options=detector_options)
        self.detector = pupil_apriltags.Detector(families='tagStandard41h12')
        self.tag2kukabase = RigidTransform()
        self.cameras_datapoints = defaultdict(list)
        self.cam_debug_poses = dict()
        self.obs = None
        
        
        if save_data:
            # take as input Kuka
            self.DeclareAbstractInputPort("tag2kukabase", Value(RigidTransform()))            
            #get streamed info
            self.DeclarePeriodicPublishEvent(period_sec=1.0/self.cameras.capture_fps, offset_sec=0.0, publish=self.SaveKukaPose)
            self.DeclareForcedPublishEvent(self.DetectTagEvent)
        
        #do something with streamed info
        self.DeclarePeriodicPublishEvent(period_sec=1.0/self.cameras.capture_fps, offset_sec=0.0, publish=self.SaveObservation)
        self.DeclarePeriodicPublishEvent(period_sec=1.0/self.cameras.capture_fps, offset_sec=0.1, publish=self.VisualizeCameras)
        
        
    def SaveObservation(self, context):
        self.obs = self.cameras.get_obs(get_color=True, get_depth=False)
    def SaveKukaPose(self, context):
        self.tag2kukabase: RigidTransform = self.get_input_port().Eval(context)
        
    def DetectTagEvent(self, context):
        obs = self.obs
        for cam_idx in range(self.n_cam):
            color = obs[f'color_{cam_idx}'][-1]
            gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
            detections = self.detector.detect(gray)
            for detect in detections:
                if detect.tag_id == 0:
                    print(f"Camera {cam_idx} detected tag pose")
                    pt_2d = detect.center
                    pt_3d = self.tag2kukabase.translation()
                    self.cameras_datapoints[f'cam{cam_idx}'].append( (pt_2d,pt_3d) )
                    
                    #get debug pose
                    K = self.Ks[cam_idx]
                    tag2kukabase = self.tag2kukabase.GetAsMatrix4()
                    camera_params = K[0,0],K[1,1],K[0,2],K[1,2]
                    tag2cam, _, _ = self.detector.detection_pose(detect, camera_params=camera_params, tag_size=self.tag_width)
                    cam2tag = np.linalg.inv(tag2cam)
                    cam2kukabase = tag2kukabase @ cam2tag
                    self.cam_debug_poses[f'cam{cam_idx}'] = RigidTransform(cam2kukabase)
                
        print()
    def VisualizeCameras(self, context):
        obs = self.obs
        for i in range(self.n_cam):
            color = obs[f'color_{i}'][-1]
            detections = self.detector.detect(cv2.cvtColor(color, cv2.COLOR_BGR2GRAY))
            color = visualize_detections(color, detections)
            cv2.imshow(f'cam{i}', color)
        cv2.waitKey(1)