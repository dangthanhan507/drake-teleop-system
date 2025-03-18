import sys
sys.path.append('./')

import numpy as np
from perception.perception3d_module import Perception3DModule
import cv2
from oculus_drake.realsense.cameras import Cameras
import torch

#bbox is (xcenter,ycenter,w,h)
def visualize_bbox(image, bbox, label):
    w,h = bbox[2],bbox[3]
    x0,y0 = bbox[0]-w/2,bbox[1]-h/2
    x1,y1 = bbox[0]+w/2,bbox[1]+h/2
    x0,y0,x1,y1 = int(x0),int(y0),int(x1),int(y1)
    image = cv2.rectangle(image, (x0, y0), (x1, y1), (0, 255, 0), 10)
    # image = cv2.putText(image, label, (x0, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return image

def randomly_generate_color():
    '''
    Generate random color in hsv
    convert to rgb
    
    hsv:
        -> hue: 0-360
        -> saturation: 0-1
        -> value: 0-100
    '''
    hue = np.random.randint(0, 360)
    saturation = np.random.rand()
    value = np.random.rand()*100.0
    hsv = np.array([[[hue, saturation, value]]], dtype=np.float32)
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB).flatten()
    return rgb
def visualize_masks(image, masks):
    # different masks are colored differently on black background
    display_im = np.zeros_like(image)
    for i in range(masks.shape[0]):
        display_im[masks[i]] = randomly_generate_color()
    return display_im

if __name__ == '__main__':
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
    
    obs = cameras.get_obs(get_depth=True, get_color=True)
    det_displays = []
    mask_displays = []
    perception = Perception3DModule()
    for i in range(3):
        color = obs[f'color_{i}'][-1] # RGB
        depth = obs[f'depth_{i}'][-1]
        
        H,W,_ = color.shape
        
        
        
        # test detection module
        # text_prompts = ['box', 'table']
        text_prompts = ['red mug']
        boxes,scores,labels = perception.detect(color, captions=text_prompts, box_thresholds=0.5)
        boxes_np = boxes.detach().cpu().numpy()
        boxes_np = boxes_np * np.array([[W, H, W, H]])
        
        # visualize bounding box
        det_display = color.copy()
        for i in range(boxes_np.shape[0]):
            label = 'bottle'
            det_display = visualize_bbox(det_display, boxes_np[i], label)
        # cv2.imshow('detection', det_display)
        # cv2.waitKey(0)
        det_displays.append(det_display)
        
        # test segmentation module
        boxes = boxes * torch.Tensor([[W, H, W, H]]).to(device=perception.device, dtype=boxes.dtype)
        boxes[:,:2] -= boxes[:,2:] / 2
        boxes[:, 2:] += boxes[:,:2] #NOTE: now boxes are in format [x0,y0,x1,y1]
        (masks, _, text_labels), _ = perception.segment(color, boxes, scores, labels, text_prompts)
        masks = masks.detach().cpu().numpy()
        
        # import matplotlib.pyplot as plt
        # plt.imshow(masks[0])
        # plt.show()
        
        mask_display = visualize_masks(color.copy(), masks)
        mask_displays.append(mask_display)
        # cv2.imshow('mask', mask_display)
        # cv2.waitKey(0)
    mask_display_total = np.hstack(mask_displays)
    det_display_total = np.hstack(det_displays)
    cv2.imshow('mask', mask_display_total)
    cv2.imshow('detection', det_display_total)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    print("Done")
    exit()