
import numpy as np
import cv2
from oculus_drake.realsense.cameras import Cameras
import argparse
import os

M_TO_MM = 1000
# make aruco_board
if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--pixel_per_mm", type=float, default=10)
    args = argparser.parse_args()
    
    d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    
    square_length = 0.05
    marker_length = 0.03
    board_width_squares = 16
    board_height_squares = 10
    margin_mm = 20
    charuco = cv2.aruco.CharucoBoard_create(16, 10, square_length, marker_length, d)
    height = (square_length * board_height_squares * M_TO_MM) * args.pixel_per_mm + 2 * margin_mm * args.pixel_per_mm
    width  = (square_length * board_width_squares * M_TO_MM) * args.pixel_per_mm + 2 * margin_mm * args.pixel_per_mm
    img = charuco.draw((width, height))
    cv2.imwrite("charuco.png", img)