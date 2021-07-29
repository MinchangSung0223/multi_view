#!/usr/bin/env python3

import cv2
import numpy as np
import argparse
import pyrealsense2 as rs
import pudb
import sys
import os


def setArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--make')
    parser.add_argument('-d', '--detect')
    args = parser.parse_args()
    return args


class arucoMarker:
    def __init__(self, frame_height, markerLength, dictionary):

        # Initialize the detector parameters using default values
        self.parameters =  cv2.aruco.DetectorParameters_create()
        self.markerLength = markerLength
        self.dictionary = dictionary
        if frame_height == 1080:
            self.frame_width = 1920
            self.frame_height = 1080
        if frame_height == 480:
            self.frame_width = 640
            self.frame_height = 480

    def make(self, markerPix, markerId):
        # Generate the marker
        markerImage = np.zeros((markerPix, markerPix), dtype=np.uint8)
        markerImage = cv2.aruco.drawMarker(self.dictionary, markerId, markerPix, markerImage, 1);
        cv2.imwrite('marker' + str(markerId) + '.png', markerImage);

    def detect(self, color_img, cam_calib_file):

        # Load previously saved data
        s = cv2.FileStorage()
        s.open(cam_calib_file, cv2.FileStorage_READ)
        if not s.isOpened():
            print('Failed to open,', args.detect)
            exit(1)
        else:
            dist = s.getNode('dist').mat()
            camera_matrix = s.getNode('mtx').mat()

        # Detect the markers in the image
        markerCorners, markerIds, rejectedCandidates = cv2.aruco.detectMarkers(color_img, self.dictionary, parameters=self.parameters)

        try:
            cv2.aruco.drawDetectedMarkers(color_img, markerCorners, markerIds, (0, 255, 0))
            rvec, tvec, retval = cv2.aruco.estimatePoseSingleMarkers(markerCorners, self.markerLength, camera_matrix, dist)
            color_img = cv2.resize(color_img, (640, 480))
            cv2.imshow('realsense', color_img)
            key = cv2.waitKey(1)
            if key == 27:
                cv2.destroyAllWindows()
                return None, False
            if rvec is not None:
                num_marker = rvec.shape[0]
                print(num_marker, 'detected')
                T_cam2marker = np.zeros((num_marker, 4, 4))
                for i, (rvec_, tvec_) in enumerate(zip(rvec, tvec)):
                    R = np.eye(3)
                    T_cam2marker_ = np.eye(4)
                    cv2.Rodrigues(rvec_, R)
                    T_cam2marker_[:3, :3] = R
                    T_cam2marker_[:3, 3] = tvec_
                    T_cam2marker[i] = T_cam2marker_
                return T_cam2marker, True
            else:
                print('cannot detect..')
                #return None, False
            print('=================')
        except Exception as e:
            return None, False
            print(e)

    def graph(self):
        print("TODO")


if __name__ == "__main__":
    args = setArgs()

    # markerLength	:the length of the markers' side. The returning translation vectors will be in the same unit. Normally, unit is meters.
    markerLength = 70 / 1000  # meter
    frame_height = 1080

    if frame_height == 1080:
        frame_width = 1920
    if frame_height == 480:
        frame_width = 640

    #Load the dictionary that was used to generate the markers.
    dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)
    #dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_5X5_250)

    a = arucoMarker(frame_height, markerLength, dictionary)

    if args.make:
        print('make')
        # make 할 때는 id 입력이 필요
        markerId = 22
        a.make(int(args.make), markerId)

    elif args.detect:
        cam_calib_file = args.detect
        ctx = rs.context()
        num_sensors = len(ctx.sensors)
        if num_sensors == 0:
            print('No intel device')
        else:
            pipeline = rs.pipeline()
            d = ctx.devices[0]
            print ('Found device: ', d.get_info(rs.camera_info.name), ' ', d.get_info(rs.camera_info.serial_number))
            config = rs.config()
            config.enable_device(d.get_info(rs.camera_info.serial_number))
            config.enable_stream(rs.stream.color, frame_width, frame_height, rs.format.bgr8, 30)
            #config.enable_stream(rs.stream.depth, depth_frame_width, depth_frame_height, rs.format.z16, 30)
            print('Frame size: ', frame_width, 'x', frame_height)
            profile = pipeline.start(config)
            while True:
                frame = pipeline.wait_for_frames()
                color_frm = frame.get_color_frame()
                color_img = np.asanyarray(color_frm.get_data())
                a.detect(color_img, cam_calib_file)
#                T_cam2marker, ret = a.detect(color_img, cam_calib_file)
#                if ret:
#                    print(T_cam2marker)
#                else:
#                    print('exit')
#                    break
    else:
        print('Use the options. -d or -m')
