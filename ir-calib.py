#!/usr/bin/env python3

import numpy as np
import cv2
import glob
import os
import sys
import argparse
import pyrealsense2 as rs
import time
import math
import shutil
import pudb
from itertools import combinations
import matplotlib.pyplot as plt


def setArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--stream', action='store_true')
    parser.add_argument('-c', '--calibrate', action='store_true')
    parser.add_argument('-sc', '--stereo', action='store_true')
    parser.add_argument('-bw', '--board_width', type=int, default=8)
    parser.add_argument('-bh', '--board_height', type=int, default=6)
    parser.add_argument('-bs', '--board_size', type=float, default=0.027)
    parser.add_argument('-fh', '--frame_height', type=int, default=1080)
    parser.add_argument('-n', '--total_count', type=int, default=13)
    parser.add_argument('-d', '--selected_devices', nargs='*', help='Set of the devices you want to use')
    parser.add_argument('-r', '--refine', action='store_true')
    parser.add_argument('-e', '--estimate', help='Input the calibration filename. Estimate the pose of chessboard')
    parser.add_argument('-i', '--internal', action='store_true')
    parser.add_argument('-ex', '--external', action='store_true')

    args = parser.parse_args()
    return args


class StereoCalibration:
    def __init__(self, board_width, board_height, board_size, frame_height, total_count, selected_devices):
        self.board_width = board_width
        self.board_height = board_height
        self.board_size = board_size
        self.total_count = total_count
        self.selected_devices = selected_devices
        self.wait_time = 50

        if frame_height == 480:
            self.frame_width = 640
            self.frame_height = 480
            self.depth_frame_width = 640
            self.depth_frame_height = 480

        elif frame_height == 1080:
            self.frame_width = 1920
            self.frame_height = 1080
            self.depth_frame_width = 1280
            self.depth_frame_height = 800
        
        else:
            print("Use the correct size of the frame height(480 or 1080)")
            sys.exit(0)

        self.re_frame_width = 640
        self.re_frame_height = 480

        self.imgpoints_l = []
        self.imgpoints_r = []   

        # Subpixel corner
        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        self.objp = np.zeros((board_width*board_height, 3), np.float32)
        self.objp[:, :2] = np.mgrid[0:board_width, 0:board_height].T.reshape(-1, 2) * board_size
        self.objpoints = []     # 3d point in real world space
        self.imgpoints = []     # 2d points in image plane.

        self.save_cnt = 0

        ctx = rs.context()
        self.pipeline = []
        
        # for stereo-calibrate
        self.stereocalib_criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 100, 1e-5)
        self.mtx = []           # Camera matrix
        self.dist = []          # Distortion
        self.corners2 = []

        self.align_to_color = []

        if len(ctx.devices) > 0:
            for i in ctx.devices:
                self.pipeline.append(rs.pipeline())
                for j in range(3):
                    self.mtx.append([])
                    self.dist.append([])
                    self.corners2.append([])

            for idx_d in self.selected_devices:
                d = ctx.devices[idx_d]
                print ('Found device: ', d.get_info(rs.camera_info.name), ' ', d.get_info(rs.camera_info.serial_number))
                config = rs.config()
                config.enable_device(d.get_info(rs.camera_info.serial_number))
                config.enable_stream(rs.stream.color, self.frame_width, self.frame_height, rs.format.bgr8, 30)
                config.enable_stream(rs.stream.infrared, 1, self.depth_frame_width, self.depth_frame_height, rs.format.y8, 30)
                config.enable_stream(rs.stream.infrared, 2, self.depth_frame_width, self.depth_frame_height, rs.format.y8, 30)

                print('Frame size(RGB): ', self.frame_width, 'x', self.frame_height)
                print('Frame size(Depth): ', self.depth_frame_width, 'x', self.depth_frame_height)
                profile = self.pipeline[idx_d].start(config)

                device = profile.get_device()
                depth_sensor = device.query_sensors()[0]
                depth_sensor.set_option(rs.option.laser_power, 0)
                laser_pwr = depth_sensor.get_option(rs.option.laser_power)
                print("laser power = ", laser_pwr)
                laser_range = depth_sensor.get_option_range(rs.option.laser_power)
                print("laser power range = " , laser_range.min , "~", laser_range.max)
                print('Board size: ', self.board_size)

            self.total_pipe = len(self.pipeline)

            # for stereo-calibrate
            self.n_ir = 2 # Constant
            self.imgpoints2 = np.zeros((self.total_pipe, self.n_ir, self.total_count, self.board_height * self.board_width, 1, 2), dtype=np.float32)
        else:
            print('No Intel Device connected...', len(ctx.devices))
            sys.exit(0)

    def stream(self):
        ir_image = np.zeros((self.n_ir, self.depth_frame_height, self.depth_frame_width), dtype=np.uint8)
        while True:
            for i in self.selected_devices:
                pipeline = self.pipeline[i]
                frames = pipeline.wait_for_frames()

                for j, frame in enumerate(frames):
                    ftype = frame.profile.stream_type()
                    if ftype == rs.stream.infrared:
                        #ir_image[j-1, :] = np.asanyarray(frame.get_data())
                        ir_image[j, :] = np.asanyarray(frame.get_data())
                    else:
                        rgb_image = np.asanyarray(frame.get_data())
                
                ir_color_0 = cv2.cvtColor(ir_image[0], cv2.COLOR_GRAY2BGR)
                ir_color_1 = cv2.cvtColor(ir_image[1], cv2.COLOR_GRAY2BGR)

                ir_color_0 = cv2.resize(ir_color_0, (self.re_frame_width, self.re_frame_height))
                ir_color_1 = cv2.resize(ir_color_1, (self.re_frame_width, self.re_frame_height))
                rgb_image = cv2.resize(rgb_image, (self.re_frame_width, self.re_frame_height))
                # Stack both images horizontally
                images = np.hstack((ir_color_0, ir_color_1, rgb_image))

                cv2.imshow('realsense' + str(i), images)
                key = cv2.waitKey(1)
                if key == 27:
                    cv2.destroyAllWindows()
                    pipeline.stop()
                    sys.exit(0)

    def calibrate(self):
        dir_name = 'config'
        if os.path.exists(dir_name):
            shutil.rmtree(dir_name)
        os.makedirs(dir_name, exist_ok=True)
        print('Initializing..')

        for idx_pipe in self.selected_devices:
            print('---------------')
            print('Camera', str(idx_pipe), 'Calibration')
            for i in range(3):
                time.sleep(1)
                print(3-i)
            print('---------------')
            pipe = self.pipeline[idx_pipe]

            for j in range(3): # ir1, ir2, rgb
                n_img = 0
                print(j, 'th camera calibration(0: ir_0, 1: ir_1, 2: rgb)')
                    
                while n_img < self.total_count:

                    frame = pipe.wait_for_frames()
                    gray_img = np.asanyarray(frame[j].get_data())
                    if j == 2:
                        gray_img = cv2.cvtColor(gray_img, cv2.COLOR_BGR2GRAY)
                    st_time = time.time()
                    found_board, corners = cv2.findChessboardCorners(gray_img, (self.board_width, self.board_height), None)
                    #print('time: ', time.time() - st_time)

                    if found_board: 
                        corners2 = cv2.cornerSubPix(gray_img, corners, (11, 11), (-1, -1), self.criteria)
                        cv2.drawChessboardCorners(gray_img, (self.board_width, self.board_height), corners2, found_board)
                        self.imgpoints.append(corners2)
                        self.objpoints.append(self.objp)
                        gray_img = cv2.resize(gray_img, (self.re_frame_width, self.re_frame_height))
                        print(n_img + 1, '/', self.total_count)
                        n_img += 1
                        cv2.imshow('cam' + str(j), gray_img)
                        key = cv2.waitKey(self.wait_time)
                        if key == 27:
                            cv2.destroyAllWindows()
                            pipe.stop()
                            sys.exit(0)

                cv2.destroyAllWindows()
                if j == 2:
                    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(self.objpoints, self.imgpoints, (self.frame_width, self.frame_height), None, None)
                else:
                    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(self.objpoints, self.imgpoints, (self.depth_frame_width, self.depth_frame_height), None, None)
                # Re-projection error
                mean_error = 0
                for i in range(len(self.objpoints)):
                    imgpoints2, _ = cv2.projectPoints(self.objpoints[i], rvecs[i], tvecs[i], mtx, dist)
                    error = cv2.norm(self.imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
                    mean_error += error
                print("Reprojection error: {}".format(mean_error / len(self.objpoints)))

                # Save camera parameters
                s = cv2.FileStorage('config/cam_calib_'+str(idx_pipe)+str(j)+'.xml', cv2.FileStorage_WRITE)
                s.write('mtx', mtx)
                s.write('dist', dist)
                s.write('frame_height', self.frame_height)
                s.write('board_width', self.board_width)
                s.write('board_height', self.board_height)
                s.write('board_size', self.board_size)
                s.release()
        pipe.stop()

    def estimatePose(self, cam_calibration_file):
        # Estimate the chessboard using only one camera
        # If you connect many cameras, the first camera will be selected.
        length_of_axis = 0.1 

        # Load previously saved data
        s = cv2.FileStorage()
        s.open(cam_calibration_file, cv2.FileStorage_READ)

        if not s.isOpened():
            print('Failed to open,', cam_calibration_file)
            exit(1)
        else:
            mtx = s.getNode('mtx').mat()
            dist = s.getNode('dist').mat()
            self.board_height = int(s.getNode('board_height').real())
            self.board_width = int(s.getNode('board_width').real())
            self.board_size = float(s.getNode('board_size').real())
            self.objp = np.zeros((self.board_width*self.board_height, 3), np.float32)
            self.objp[:, :2] = np.mgrid[0:self.board_width, 0:self.board_height].T.reshape(-1, 2) * self.board_size

        def draw(img, corners, imgpts):
            corner = tuple(corners[0].ravel())
            imgpts = np.array(imgpts, dtype=np.float32)
            img = cv2.line(img, corner, tuple(imgpts[0].ravel()), (0, 0, 255), 3)
            img = cv2.line(img, corner, tuple(imgpts[1].ravel()), (0, 255, 0), 3)
            img = cv2.line(img, corner, tuple(imgpts[2].ravel()), (255, 0, 0), 3)
            return img

        axis = np.float64([[length_of_axis, 0, 0], [0, length_of_axis, 0], [0, 0, length_of_axis]]).reshape(-1, 3)

        while True:
            n_cam = int(cam_calibration_file[-5])
            pipeline = self.pipeline[0]
            frames = pipeline.wait_for_frames()
            if n_cam != 2:
                gray = np.asanyarray(frames[n_cam].get_data())
                color = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            else:
                color = np.asanyarray(frames[n_cam].get_data())
                gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(color, (self.board_width, self.board_height), None)
            if ret == True:
                corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), self.criteria)
                # Find the rotation and translation vectors.
                ret, rvecs, tvecs = cv2.solvePnP(self.objp, corners2, mtx, dist)
                RR = np.eye(3)
                cv2.Rodrigues(rvecs, RR)
                print('rvecs', RR)
                print('tvecs', tvecs)
                # Project 3D points to image plane
                imgpts, jac = cv2.projectPoints(axis, rvecs, tvecs, mtx, dist)
                img = cv2.drawChessboardCorners(color, (self.board_width, self.board_height), corners2, ret)
                cv2.imshow('Chessboard', img)

            key = cv2.waitKey(1)
            if key == 27:
                pipeline.stop()
                sys.exit(0)

        cv2.destroyAllWindows()

    def internal_calibrate(self, folder_name):
        # Calibration between rgb and ir0, ir0 and ir1 in a d435 sensor

        # Load previously saved data
        # i: the number of d435 
        # j: the number of internal camera(ir0, ir1, rgb) = 3 (constant)
        # mtx: camera matrix, 
        # dist: distortion
        for i in self.selected_devices:
            for j in range(3):
                filename = 'config/cam_calib_'+str(i)+str(j)+'.xml'
                s = cv2.FileStorage()
                s.open(filename, cv2.FileStorage_READ)
                if not s.isOpened():
                    print('Failed to open,', filename)
                    exit(1)
                self.mtx[3*i+j] = s.getNode('mtx').mat()
                self.dist[3*i+j] = s.getNode('dist').mat()

        for idx_pipe in self.selected_devices:
            print('---------------')
            print('Camera', str(idx_pipe), 'Calibration')
            for i in range(3):
                time.sleep(1)
                print(3-i)
            print('---------------')
            pipe = self.pipeline[idx_pipe]
            n_img = 0

            # IR-IR, not RGB
            while n_img < self.total_count:
                gray_img = np.zeros((self.n_ir, self.depth_frame_height, self.depth_frame_width), dtype=np.uint8) 
                total_gray_img = np.zeros((self.depth_frame_height, 1), dtype=np.uint8)
                found_board = True
                frame = pipe.wait_for_frames()

                for j in range(self.n_ir): # ir0, ir1
                    gray_img_ = np.asanyarray(frame[j].get_data())
                    found_board_, corners = cv2.findChessboardCorners(gray_img_, (self.board_width, self.board_height), None)
                    found_board = found_board * found_board_

                    if found_board_: 
                        corners2_ = cv2.cornerSubPix(gray_img_, corners, (11, 11), (-1, -1), self.criteria)
                        cv2.drawChessboardCorners(gray_img_, (self.board_width, self.board_height), corners2_, found_board)
                        self.corners2[j] = corners2_
                        gray_img[j, :, :] = gray_img_

                if found_board: # When both ir cameras detect the chessboard
                    for i in self.selected_devices:
                        for j in range(self.n_ir):
                            total_gray_img = np.hstack((total_gray_img, gray_img[j]))
                            self.imgpoints2[i, j, n_img] = self.corners2[j]
                    
                    self.objpoints.append(self.objp)
                    print(n_img + 1, '/', self.total_count)
                    cv2.putText(total_gray_img, str(n_img+1) + '/' + str(self.total_count), (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 0), 2, cv2.LINE_AA)
                    total_gray_img = cv2.resize(total_gray_img, (self.re_frame_width * self.n_ir, self.re_frame_height))
                    cv2.imshow('stereo-calibrate', total_gray_img)
                    key = cv2.waitKey(self.wait_time)
                    n_img += 1
                    if key == 27:
                        cv2.destroyAllWindows()
                        pipe.stop()
                        sys.exit(0)

            flags = cv2.CALIB_FIX_INTRINSIC

        imgpoints_l = self.imgpoints2[self.selected_devices[0], 0]
        imgpoints_r = self.imgpoints2[self.selected_devices[0], 1]
        _, _, _, _, _, R, tvec, _, _ = cv2.stereoCalibrate(self.objpoints, imgpoints_l, imgpoints_r, self.mtx[0], self.dist[0], self.mtx[1], self.dist[1], (self.depth_frame_width, self.depth_frame_height), criteria=self.stereocalib_criteria, flags=flags)

        # R, tvec: Transformation from Cam 2 to Cam 1
        #tvec = -R.T.dot(tvec)
        #rotation_matrix = R.T
        #eulerAngle = self.rotationMatrixToEulerAngles(rotation_matrix) 
        eulerAngle = self.rotationMatrixToEulerAngles(R) 

        print('-----------------------------------------')
        print('Combination:', 'ir0', 'ir1')
        print('Translation vector(X, Y, Z) [meter]', tvec.T)
        print('Rotation matrix', R)
        print('Euler angle(Rx, Ry, Rz) [deg]', eulerAngle * 180 / math.pi)

        # Save camera parameters
        s = cv2.FileStorage('config/trans'+str(self.selected_devices[0])+'_ir.xml', cv2.FileStorage_WRITE)
        s.write('R', R)
        s.write('tvec', tvec)
        s.write('test', 10)
        s.release()
        return R, tvec

    def external_calibrate(self, folder_name):
        # Calibration between multiple d435 sensors
        # From first left IR sensor to second left IR sensor
        # Load previously saved data
        for i in self.selected_devices:
            filename = 'config/cam_calib_'+str(i)+'0.xml'
            s = cv2.FileStorage()
            s.open(filename, cv2.FileStorage_READ)
            if not s.isOpened():
                print('Failed to open,', filename)
                exit(1)
            self.mtx[3*i] = s.getNode('mtx').mat()
            self.dist[3*i] = s.getNode('dist').mat()

        print('Initializing..')
        for i in range(3):
            time.sleep(1)
            print(3-i)
        print('---------------')
        n_img = 0

        while n_img < self.total_count:

            gray_img = np.zeros((self.total_pipe, self.depth_frame_height, self.depth_frame_width), dtype=np.uint8)
            total_gray_img = np.zeros((self.depth_frame_height, 1), dtype=np.uint8)
            found_board = True
            self.depth_intrin = []
            self.color_intrin = []

            for i in self.selected_devices: # to calibrate btw d435 cameras
                pipe = self.pipeline[i]
                frame = pipe.wait_for_frames()
                gray_img_ = np.asanyarray(frame[0].get_data()) # frame[0] : left ir camera
                found_board_, corners = cv2.findChessboardCorners(gray_img_, (self.board_width, self.board_height), None)
                found_board = found_board * found_board_ # When every camera detects the chessboard, found_board will be True

                if found_board_:    # When each camera detects the chessboard
                    corners2_ = cv2.cornerSubPix(gray_img_, corners, (11, 11), (-1, -1), self.criteria)
                    cv2.drawChessboardCorners(gray_img_, (self.board_width, self.board_height), corners2_, found_board_)
                    self.corners2[i] = corners2_
                    gray_img[i, :, :] = gray_img_
                    #gray_img = np.concatenate((gray_img, np.expand_dims(gray_img_, axis=0)), axis=0)

            if found_board:# When both ir cameras detect the chessboard

                for i in self.selected_devices:
                    dir_name = folder_name + "/cam_" + str(i)

                    if n_img == 0:
                        if os.path.exists(dir_name):
                            shutil.rmtree(dir_name)
                        os.makedirs(dir_name, exist_ok=True)

                    cv2.imwrite(dir_name + '/img' + f'{n_img:02}.png', gray_img[i])
                    cv2.putText(gray_img[i], 'cam'+str(i), (960, 90), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 0), 2, cv2.LINE_AA)
                    total_gray_img = np.hstack((total_gray_img, gray_img[i]))
                    self.imgpoints2[i, 0, n_img] = self.corners2[i]

                self.objpoints.append(self.objp)
                print(n_img + 1, '/', self.total_count)
                
                cv2.putText(total_gray_img, str(n_img+1) + '/' + str(self.total_count), (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 0), 2, cv2.LINE_AA)
                total_gray_img = cv2.resize(total_gray_img, (self.re_frame_width * len(self.selected_devices), self.re_frame_height))
                cv2.imshow('stereo-calibrate', total_gray_img)
                key = cv2.waitKey(self.wait_time)
                n_img += 1

                if key == 27:
                    cv2.destroyAllWindows()
                    pipe.stop()
                    sys.exit(0)

        cv2.destroyAllWindows()

        flags = cv2.CALIB_FIX_INTRINSIC

        # imgpoints2[num_of_camera, left_camera]
        imgpoints_l = self.imgpoints2[self.selected_devices[0], 0] 
        imgpoints_r = self.imgpoints2[self.selected_devices[1], 0]
        # self.dist[self.selected_devices[1]*3] -> because there are three matrix. left, right, rgb.
        _, _, _, _, _, R, tvec, _, _ = cv2.stereoCalibrate(self.objpoints, imgpoints_l, imgpoints_r, self.mtx[self.selected_devices[0]], self.dist[self.selected_devices[0]], self.mtx[self.selected_devices[1]*3], self.dist[self.selected_devices[1]*3], (self.depth_frame_width, self.depth_frame_height), criteria=self.stereocalib_criteria, flags=flags)

        # R, tvec: Transformation from Cam 2 to Cam 1
        tvec = -R.T.dot(tvec)
        rotation_matrix = R.T
        eulerAngle = self.rotationMatrixToEulerAngles(rotation_matrix) 

        print('-----------------------------------------')
        print('Combination:', self.selected_devices[0], self.selected_devices[1])
        print('Translation vector(X, Y, Z) [meter]', tvec.T)
        print('Rotation matrix', rotation_matrix)
        print('Euler angle(Rx, Ry, Rz) [deg]', eulerAngle * 180 / math.pi)

        s = cv2.FileStorage('config/trans'+str(self.selected_devices[0])+str(self.selected_devices[1])+'.xml', cv2.FileStorage_WRITE)
        s.write('R', rotation_matrix)
        s.write('tvec', tvec)
        s.release()
        return rotation_matrix, tvec
        
    def rotationMatrixToEulerAngles(self, R):
        sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
        #sy = math.sqrt(R[2, 2] * R[2, 2] +  R[2, 1] * R[2, 1])
        singular = sy < 1e-6
        if not singular:
            x = math.atan2(R[2,1] , R[2,2])
            y = math.atan2(-R[2,0], sy)
            z = math.atan2(R[1,0], R[0,0])
        else:
            x = math.atan2(-R[1,2], R[1,1])
            y = math.atan2(-R[2,0], sy)
            z = 0
        return np.array([x, y, z])


if __name__ == '__main__':

    args = setArgs()

    if args.selected_devices:
        selected_devices = list(map(int, args.selected_devices))
    else:
        ctx = rs.context()
        selected_devices = list(range(len(ctx.devices)))

    s = StereoCalibration(args.board_width, args.board_height, args.board_size, args.frame_height, args.total_count, selected_devices)

    if args.stream:
        s.stream()
    elif args.calibrate:
        s.calibrate()
    elif args.internal:
        s.internal_calibrate('images')
    elif args.external:
        s.external_calibrate('images')
    elif args.stereo:
        R, t = s.stereo_calibrate('images')
    elif args.estimate:
        cam_calibration_file = args.estimate
        s.estimatePose(cam_calibration_file)
    else:
        print('Need to check the option...')
