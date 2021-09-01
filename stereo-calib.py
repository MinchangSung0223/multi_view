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

from generate_pattern import aruco


def setArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--stream', action='store_true')
    parser.add_argument('-c', '--calibrate', action='store_true')
    parser.add_argument('-sc', '--stereo', action='store_true')
    parser.add_argument('-bw', '--board_width', type=int, default=9)
    parser.add_argument('-bh', '--board_height', type=int, default=6)
    parser.add_argument('-bs', '--board_size', type=float, default=0.0235)
    parser.add_argument('-fh', '--frame_height', type=int, default=480)
    parser.add_argument('-n', '--total_count', type=int, default=13)
    parser.add_argument('-d', '--selected_devices', nargs='*', help='Set of the devices you want to use')
    parser.add_argument('-r', '--refine', action='store_true')
    parser.add_argument('-a', '--aruco')

    args = parser.parse_args()
    return args


class StereoCalibration:
    def __init__(self, board_width, board_height, board_size, frame_height, total_count, selected_devices):
        self.board_width = board_width
        self.board_height = board_height
        self.board_size = board_size
        self.total_count = total_count
        self.selected_devices = selected_devices
        self.wait_time = 1000
        #self.total_area = []

        if frame_height == 480:
            self.frame_width = 640
            self.frame_height = 480
            self.depth_frame_width = 640
            self.depth_frame_height = 480

        elif frame_height == 1080:
            self.frame_width = 1920
            self.frame_height = 1080
            self.depth_frame_width = 1280
            self.depth_frame_height = 720
        
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

#        ctx = rs.context()
#        list = ctx.query_devices()
#        for dev in list:
#              serial = dev.query_sensors()[0].get_info(rs.camera_info.serial_number)
#              # compare to desired SN
#              dev.hardware_reset()
#
#        devices = ctx.query_devices()
#        for dev in devices:
#                dev.hardware_reset()

        ctx = rs.context()
        self.pipeline = []
        
        # for stereo-calibrate
        self.stereocalib_criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 100, 1e-5)
        self.mtx = [[]]           # Camera matrix
        self.dist = [[]]          # Distortion
        self.corners2 = []

        self.align_to_color = []

        if len(ctx.devices) > 0:
            for i in ctx.devices:
                self.pipeline.append(rs.pipeline())
                self.mtx.append([])
                self.dist.append([])
                self.corners2.append([])

            for idx_d in self.selected_devices:
                d = ctx.devices[idx_d]
                print ('Found device: ', d.get_info(rs.camera_info.name), ' ', d.get_info(rs.camera_info.serial_number))
                config = rs.config()
                config.enable_device(d.get_info(rs.camera_info.serial_number))
                config.enable_stream(rs.stream.color, self.frame_width, self.frame_height, rs.format.bgr8, 30)
                config.enable_stream(rs.stream.depth, self.depth_frame_width, self.depth_frame_height, rs.format.z16, 30)
                print('Frame size: ', self.frame_width, 'x', self.frame_height)
                profile = self.pipeline[idx_d].start(config)

                # Get the depth sensor's depth scale 
                depth_sensor = profile.get_device().first_depth_sensor()
                self.depth_scale = depth_sensor.get_depth_scale()

                print('Depth frame:', self.depth_frame_width, 'x', self.depth_frame_height)
                print("Depth Scale is: " , self.depth_scale)

                align_to = rs.stream.color
                self.align_to_color.append(rs.align(align_to))

            self.total_pipe = len(self.pipeline)
            self.fiducial_pts = np.zeros((self.total_pipe, 4, 3)) # (num of camera, num of points, position)
            self.all_fiducial_pts = np.zeros((self.total_pipe, len(self.objp), 3)) # (num of camera, num of points, position)
            # for stereo-calibrate
            self.imgpoints2 = np.zeros((self.total_pipe, self.total_count, self.board_height * self.board_width, 1, 2), dtype=np.float32)
        else:
            print('No Intel Device connected...', len(ctx.devices))
            sys.exit(0)

    def stream(self):
        while True:
            for i in self.selected_devices:
                pipe = self.pipeline[i]
                frame = pipe.wait_for_frames()
                color_frm = frame.get_color_frame()
                color_img = np.asanyarray(color_frm.get_data())
                color_img = cv2.resize(color_img, (640, 480))
                cv2.imshow('realsense' + str(i), color_img)
                key = cv2.waitKey(1)
                if key == 27:
                    cv2.destroyAllWindows()
                    pipe.stop()
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
            n_img = 0
            pipe = self.pipeline[idx_pipe]

            while n_img < self.total_count:
                frame = pipe.wait_for_frames()
                color_frm = frame.get_color_frame()
                color_img = np.asanyarray(color_frm.get_data())
                gray_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)
                found_board, corners = cv2.findChessboardCorners(gray_img, (self.board_width, self.board_height), None)
                if found_board: 
                    corners2 = cv2.cornerSubPix(gray_img, corners, (11, 11), (-1, -1), self.criteria)
                    cv2.drawChessboardCorners(gray_img, (self.board_width, self.board_height), corners2, found_board)

                    self.imgpoints.append(corners2)

                    if idx_pipe == 0:
                        self.objpoints.append(self.objp)

                    gray_img = cv2.resize(gray_img, (640, 480))
                    print(n_img + 1, '/', self.total_count)
                    n_img += 1
                    cv2.imshow('cam', gray_img)
                    key = cv2.waitKey(self.wait_time)
                    if key == 27:
                        cv2.destroyAllWindows()
                        pipe.stop()
                        sys.exit(0)

            cv2.destroyAllWindows()

            ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(self.objpoints, self.imgpoints, (self.frame_width, self.frame_height), None, None)

            # Re-projection error
            mean_error = 0
            for i in range(len(self.objpoints)):
                imgpoints2, _ = cv2.projectPoints(self.objpoints[i], rvecs[i], tvecs[i], mtx, dist)
                error = cv2.norm(self.imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
                mean_error += error
            print("Reprojection error: {}".format(mean_error / len(self.objpoints)))

            self.imgpoints = []     # 2d points in image plane.

            # Save camera parameters
            s = cv2.FileStorage('config/cam_calib_'+str(idx_pipe)+'.xml', cv2.FileStorage_WRITE)
            s.write('mtx', mtx)
            s.write('dist', dist)
            s.release()
        pipe.stop()

    def stereo_calibrate(self, folder_name):
        if len(self.pipeline) < 2:
            print('Need to more cameras...(if you want to stereo-calibrate)')
            sys.exit()

        # Load previously saved data
        for i in self.selected_devices:
            filename = 'config/cam_calib_'+str(i)+'.xml'
            s = cv2.FileStorage()
            s.open(filename, cv2.FileStorage_READ)
            if not s.isOpened():
                print('Failed to open,', filename)
                exit(1)
            self.mtx[i] = s.getNode('mtx').mat()
            self.dist[i] = s.getNode('dist').mat()

        print('Initializing..')
        for i in range(3):
            time.sleep(1)
            print(3-i)
        print('---------------')
        n_img = 0

        while n_img < self.total_count:
            gray_img = np.zeros((self.total_pipe, self.frame_height, self.frame_width), dtype=np.uint8)
            total_gray_img = np.zeros((self.frame_height, 1), dtype=np.uint8)
            found_board = True
            self.depth_intrin = []
            self.color_intrin = []

            for i in self.selected_devices:

                self.pipe = self.pipeline[i]
                frame = self.pipe.wait_for_frames()
                aligned_frames = self.align_to_color[i].process(frame)

                self.depth_frm = aligned_frames.get_depth_frame()
                self.color_frm = aligned_frames.get_color_frame()

                self.depth_intrin.append(self.depth_frm.profile.as_video_stream_profile().intrinsics)
                self.color_intrin.append(self.color_frm.profile.as_video_stream_profile().intrinsics)

                self.depth_img = np.asanyarray(self.depth_frm.get_data())
                self.color_img = np.asanyarray(self.color_frm.get_data())

                gray_img_ = cv2.cvtColor(self.color_img, cv2.COLOR_BGR2GRAY)
                found_board_, corners = cv2.findChessboardCorners(gray_img_, (self.board_width, self.board_height), None)
                found_board = found_board * found_board_ # When every camera detects the chessboard, found_board will be True

                if found_board_:    # When each camera detects the chessboard
                    corners2_ = cv2.cornerSubPix(gray_img_, corners, (11, 11), (-1, -1), self.criteria)
                    cv2.drawChessboardCorners(gray_img_, (self.board_width, self.board_height), corners2_, found_board_)
                    self.corners2[i] = corners2_
                    gray_img[i, :, :] = gray_img_
                    #gray_img = np.concatenate((gray_img, np.expand_dims(gray_img_, axis=0)), axis=0)

            if found_board:

                for i in self.selected_devices:
                    dir_name = folder_name + "/cam_" + str(i)

                    if n_img == 0:
                        if os.path.exists(dir_name):
                            shutil.rmtree(dir_name)
                        os.makedirs(dir_name, exist_ok=True)

                    cv2.imwrite(dir_name + '/img' + f'{n_img:02}.png', gray_img[i])
                    cv2.putText(gray_img[i], 'cam'+str(i), (960, 90), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 0), 2, cv2.LINE_AA)

#                    left_top_u = int(self.corners2[i][0][0][0])
#                    left_top_v = int(self.corners2[i][0][0][1])
#                    right_top_u = int(self.corners2[i][self.board_width-1][0][0])
#                    right_top_v = int(self.corners2[i][self.board_width-1][0][1])
#                    
#                    left_bottom_u = int(self.corners2[i][self.board_width*(self.board_height-1)][0][0])
#                    left_bottom_v = int(self.corners2[i][self.board_width*(self.board_height-1)][0][1])
#                    right_bottom_u = int(self.corners2[i][self.board_height*self.board_width-1][0][0])
#                    right_bottom_v = int(self.corners2[i][self.board_height*self.board_width-1][0][1])
#
#                    cv2.circle(gray_img[i], (left_top_u, left_top_v), 2, (0,0,255), 18)
#                    cv2.circle(gray_img[i], (right_top_u, right_top_v), 2, (255,0,0), 18)
#                    cv2.circle(gray_img[i], (left_bottom_u, left_bottom_v), 2, (255,0,0), 18)
#                    cv2.circle(gray_img[i], (right_bottom_u, right_bottom_v), 2, (255,0,0), 18)
#
##                    depth_value = self.depth_frm.get_distance(left_top_u, left_top_v)
##                    depth_point1 = rs.rs2_deproject_pixel_to_point(self.color_intrin[i], [left_top_u, left_top_v], depth_value)
##                    depth_point2 = rs.rs2_deproject_pixel_to_point(self.depth_intrin[i], [left_top_u, left_top_v], depth_value)
##                    print('Cam ', i)
##                    print('get_distance : ', depth_value)
##                    print('deproject(color intrin) :', depth_point1)
##                    print('deproject(depth intrin) :', depth_point2)
##                    print('color intrinsic', self.color_intrin)
##                    print('depth intrinsic', self.depth_intrin)
##                    print('-------------------------')
#
#                    self.fiducial_pts[i, 0] = np.array(rs.rs2_deproject_pixel_to_point(self.depth_intrin[i], [left_top_u, left_top_v], self.depth_frm.get_distance(left_top_u, left_top_v)))
#                    self.fiducial_pts[i, 1] = np.array(rs.rs2_deproject_pixel_to_point(self.depth_intrin[i], [right_top_u, right_top_v], self.depth_frm.get_distance(right_top_u, right_top_v)))
#                    self.fiducial_pts[i, 2] = np.array(rs.rs2_deproject_pixel_to_point(self.depth_intrin[i], [left_bottom_u, left_bottom_v], self.depth_frm.get_distance(left_bottom_u, left_bottom_v)))
#                    self.fiducial_pts[i, 3] = np.array(rs.rs2_deproject_pixel_to_point(self.depth_intrin[i], [right_bottom_u, right_bottom_v], self.depth_frm.get_distance(right_bottom_u, right_bottom_v)))

                    total_gray_img = np.hstack((total_gray_img, gray_img[i]))
                    self.imgpoints2[i, n_img] = self.corners2[i]

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
#        for i in self.selected_devices:
#            self.pipeline[i].stop()

        flags = cv2.CALIB_FIX_INTRINSIC
        # flags |= cv2.CALIB_FIX_PRINCIPAL_POINT
        # flags |= cv2.CALIB_USE_INTRINSIC_GUESS
        # flags |= cv2.CALIB_FIX_FOCAL_LENGTH
        # flags |= cv2.CALIB_FIX_ASPECT_RATIO
        # flags |= cv2.CALIB_ZERO_TANGENT_DIST
        # flags |= cv2.CALIB_RATIONAL_MODEL
        # flags |= cv2.CALIB_SAME_FOCAL_LENGTH

        # cam_combi = list(combinations(range(self.total_pipe), 2))
        # for c in cam_combi:
        # ret, M1, d1, M2, d2, R, tvec, E, F = cv2.stereoCalibrate(self.objpoints, self.imgpoints[c[0]], self.imgpoints[c[1]], self.mtx[c[0]], self.dist[c[0]], self.mtx[c[1]], self.dist[c[1]], (self.frame_width, self.frame_height), criteria=self.stereocalib_criteria, flags=flags)

        imgpoints_l = self.imgpoints2[self.selected_devices[0]]
        imgpoints_r = self.imgpoints2[self.selected_devices[1]]
        _, _, _, _, _, R, tvec, _, _ = cv2.stereoCalibrate(self.objpoints, imgpoints_l, imgpoints_r, self.mtx[self.selected_devices[0]], self.dist[self.selected_devices[0]], self.mtx[self.selected_devices[1]], self.dist[self.selected_devices[1]], (self.frame_width, self.frame_height), criteria=self.stereocalib_criteria, flags=flags)

        # R, tvec: Transformation from Cam 2 to Cam 1
        tvec = -R.T.dot(tvec)
        rotation_matrix = R.T
        eulerAngle = self.rotationMatrixToEulerAngles(rotation_matrix) 

        print('-----------------------------------------')
        print('Combination:', self.selected_devices[0], self.selected_devices[1])
        print('Translation vector(X, Y, Z) [meter]', tvec.T)
        print('Rotation matrix', rotation_matrix)
        print('Euler angle(Rx, Ry, Rz) [deg]', eulerAngle * 180 / math.pi)

        # Save camera parameters
        # s = cv2.FileStorage('config/trans'+str(c[0])+str(c[1])+'.xml', cv2.FileStorage_WRITE)     # If you want to save the file at once(using combination) ex) n=3 -> 01, 12, 02


# TODO: need to this code!! uncomment! 
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

    def refine_stereo_calibrate(self, R, t):

        # Load previously saved data
        for i in self.selected_devices:
            filename = 'config/cam_calib_'+str(i)+'.xml'
            s = cv2.FileStorage()
            s.open(filename, cv2.FileStorage_READ)
            if not s.isOpened():
                print('Failed to open,', filename)
                exit(1)
            self.mtx[i] = s.getNode('mtx').mat()
            self.dist[i] = s.getNode('dist').mat()

        print('Initializing..')
        for i in range(3):
            time.sleep(1)
            print(3-i)
        print('---------------')

        gray_img = np.zeros((self.total_pipe, self.frame_height, self.frame_width), dtype=np.uint8)
        total_gray_img = np.zeros((self.frame_height, 1), dtype=np.uint8)
        found_board = True
        self.depth_intrin = []
        self.color_intrin = []

        transformed_pts = []
        new_transformed_pts = []

        self.vtx = np.zeros((1, self.frame_width * self.frame_height), dtype='f,f,f')
        self.colorful = np.zeros((1, self.frame_width * self.frame_height, 3))

        for i in self.selected_devices:

            self.pipe = self.pipeline[i]
            frame = self.pipe.wait_for_frames()
            aligned_frames = self.align_to_color[i].process(frame)

            self.depth_frm = aligned_frames.get_depth_frame()
            self.color_frm = aligned_frames.get_color_frame()
            self.depth_intrin.append(self.depth_frm.profile.as_video_stream_profile().intrinsics)
            self.color_intrin.append(self.color_frm.profile.as_video_stream_profile().intrinsics)

            pc = rs.pointcloud()
            pc.map_to(self.color_frm)
            points = pc.calculate(self.depth_frm)
            #vtx_ = np.asanyarray(points.get_vertices())
            vtx = np.array(points.get_vertices())
            self.vtx = np.concatenate((self.vtx, np.expand_dims(vtx, axis=0)), axis=0)

            self.depth_img = np.asanyarray(self.depth_frm.get_data())
            self.color_img = np.asanyarray(self.color_frm.get_data())

            colorful = self.color_img.reshape(-1, 3)    # to extract color of the roi
            self.colorful = np.concatenate((self.colorful, np.expand_dims(colorful, axis=0)), axis=0)

            gray_img_ = cv2.cvtColor(self.color_img, cv2.COLOR_BGR2GRAY)
            found_board_, corners = cv2.findChessboardCorners(gray_img_, (self.board_width, self.board_height), None)
            found_board = found_board * found_board_ # When every camera detects the chessboard, found_board will be True

            if found_board_:    # When each camera detects the chessboard
                corners2_ = cv2.cornerSubPix(gray_img_, corners, (11, 11), (-1, -1), self.criteria)
                cv2.drawChessboardCorners(gray_img_, (self.board_width, self.board_height), corners2_, found_board_)
                self.corners2[i] = corners2_
                gray_img[i, :, :] = gray_img_

        if found_board:
            print('Founded!')
            for i in self.selected_devices:

                dec_filter = rs.decimation_filter()
                spat_filter = rs.spatial_filter()
                temp_filter = rs.temporal_filter()

                # ---- #
                # TODO: Remove these lines.
                self.pipe = self.pipeline[i]
                frame = self.pipe.wait_for_frames()
                aligned_frames = self.align_to_color[i].process(frame)

                self.depth_frm = aligned_frames.get_depth_frame()

                #self.depth_frm = dec_filter.process(self.depth_frm)
                # dec_filter : change the frame size
                # self.depth_frm = spat_filter.process(self.depth_frm)
                # self.depth_frm = temp_filter.process(self.depth_frm)

                self.color_frm = aligned_frames.get_color_frame()
                self.depth_intrin.append(self.depth_frm.profile.as_video_stream_profile().intrinsics)
                self.color_intrin.append(self.color_frm.profile.as_video_stream_profile().intrinsics)

                pc = rs.pointcloud()
                pc.map_to(self.color_frm)
                points = pc.calculate(self.depth_frm)
                vtx = np.array(points.get_vertices())

                self.depth_img = np.asanyarray(self.depth_frm.get_data())
                self.color_img = np.asanyarray(self.color_frm.get_data())
                colorful = self.color_img.reshape(-1, 3)    # to extract color of the roi
                # ---- #
                for idx, fidu in enumerate(self.corners2[i]):
                    self.all_fiducial_pts[i, idx] = np.array(rs.rs2_deproject_pixel_to_point(self.depth_intrin[i], fidu[0].tolist(), self.depth_frm.get_distance(fidu[0][0], fidu[0][1])))

                cv2.putText(gray_img[i], 'cam'+str(i), (960, 90), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 0), 2, cv2.LINE_AA)
                left_top_u = int(self.corners2[i][0][0][0])
                left_top_v = int(self.corners2[i][0][0][1])
                right_top_u = int(self.corners2[i][self.board_width-1][0][0])
                right_top_v = int(self.corners2[i][self.board_width-1][0][1])
                
                left_bottom_u = int(self.corners2[i][self.board_width*(self.board_height-1)][0][0])
                left_bottom_v = int(self.corners2[i][self.board_width*(self.board_height-1)][0][1])
                right_bottom_u = int(self.corners2[i][self.board_height*self.board_width-1][0][0])
                right_bottom_v = int(self.corners2[i][self.board_height*self.board_width-1][0][1])

                cv2.circle(gray_img[i], (left_top_u, left_top_v), 2, (0,0,255), 18)
                cv2.circle(gray_img[i], (right_top_u, right_top_v), 2, (255,0,0), 18)
                cv2.circle(gray_img[i], (left_bottom_u, left_bottom_v), 2, (255,0,0), 18)
                cv2.circle(gray_img[i], (right_bottom_u, right_bottom_v), 2, (255,0,0), 18)

                self.fiducial_pts[i, 0] = np.array(rs.rs2_deproject_pixel_to_point(self.depth_intrin[i], [left_top_u, left_top_v], self.depth_frm.get_distance(left_top_u, left_top_v)))
                self.fiducial_pts[i, 1] = np.array(rs.rs2_deproject_pixel_to_point(self.depth_intrin[i], [right_top_u, right_top_v], self.depth_frm.get_distance(right_top_u, right_top_v)))
                self.fiducial_pts[i, 2] = np.array(rs.rs2_deproject_pixel_to_point(self.depth_intrin[i], [left_bottom_u, left_bottom_v], self.depth_frm.get_distance(left_bottom_u, left_bottom_v)))
                self.fiducial_pts[i, 3] = np.array(rs.rs2_deproject_pixel_to_point(self.depth_intrin[i], [right_bottom_u, right_bottom_v], self.depth_frm.get_distance(right_bottom_u, right_bottom_v)))

                # depth_frame.get_distance <-> depth_image
#                self.fiducial_pts[i, 0] = np.array(rs.rs2_deproject_pixel_to_point(self.depth_intrin[i], [left_top_u, left_top_v], self.depth_img[left_top_v, left_top_u]))
#                self.fiducial_pts[i, 1] = np.array(rs.rs2_deproject_pixel_to_point(self.depth_intrin[i], [right_top_u, right_top_v], self.depth_img[right_top_v, right_top_u]))
#                self.fiducial_pts[i, 2] = np.array(rs.rs2_deproject_pixel_to_point(self.depth_intrin[i], [left_bottom_u, left_bottom_v], self.depth_img[left_bottom_v, left_bottom_u]))
#                self.fiducial_pts[i, 3] = np.array(rs.rs2_deproject_pixel_to_point(self.depth_intrin[i], [right_bottom_u, right_bottom_v], self.depth_img[right_bottom_v, right_bottom_u]))

                total_gray_img = np.hstack((total_gray_img, gray_img[i]))

                num_vtx = len(vtx)
                roi_pts = []

                for j in range(num_vtx):
                    point = np.array([np.float(vtx[j][0]), np.float(vtx[j][1]), np.float(vtx[j][2])])
                    if self.pointsInBoundingBox(self.fiducial_pts[i], point):
                        roi_pts.append([np.float(vtx[j][0]), np.float(vtx[j][1]), np.float(vtx[j][2]), np.int(colorful[j][0]), np.int(colorful[j][1]), np.int(colorful[j][2])])

                cloud_file = 'cloud' + str(i) + '.ply'
                num_roi = len(roi_pts)
                print('total points: ', num_vtx)
                print('roi points: ', num_roi)

                # To write the ply format, it should be written on the first line.
                # So, it will be written after the points were checked.
                with open(cloud_file,'w') as f:
                    f.write('ply\n')
                    f.write('format ascii 1.0\n')
                    f.write('element vertex ' + str(num_roi) + '\n')
                    f.write('property float32 x\n')
                    f.write('property float32 y\n')
                    f.write('property float32 z\n')
                    f.write('property uchar red\n')
                    f.write('property uchar green\n')
                    f.write('property uchar blue\n')
                    f.write('end_header\n')
                    for pt in zip(roi_pts):
                        f.write(str(pt[0][0])+' '+str(pt[0][1])+' '+str(pt[0][2])+' '+str(pt[0][3])+' '+str(pt[0][4])+' '+str(pt[0][5])+'\n')

                with open('cloud'+str(i)+'_corners.ply','w') as f:
                    f.write('ply\n')
                    f.write('format ascii 1.0\n')
                    f.write('element vertex 4\n')
                    f.write('property float32 x\n')
                    f.write('property float32 y\n')
                    f.write('property float32 z\n')
                    f.write('end_header\n')
                    for pt in zip(self.fiducial_pts[i]):
                        f.write(str(pt[0][0])+' '+str(pt[0][1])+' '+str(pt[0][2])+'\n')

                with open('cloud'+str(i)+'_fiducial.ply','w') as f:
                    f.write('ply\n')
                    f.write('format ascii 1.0\n')
                    f.write('element vertex ' + str(len(self.objp)) + '\n')
                    f.write('property float32 x\n')
                    f.write('property float32 y\n')
                    f.write('property float32 z\n')
                    f.write('end_header\n')
                    for pt in zip(self.all_fiducial_pts[i]):
                        f.write(str(pt[0][0])+' '+str(pt[0][1])+' '+str(pt[0][2])+'\n')
                
                self.all_fiducial_pts = np.zeros((self.total_pipe, len(self.objp), 3)) # (num of camera, num of points, position)

            total_gray_img = cv2.resize(total_gray_img, (self.re_frame_width * len(self.selected_devices), self.re_frame_height))
            cv2.imshow('stereo-calibrate', total_gray_img)
            key = cv2.waitKey(self.wait_time)
            #key = cv2.waitKey(0)

            if key == 27:
                cv2.destroyAllWindows()
                self.pipe.stop()
                sys.exit(0)

        else:
            print('Cannot detect the marker!')

        # Close all cameras
        for i in self.selected_devices:
            self.pipeline[i].stop()

        # Transform the points w.r.t camera 1 to camera 0
        for pts in zip(self.fiducial_pts[1]):
            transformed_pts.append(R.dot(np.reshape(pts, (3, 1))) + t)

        width = (self.board_width - 1) * self.board_size
        height = (self.board_height - 1) * self.board_size

        est_width = []
        est_height = []
        est_width.append(np.linalg.norm(self.fiducial_pts[0][0] - self.fiducial_pts[0][1]))
        est_width.append(np.linalg.norm(self.fiducial_pts[0][2] - self.fiducial_pts[0][3]))
        est_height.append(np.linalg.norm(self.fiducial_pts[0][0] - self.fiducial_pts[0][2]))
        est_height.append(np.linalg.norm(self.fiducial_pts[0][1] - self.fiducial_pts[0][3]))

        print('----------------------------')
        print('Ground truth : ', width)
        print('Estimation width 1: ', est_width[0])
        print('Estimation width 2: ', est_width[1])
        print('----------------------------')
        print('Ground truth : ', height)
        print('Estimation height 1: ', est_height[0])
        print('Estimation height 2: ', est_height[1])
        print('----------------------------')
        print(abs(width - est_width[0])/width * 100)
        print(abs(width - est_width[1])/width * 100)
        print('----------------------------')
        print(abs(height - est_height[0])/height * 100)
        print(abs(height - est_height[1])/height * 100)

        print('----------------------------')
        print('point 1(by cam 1): ', transformed_pts[0].T)
        print('point 2(by cam 1): ', transformed_pts[1].T)
        print('point 3(by cam 1): ', transformed_pts[2].T)
        print('point 4(by cam 1): ', transformed_pts[3].T)
        print('------------------')
        print('points 1,2,3,4(cam 0):\n ', self.fiducial_pts[0])
        print('----------------------------')
        R_re, t_re = self.absoluteOrientation(np.array(transformed_pts).reshape(4,3), self.fiducial_pts[0])
        print('R_re', R_re)
        print('t_re', t_re)
        print('----------------------------')
        
        T_1 = np.hstack((R, t))
        T_1 = np.vstack((T_1, np.array([[0, 0, 0, 1]])))
        T_re = np.hstack((R_re, t_re.reshape(3, 1)))
        T_re = np.vstack((T_re, np.array([[0, 0, 0, 1]])))

        print('-----------------------------')
        print('T1 \n', T_1)
        print('T_re \n', T_re)
        print('T = T_re * T1 \n', T_re.dot(T_1))
        print('-----------------------------')
        
        err_abs = []
        for pts0, pts1 in zip(self.fiducial_pts[0], self.fiducial_pts[1]):
            temp_p = np.append(pts1, 1)
            err_abs_ = np.linalg.norm(T_re.dot(T_1).dot(temp_p)[:3] - pts0)
            print('Norm error :', err_abs_)
            err_abs.append(err_abs_)

        print('Total error :', sum(err_abs) / len(err_abs))
        print('-----------------------------')
        err_bef = []
        for pts0, pts1 in zip(self.fiducial_pts[0], self.fiducial_pts[1]):
            temp_p = np.append(pts1, 1)
            err_bef_ = np.linalg.norm(T_1.dot(temp_p)[:3] - pts0)
            print('Norm error(before refinement) :', err_bef_)
            err_bef.append(err_bef_)

        print('Total error(before_refinement) :', sum(err_bef) / len(err_bef))

        print('-----------------------------')
        print('-- To check the depth hole --')
        for i in zip(self.fiducial_pts):
            print(i)
        print('-----------------------------')

        # Save camera parameters
        s = cv2.FileStorage('config/new_trans'+str(self.selected_devices[0])+str(self.selected_devices[1])+'.xml', cv2.FileStorage_WRITE)
        s.write('R', T_re.dot(T_1)[:3, :3])
        s.write('tvec', T_re.dot(T_1)[:3, 3])
        s.release()

    def absoluteOrientation(self, A, B):

        centroid_a = np.mean(A, axis=0)
        centroid_b = np.mean(B, axis=0)
        A_wrt_centroid = A - centroid_a
        B_wrt_centroid = B - centroid_b
        M = 0
        for a_, b_ in zip(A_wrt_centroid, B_wrt_centroid):
            M_ = np.dot(np.expand_dims(a_, axis=1), np.expand_dims(b_, axis=0))
            M = M + M_
        Sxx, Sxy, Sxz = M[0, 0], M[0, 1], M[0, 2]
        Syx, Syy, Syz = M[1, 0], M[1, 1], M[1, 2]
        Szx, Szy, Szz = M[2, 0], M[2, 1], M[2, 2]
        N = np.array([[Sxx + Syy + Szz, Syz - Szy, Szx - Sxz, Sxy - Syx],
                      [Syz - Szy, Sxx - Syy - Szz, Sxy + Syx, Szx + Sxz],
                      [Szx - Sxz, Sxy + Syx, -Sxx + Syy - Szz, Syz + Szy],
                      [Sxy - Syx, Szx + Sxz, Syz + Szy, -Sxx - Syy + Szz]])

        eig_val, eig_vec = np.linalg.eig(N)
        q = eig_vec[:, np.argmax(eig_val)]
        qw, qx, qy, qz = q[0], q[1], q[2], q[3]
    #    rot1 = np.array([[1 - 2 * qy ** 2 - 2 * qz ** 2, 2 * qx * qy - 2 * qz * qw, 2 * qx * qz + 2 * qy * qw],
    #                     [2 * qx * qy + 2 * qz * qw, 1 - 2 * qx ** 2 - 2 * qz ** 2, 2 * qy * qz - 2 * qx * qw],
    #                     [2 * qx * qz - 2 * qy * qw, 2 * qy * qz + 2 * qx * qw, 1 - 2 * qx ** 2 - 2 * qy ** 2]])

        v = q[1:4]
        v_out = np.outer(v, v)
        Z = np.array([[qw, -qz, qy],
                      [qz, qw, -qx],
                      [-qy, qx, qw]])
        rot2 = v_out + np.dot(Z, Z)

        rot = rot2
        t = centroid_b - np.dot(rot, centroid_a)
        return rot, t

    def fit_plane(self, pts):
        print('set')


    def crossProduct(self, a, b):
        c = [a[1]*b[2] - a[2]*b[1],
             a[2]*b[0] - a[0]*b[2],
             a[0]*b[1] - a[1]*b[0]]
        return c

    def norm(self, v):
        s = 0.
        #for i in range(v.shape[0]):
        for i in range(len(v)):
            s += v[i]**2
        return np.sqrt(s)

    def pointsInBoundingBox(self, corners, point):

        area = []
        l1 = corners[1] - corners[0]
        l2 = corners[2] - corners[0]
        l_p = point - corners[0]

        # TODO: Check the computation time

        cross = self.crossProduct(l1, l_p)
        area.append(self.norm(cross)/2)
        #cross = np.cross(l1, l_p)
        #area.append(np.linalg.norm(cross)/2)
        
        cross = self.crossProduct(l2, l_p)
        area.append(self.norm(cross)/2)
        #cross = np.cross(l2, l_p)
        #area.append(np.linalg.norm(cross)/2)

        l3 = corners[1] - corners[3]
        l4 = corners[2] - corners[3]
        l_p = point - corners[3]

        cross = self.crossProduct(l3, l_p)
        area.append(self.norm(cross)/2)
        #cross = np.cross(l3, l_p)
        #area.append(np.linalg.norm(cross)/2)

        cross = self.crossProduct(l4, l_p)
        area.append(self.norm(cross)/2)
        #cross = np.cross(l4, l_p)
        #area.append(np.linalg.norm(cross)/2)
        
        total_area = sum(area)
        #self.total_area.append(total_area)
        thre = 0.005 + ((self.board_width - 1) * (self.board_height - 1) * self.board_size * self.board_size)
        # meter
        if total_area < thre:
            #print("Threshold: ", thre)
            return True
        else:
            return False

    def arucoDetect(self, markerLength):

        dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)
        a = aruco.arucoMarker(self.frame_height, markerLength, dictionary)
        while True:
            for i in self.selected_devices:
                pipe = self.pipeline[i]
                frame = pipe.wait_for_frames()
                color_frm = frame.get_color_frame()
                color_img = np.asanyarray(color_frm.get_data())
                T_cam2marker, ret = a.detect(color_img, args.aruco)
                if ret:
                    print(T_cam2marker)
                    s = cv2.FileStorage('config/cam2mobile.xml', cv2.FileStorage_WRITE)
                    s.write('R', T_cam2marker[:3, :3])
                    s.write('tvec', T_cam2marker[:3, 3])
                    s.release()
                else:
                    print('exit!')
                    exit()


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
    elif args.stereo:
        R, t = s.stereo_calibrate('images')
    elif args.refine:
        R, t = s.stereo_calibrate('images')
        s.refine_stereo_calibrate(R, t)
    elif args.aruco:
        markerLength = 70 /1000
        s.arucoDetect(markerLength)
    else:
        print('Need to check the option...')
