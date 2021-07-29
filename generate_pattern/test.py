#!/usr/bin/env python3

import socket
#import rospy
#from geometry_msgs.msg import Pose
#from std_msgs.msg import Bool, String
import cv2
from cv2 import aruco
import numpy as np
import pyrealsense2 as rs
from scipy.spatial.transform import Rotation as R


pipe = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 1280, 720, rs.format.rgb8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

profile = pipe.start(config)
#depth_sensor = profile.get_device().first_depth_sensor()
#depth_scale = depth_sensor.get_depth_scale()
align_to = rs.stream.color
align = rs.align(align_to)
frames = pipe.wait_for_frames()
aligned_frames = align.process(frames)

def get_intrinsic(frame):
    intrinsics = frame.profile.as_video_stream_profile().intrinsics
    camera_matrix = np.array([[intrinsics.fx, 0, intrinsics.ppx],
                            [0, intrinsics.fy, intrinsics.ppy],
                            [0,             0,              1]])
    return camera_matrix

def get_frame():
    frames = pipe.wait_for_frames()
    aligned_frames = align.process(frames)
    # Get aligned frames
    # internal_calibration = o3d.camera.PinholeCameraIntrinsic(get_intrinsic(aligned_frames)).intrinsic_matrix
    aligned_depth_frame = aligned_frames.get_depth_frame() # aligned_depth_frame is a 640x480 depth image
    color_frame = aligned_frames.get_color_frame()
    depth_intrin = aligned_depth_frame.profile.as_video_stream_profile().intrinsics
    # Convert images to numpy arrays
    depth_image = np.asanyarray(aligned_depth_frame.get_data())
    # depth_image = np.asanyarray(depth_frame.get_data())
    color_image = np.asanyarray(color_frame.get_data())
    color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
    return color_image

def set_str_range(float_):
    if str(float_)[0] != '-':
        if len(str(float_)) == 3:
            str_ = str(float_) + '00000'
        elif len(str(float_)) == 4:
            str_ = str(float_) + '0000'
        elif len(str(float_)) == 5:
            str_ = str(float_) + '000'
        elif len(str(float_)) == 6:
            str_ = str(float_) + '00'
        else:
            str_ =str(float_) + '0'
    else:
        if len(str(float_)[1:]) == 3:
            str_ = str(float_) + '0000'
        if len(str(float_)[1:]) == 4:
            str_ = str(float_) + '000'
        elif len(str(float_)[1:]) == 5:
            str_ = str(float_)+ '00'
        elif len(str(float_)[1:]) == 6:
            str_ = str(float_)+ '0'
        else:
            str_ =str(float_)
    if len(str_) != 8:
        print('connection fail!!!!!!!')
    return str_


internal_calibration = get_intrinsic(aligned_frames)
camera_matrix = internal_calibration
dist_coeffs = np.asarray([[0.0],[0.0],[0.0],[0.0]])
new_camera_matrix = internal_calibration
#image_size = (1280, 720)

aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)

markerLength = 157/1000 # Here, our measurement unit is centimetre.
markerSeparation = 19.5/1000 # Here, our measurement unit is centimetre.

board = aruco.GridBoard_create(1, 1, markerLength, markerSeparation, aruco_dict)
arucoParams = aruco.DetectorParameters_create()

rvec = np.asarray([0, 0, 0])
tvec = np.asarray([0, 0, 0])
#pub = rospy.Publisher('target_pose', Pose, queue_size=100)
#rospy.init_node('ARUCO_VISION', anonymous=True)

while True:
    frame = get_frame() # Capture frame-by-frame
    frame_remapped_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, rejectedImgPoints = aruco.detectMarkers(frame_remapped_gray, aruco_dict,
    parameters=arucoParams,
    cameraMatrix=new_camera_matrix,
    distCoeff=dist_coeffs)
    aruco.refineDetectedMarkers(frame_remapped_gray, board, corners, ids, rejectedImgPoints)

    try: # if there is at least one marker detected
        im_with_aruco_board = aruco.drawDetectedMarkers(frame, corners, ids, (0, 255, 0))
        rvec, tvec, retval = aruco.estimatePoseSingleMarkers(corners, markerLength, camera_matrix, np.zeros((5, 1)))

        rvec = rvec[0]
        tvec = tvec[0]
        if retval is not None:
            im_with_aruco_board = aruco.drawAxis(im_with_aruco_board, camera_matrix, dist_coeffs, rvec, tvec, 100/1000) # axis length 100 can be changed according to your requirement

            r_temp = R.from_rotvec(np.array([rvec[0][0], rvec[0][1], rvec[0][2]]))
            r_quat = r_temp.as_quat()

            pose_goal = Pose()
            goal_position = pose_goal.position
            goal_position.x, goal_position.y, goal_position.z = tvec[0][0], tvec[0][1], tvec[0][2]
            goal_ori = pose_goal.orientation
            goal_ori.x, goal_ori.y, goal_ori.z, goal_ori.w = r_quat[0], r_quat[1], r_quat[2], r_quat[3]
#            pub.publish(pose_goal)
            tvec=tvec[0]

            for t in range(len(tvec)):
                tvec[t]=round(tvec[t],4)
            x_str = set_str_range(tvec[0])
            y_str = set_str_range(tvec[1])
            z_str = set_str_range(tvec[2])
            #message = 'test'
            messae = xstr + y_str + z_str

            print("##### translate #####")
            print(tvec)
            print("##### euler #####")
            print(rvec)
            print("##### quat #####")
            print(r_quat)
    except:
        im_with_aruco_board = frame

    cv2.imshow("arucoboard", im_with_aruco_board)

    if cv2.waitKey(2) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()



