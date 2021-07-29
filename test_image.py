#!/usr/bin/env python
from __future__ import print_function
import roslib
import sys
import rospy
import cv2
from std_msgs.msg import String
from sensor_msgs.msg import Image
import sensor_msgs.point_cloud2 as pc2

from cv_bridge import CvBridge, CvBridgeError

import message_filters

import open3d as o3d
from open3d_ros_helper import open3d_ros_helper as orh
import numpy as np
from sensor_msgs import point_cloud2
import pudb


class image_converter:
    def __init__(self):
        rospy.init_node('image_converter', anonymous=True)
        self.bridge = CvBridge()
        self.image_pub = rospy.Publisher("checker_image", Image, queue_size=10)
        self.image_sub = message_filters.Subscriber("/cam_1/color/image_raw", Image)
        self.pc_sub = message_filters.Subscriber('/cam_1/depth_registered/points', pc2.PointCloud2)
        #self.pc_sub = message_filters.Subscriber('/transformed_1', pc2.PointCloud2)
        ts = message_filters.ApproximateTimeSynchronizer([self.image_sub, self.pc_sub], 1, 1)
        ts.registerCallback(self.callback)
        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        self.board_width = 11
        self.board_height = 8


    def crop(self, roi, pc):
        print(pc.shape)


    def callback(self, img_data, pc_data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(img_data, "bgr8")
            gray_img = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        except CvBridgeError as e:
            print(e)

        found_board, corners = cv2.findChessboardCorners(gray_img, (self.board_width, self.board_height), None)

        roi = []
        print('tt')

        if found_board: 

            corners2 = cv2.cornerSubPix(gray_img, corners, (11, 11), (-1, -1), self.criteria)
            left_top_u = int(corners2[0][0][0])
            left_top_v = int(corners2[0][0][1])
            right_top_u = int(corners2[self.board_width-1][0][0])
            right_top_v = int(corners2[self.board_width-1][0][1])
            
            left_bottom_u = int(corners2[self.board_height*(self.board_width-1)-3][0][0])
            left_bottom_v = int(corners2[self.board_height*(self.board_width-1)-3][0][1])
            right_bottom_u = int(corners2[self.board_height*self.board_width-1][0][0])
            right_bottom_v = int(corners2[self.board_height*self.board_width-1][0][1])

            roi.append([left_top_u, left_top_v])
            roi.append([right_top_u, right_top_v])
            roi.append([left_bottom_u, left_bottom_v])
            roi.append([right_bottom_u, right_bottom_v])

            #print('width, height', gray_img.shape)

            cv2.circle(cv_image, (left_top_u, left_top_v), 2, (0,0,255), 2)
            cv2.circle(cv_image, (right_top_u, right_top_v), 2, (255,0,0), 2)
            cv2.circle(cv_image, (left_bottom_u, left_bottom_v), 2, (255,0,0), 2)
            cv2.circle(cv_image, (right_bottom_u, right_bottom_v), 2, (255,0,0), 2)

            gen = point_cloud2.read_points_list(pc_data, field_names=("x", "y", "z"), skip_nans=True)


#
#            left_top = (left_top_v-1) * gray_img.shape[1] + left_top_u
#            right_top = (right_top_v-1) * gray_img.shape[1] + right_top_u
#            left_bottom = (left_bottom_v-1) * gray_img.shape[1] + left_bottom_u
#            right_bottom = (right_bottom_v-1) * gray_img.shape[1] + right_bottom_u
#
#            print(gen[left_top])
#            print(gen[right_top])
#            print(gen[left_bottom])
#            print(gen[right_bottom])



#            err_lr_top = [gen[left_top].x - gen[right_top].x, 
#                        gen[left_top].y - gen[right_top].y, 
#                        gen[left_top].z - gen[right_top].z]
#
#            print(np.linalg.norm(err_lr_top))

#            for p in gen:
#                print(p)

            #gen = point_cloud2.read_points_list
            #o3dpc = orh.rospc_to_o3dpc(pc_datadata)
#            np_pc = np.array(o3dpc.points)
#            self.crop(roi, np_pc)

#            cv_image = cv2.resize(cv_image, (640, 480))
#            cv2.imshow('cam', cv_image)
#            key = cv2.waitKey(1)
            self.image_pub.publish(self.bridge.cv2_to_imgmsg(cv_image, "bgr8"))

        else:
            print('Cannot detect...')


def main(args):
    ic = image_converter() 
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main(sys.argv)
