#!/usr/bin/env python
from __future__ import print_function
import roslib
import sys
import rospy
import cv2
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError


class image_converter:
    def __init__(self):
        rospy.init_node('image_converter', anonymous=True)
        self.bridge = CvBridge()
        self.image_pub = rospy.Publisher("image_topic_2", Image, queue_size=10)
        self.image_sub = rospy.Subscriber("/cam_0/color/image_raw", Image, self.callback)
        self.n_img = 0
        self.total_cnt = 10
        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        self.board_width = 8
        self.board_height = 5

    def callback(self, data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)

        (rows, cols, channels) = cv_image.shape

        #cv2.circle(cv_image, (int(cols/2),int(rows/2)), 10, 255)
        #cv2.imshow("Image window", cv_image)
        #cv2.waitKey(3)


#        dir_name = 'config'
#        if os.path.exists(dir_name):
#            shutil.rmtree(dir_name)
#        os.makedirs(dir_name, exist_ok=True)

        while self.n_img < self.total_cnt:
            gray_img = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
            found_board, corners = cv2.findChessboardCorners(gray_img, (self.board_width, self.board_height), None)

            if found_board: 
                print('founded!')
                corners2 = cv2.cornerSubPix(gray_img, corners, (11, 11), (-1, -1), self.criteria)
                cv2.drawChessboardCorners(gray_img, (self.board_width, self.board_height), corners2, found_board)
    #            self.imgpoints.append(corners2)
    #            self.objpoints.append(self.objp)
                gray_img = cv2.resize(gray_img, (640, 480))
                print(n_img + 1, '/', self.total_count)
                n_img += 1
                cv2.imshow('cam', gray_img)
                key = cv2.waitKey(1000)
                if key == 27:
                    cv2.destroyAllWindows()
                    pipe.stop()
                    sys.exit(0)
            else:
                print('cannot find')
                cv2.imshow("Image window", gray_img)
                cv2.waitKey(3)

        cv2.destroyAllWindows()
        print('finished.')

#        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(self.objpoints, self.imgpoints, (self.frame_width, self.frame_height), None, None)
#
#        # Re-projection error
#        mean_error = 0
#        for i in range(len(self.objpoints)):
#            imgpoints2, _ = cv2.projectPoints(self.objpoints[i], rvecs[i], tvecs[i], mtx, dist)
#            error = cv2.norm(self.imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
#            mean_error += error
#        print("Reprojection error: {}".format(mean_error / len(self.objpoints)))

        # Save camera parameters
#        s = cv2.FileStorage('config/cam_calib_'+str(idx_pipe)+'.xml', cv2.FileStorage_WRITE)
#        s.write('mtx', mtx)
#        s.write('dist', dist)
#        s.release()
#    pipe.stop()

        
        try: 
            self.image_pub.publish(self.bridge.cv2_to_imgmsg(cv_image, "bgr8"))
        except CvBridgeError as e:
            print(e)


def main(args):
    ic = image_converter() 
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main(sys.argv)
