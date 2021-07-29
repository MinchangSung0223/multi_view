#!/usr/bin/env python3

import pyrealsense2 as rs
import numpy as np
import cv2
import scipy.misc
import pcl


def get_image():
    # Configure depth and color streams
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    # Start streaming
    pipeline.start(config)

    #Get the image, the image will be a little distorted when realsense is just started, we save the 100th frame picture.
    for i in range(100):
        data = pipeline.wait_for_frames()
        depth = data.get_depth_frame()
        color = data.get_color_frame()

    #Get internal reference
    dprofile = depth.get_profile()
    cprofile = color.get_profile()

    cvsprofile = rs.video_stream_profile(cprofile)
    dvsprofile = rs.video_stream_profile(dprofile)

    color_intrin=cvsprofile.get_intrinsics()
    print(color_intrin)
    depth_intrin=dvsprofile.get_intrinsics()
    print(color_intrin)
    extrin = dprofile.get_extrinsics_to(cprofile)
    print(extrin)

    depth_image = np.asanyarray(depth.get_data())
    color_image = np.asanyarray(color.get_data())

    # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
    depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

    cv2.imwrite('color.png', color_image)
    cv2.imwrite('depth.png', depth_image)
    cv2.imwrite('depth_colorMAP.png', depth_colormap)
    scipy.misc.imsave('outfile1.png', depth_image)
    scipy.misc.imsave('outfile2.png', color_image)




def my_depth_to_cloud():
    pc = rs.pointcloud()
    points = rs.points()
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 1920, 1080, rs.format.bgr8, 30)
    pipe_profile = pipeline.start(config)

    for i in range(100):
        data = pipeline.wait_for_frames()

        depth = data.get_depth_frame()
        color = data.get_color_frame()

    frames = pipeline.wait_for_frames()
    depth = frames.get_depth_frame()
    color = frames.get_color_frame()

    colorful = np.asanyarray(color.get_data())
    colorful=colorful.reshape(-1,3)

    pc.map_to(color)
    points = pc.calculate(depth)

    #Get vertex coordinates
    vtx = np.asanyarray(points.get_vertices())
    #Get texture coordinates
    #tex = np.asanyarray(points.get_texture_coordinates())



    

    with open('could.txt','w') as f:
        for i in range(len(vtx)):
            f.write(str(np.float(vtx[i][0])*1000)+' '+str(np.float(vtx[i][1])*1000)+' '+str(np.float(vtx[i][2])*1000)+' '+str(np.float(colorful[i][0]))+' '+str(np.float(colorful[i][1]))+' '+str(np.float(colorful[i][2]))+'\n')




if __name__ == "__main__":


    my_depth_to_cloud()

#
#    # Configure depth and color streams
#    pipeline = rs.pipeline()
#    config = rs.config()
#    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
#    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
#    # Start streaming
#    pipeline.start(config)
#
#    try:
#        while True:
#            # Wait for a coherent pair of frames: depth and color
#            frames = pipeline.wait_for_frames()
#            depth_frame = frames.get_depth_frame()
#            color_frame = frames.get_color_frame()
#            if not depth_frame or not color_frame:
#                continue
#            # Convert images to numpy arrays
#
#            depth_image = np.asanyarray(depth_frame.get_data())
#            color_image = np.asanyarray(color_frame.get_data())
#
#            # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
#            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
#
#            # Stack both images horizontally
#            images = np.hstack((color_image, depth_colormap))
#            # Show images
#            cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
#            cv2.imshow('RealSense', images)
#            key = cv2.waitKey(1)
#            # Press esc or 'q' to close the image window
#            if key & 0xFF == ord('q') or key == 27:
#                cv2.destroyAllWindows()
#                break
#    finally:
#        # Stop streaming
#        pipeline.stop()
