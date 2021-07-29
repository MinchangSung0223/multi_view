#!/usr/bin/env python3

import pyrealsense2 as rs
import numpy as np
import cv2
import sys
import pudb

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.infrared, 1, 640, 480, rs.format.y8, 30)
config.enable_stream(rs.stream.infrared, 2, 640, 480, rs.format.y8, 30)

# Start streaming
pipeline_profile = pipeline.start(config)
device = pipeline_profile.get_device()
depth_sensor = device.query_sensors()[0]
laser_pwr = depth_sensor.get_option(rs.option.laser_power)
print("laser power = ", laser_pwr)
laser_range = depth_sensor.get_option_range(rs.option.laser_power)
print("laser power range = " , laser_range.min , "~", laser_range.max)
depth_sensor.set_option(rs.option.laser_power, 0)

ir_image = np.zeros((2, 480, 640), dtype=np.uint8)

while True:
    frames = pipeline.wait_for_frames()
    for i, frame in enumerate(frames):
        ftype = frame.profile.stream_type()
        if ftype == rs.stream.infrared:
            ir_image[i-1, :] = np.asanyarray(frame.get_data())
        else:
            rgb_image = np.asanyarray(frame.get_data())
    
    ir_color_0 = cv2.cvtColor(ir_image[0], cv2.COLOR_GRAY2BGR)
    ir_color_1 = cv2.cvtColor(ir_image[1], cv2.COLOR_GRAY2BGR)

    # Stack both images horizontally
    images = np.hstack((ir_color_0, ir_color_1, rgb_image))

    # Show images
    cv2.namedWindow("RealSense", cv2.WINDOW_AUTOSIZE)
    cv2.imshow("RealSense", images)
    key = cv2.waitKey(1)
    if key == 27:
        cv2.destroyAllWindows()
        pipeline.stop()
        sys.exit(0)
