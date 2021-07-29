#!/usr/bin/env python3

# https://github.com/IntelRealSense/librealsense/blob/jupyter/notebooks/depth_filters.ipynb

import numpy as np
import cv2
import pyrealsense2 as rs
import sys

# Setup:
pipe = rs.pipeline()
cfg = rs.config()
#cfg.enable_device_from_file("stairs.bag")

profile = pipe.start(cfg)

# Skip 5 first frames to give the Auto-Exposure time to adjust
for x in range(5):
    pipe.wait_for_frames()
  

decimation = rs.decimation_filter()
decimation.set_option(rs.option.filter_magnitude, 1)

spatial = rs.spatial_filter()
spatial.set_option(rs.option.holes_fill, 3)

temporal = rs.temporal_filter()


while True:

    # Store next frameset for later processing:

    for x in range(10):
        frameset = pipe.wait_for_frames()
        temp_filtered = temporal.process(frameset.get_depth_frame())

    depth_frame = temp_filtered
    frameset = pipe.wait_for_frames()
    depth_frame = frameset.get_depth_frame()

    colorizer = rs.colorizer()
    colorized_depth = np.asanyarray(colorizer.colorize(depth_frame).get_data())

    #cv2.imshow('test', colorized_depth)


#    decimated_depth = decimation.process(depth_frame)
#    colorized_depth = np.asanyarray(colorizer.colorize(decimated_depth).get_data())
#    cv2.imshow('dec', colorized_depth)

#    spatial.set_option(rs.option.filter_magnitude, 5)
#    spatial.set_option(rs.option.filter_smooth_alpha, 1)
#    spatial.set_option(rs.option.filter_smooth_delta, 50)
#
#    filtered_depth = spatial.process(depth_frame)
#    colorized_depth = np.asanyarray(colorizer.colorize(filtered_depth).get_data())

    cv2.imshow('spa', colorized_depth)
    key = cv2.waitKey(1)
    if key == 27:
        sys.exit(1)
