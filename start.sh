#!/bin/bash
IFS=$'\n'  
ARR=(`rs-enumerate-devices | grep "Serial Number" | sed 's/[^0-9]//g'`)
roslaunch realsense2_camera rs_camera.launch camera:=cam_0 serial_no:=${ARR[0]} filters:=pointcloud &
sleep 10s
roslaunch realsense2_camera rs_camera.launch camera:=cam_1 serial_no:=${ARR[2]} filters:=pointcloud &
sleep 10s
cp extract config/extract;
./config/extract &
sleep 10s
python base_zf.py cam_0 &
sleep 5s
python send_transform.py config/trans01.xml &
sleep 5s
python merge.py 
