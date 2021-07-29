#!/usr/bin/env python
import rospy
import sys
import numpy as np
import cv2
import argparse
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import CameraInfo
from open3d_ros_helper import open3d_ros_helper as orh
import message_filters

import open3d as o3d
#import tf2_ros
#import geometry_msgs.msg as gmsg
#import copy

def setArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--trans', help='Transformation matrix between two other cameras')
    args = parser.parse_args()
    return args


def preprocess_point_cloud(pcd, voxel_size):
    pcd_down = o3d.geometry.voxel_down_sample(pcd, voxel_size)
    radius_normal = voxel_size * 2
    #print(":: Estimate normal with search radius %.3f." % radius_normal)
    max_nn_neighbors = 30   # At maximum, max_nn neighbors will be searched.
    o3d.geometry.estimate_normals(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=max_nn_neighbors))

    radius_feature = voxel_size * 5
    #print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
    pcd_fpfh = o3d.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd_down, pcd_fpfh


def prepare_dataset(source, target, voxel_size):
    source_down, source_fpfh = preprocess_point_cloud(source, voxel_size)
    target_down, target_fpfh = preprocess_point_cloud(target, voxel_size)
    return source, target, source_down, target_down, source_fpfh, target_fpfh


def execute_global_registration(source_down, target_down, source_fpfh,
                                target_fpfh, voxel_size):
    distance_threshold = voxel_size * 4
    ransac_n = 4
    max_iteration = 4000000
    max_validation = 10000
    corre_edgelength = 0.9  # Similarity_threshold is a number between 0 (loose) and 1 (strict)

    print("Downsampling voxel size is %.3f," % voxel_size)
    print("Distance threshold %.3f." % distance_threshold)
    result = o3d.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, distance_threshold,
        o3d.registration.TransformationEstimationPointToPoint(False), ransac_n, [
            o3d.registration.CorrespondenceCheckerBasedOnEdgeLength(corre_edgelength),
            o3d.registration.CorrespondenceCheckerBasedOnDistance(
                distance_threshold)
        ], o3d.registration.RANSACConvergenceCriteria(max_iteration, max_validation))
    return result


def refine_registration(source, target, source_fpfh, target_fpfh, voxel_size):
    global result_ransac
    distance_threshold = voxel_size * 0.4
    print(":: Point-to-plane ICP registration is applied on original point")
    print("   clouds to refine the alignment. This time we use a strict")
    print("   distance threshold %.3f." % distance_threshold)
    result = o3d.registration.registration_icp(
        source, target, distance_threshold, result_ransac.transformation,
        o3d.registration.TransformationEstimationPointToPlane())
    return result


def homogeneousInverse(T):
    invT = np.eye(4)
    R = T[:3, :3]
    t = T[:3, 3]
    t = np.expand_dims(t, axis=1)
    invT[:3, :3] = R.T
    invT[:3, 3] = np.squeeze(-R.T.dot(t), axis=1)
    return invT


def callback_cam(pc2_msg): 
    #global required_ransac, cnt_ransac

    o3dpc = orh.rospc_to_o3dpc(pc2_msg)
    T_rgb_to_ir_base, T_rgb_to_ir_side = np.eye(4), np.eye(4)

    global rgb_to_ir_basecam, rgb_to_ir_sidecam

    T_rgb_to_ir_base[:3, :3] = np.reshape(rgb_to_ir_basecam[:-3], (3, 3))
    T_rgb_to_ir_base[:3, 3] = rgb_to_ir_basecam[-3:]
    T_ir_to_rgb_base = homogeneousInverse(T_rgb_to_ir_base)

    #T_btw_cameras_ = t_btw_cameras.dot(R_btw_cameras)

    T_rgb_to_ir_side[:3, :3] = np.reshape(rgb_to_ir_sidecam[:-3], (3, 3))
    T_rgb_to_ir_side[:3, 3] = rgb_to_ir_sidecam[-3:]
    
    T_btw_cameras = T_ir_to_rgb_base.dot(T_btw_cameras_).dot(T_rgb_to_ir_side)

    # Transform the point cloud data from cam 2 coordinate to cam 1 coordinate
    o3dpc.transform(T_btw_cameras)
    rospc = orh.o3dpc_to_rospc(o3dpc)
    rospc.header.frame_id = base_coordi
    pub.publish(rospc)


def callback_cam01(pc2_msg0, pc2_msg1):
    o3dpc0 = orh.rospc_to_o3dpc(pc2_msg0)
    o3dpc1 = orh.rospc_to_o3dpc(pc2_msg1)
    np_pc0 = np.array(o3dpc0.points)
    np_pc1 = np.array(o3dpc1.points)
    np_pc = np.concatenate((np_pc0, np_pc1), axis=0)
    pcd.points = o3d.utility.Vector3dVector(np_pc)
    rospc = orh.o3dpc_to_rospc(pcd)
    rospc.header.frame_id = 'cam_0_link'
    pub.publish(rospc)


if __name__ == '__main__':

    print('Start!')
    pcds = [] 
    pcd = o3d.geometry.PointCloud()

    base_cam = 'cam_0_zf'
    side_cam = '1'

    rospy.init_node('refine', disable_signals=True)

#    cam0_sub = message_filters.Subscriber('/cam_0/color/image_raw', 
#    cam1_sub = message_filters.Subscriber('/transformed_' + side_cam, 

    cam0_sub = message_filters.Subscriber('/' + base_cam, pc2.PointCloud2)
    cam1_sub = message_filters.Subscriber('/transformed_' + side_cam, pc2.PointCloud2)
    ts = message_filters.ApproximateTimeSynchronizer([cam0_sub, cam1_sub], 1, 1)
    ts.registerCallback(callback_cam01)
    pub = rospy.Publisher('/refine', pc2.PointCloud2, queue_size=10)
    rospy.spin()
