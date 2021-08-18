#!/usr/bin/env python3

import open3d as o3d
import numpy as np
import copy
import pudb
import cv2
import argparse

def setArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument('ply_file', nargs='*', metavar='PLY_FILE_NAME', help='ply file. ')
#    parser.add_argument('-m', '--multiple', action='store_true', help='A number of experiments')
#    parser.add_argument('cam_calibration_file', metavar='CAMERA_CALIBRATION_FILE_NAME', help='The file name of the camera calibration')
#    parser.add_argument('-r', '--observ_rank', action='store_true', help='Rank of the matrix')
#    parser.add_argument('-p', '--observ_pose', action='store_true', help='Use redundant stations')
#    parser.add_argument('-d', '--observ_distance', action='store_true', help='Use different distance between stations')
#    parser.add_argument('-e', '--exclude_method', nargs='*', default=0, help='Exclude the specific methods(1: Tsai, 2: Park, 3: Horaud, 4: Andreff, 5: Daniilidis, 6: Ali(Xc1), 7: Ali(Xc2), 8: Horaud(Nonlinear), 9: Proposed(e1), 10: Proposed(e2), 11: Li, 12: Shah, 13: Tabb(Zc1), 14: Tabb(Zc2), 15: Shin(cay), 16: Tabb(rp1), 17: Shin(rp))') 
#    parser.add_argument('-v', '--verification', action='store_true', help='Use a verification in other poses')
    args = parser.parse_args()
    return args


class PointSetRegistration:
    def __init__(self):
        print('hello')

    def prepare_dataset(self, file_name, voxel_size):
        print(":: Load two point clouds and disturb initial pose.")
        num_ply_file = len(file_name)
        ply = []
        for i in range(num_ply_file):
            ply.append(o3d.io.read_point_cloud(file_name[i]))
            
        if len(file_name) == 2:
            self.source = ply[0]
            self.target = ply[1]
            o3d.visualization.draw_geometries([self.source, self.target])
            #self.draw_registration_result(self.source, self.target, np.identity(4))
            source_down, source_fpfh = self.preprocess_point_cloud(self.source, voxel_size)
            target_down, target_fpfh = self.preprocess_point_cloud(self.target, voxel_size)
            return self.source, self.target, source_down, target_down, source_fpfh, target_fpfh
        else:
            for ply_ in ply:
                o3d.visualization.draw_geometries([ply_])
            print('To registration the pointcloud datasets, write two ply files.')

    def draw_registration_result(self, source, target, transformation):
        source_temp = copy.deepcopy(source)
        target_temp = copy.deepcopy(target)
        source_temp.paint_uniform_color([1, 0.706, 0])
        target_temp.paint_uniform_color([0, 0.651, 0.929])
        source_temp.transform(transformation)
        o3d.visualization.draw_geometries([source_temp, target_temp])

    def preprocess_point_cloud(self, pcd, voxel_size):
        print(":: Downsample with a voxel size %.3f." % voxel_size)
        pcd_down = pcd.voxel_down_sample(voxel_size)

        radius_normal = voxel_size * 2
        print(":: Estimate normal with search radius %.3f." % radius_normal)
        pcd_down.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

        radius_feature = voxel_size * 5
        print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
        pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            pcd_down,
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
        return pcd_down, pcd_fpfh

    def execute_global_registration(self, source_down, target_down, source_fpfh, target_fpfh, voxel_size):
        distance_threshold = voxel_size * 1.5
        print(":: RANSAC registration on downsampled point clouds.")
        print("   Since the downsampling voxel size is %.3f," % voxel_size)
        print("   we use a liberal distance threshold %.3f." % distance_threshold)
        result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
            source_down, target_down, source_fpfh, target_fpfh, True, distance_threshold,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(False), 3, 
            [o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength( 0.9),
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)], 
                o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999))
        return result

    def transform_points(self, T):
        source_temp = copy.deepcopy(self.source)
        target_temp = copy.deepcopy(self.target)
        #source_temp.paint_uniform_color([1, 0.706, 0])
        #target_temp.paint_uniform_color([0, 0.651, 0.929])
        target_temp.transform(T)
        o3d.visualization.draw_geometries([source_temp, target_temp])
        return source_temp, target_temp

    def fit_plane(self, src_pcd, tg_pcd):
        source_temp = src_pcd
        target_temp = tg_pcd
        #source_temp = copy.deepcopy(self.source)
        #target_temp = copy.deepcopy(self.target)

        src_plane_model, src_inliers = source_temp.segment_plane(distance_threshold=0.001, ransac_n=3, num_iterations=1000)
        [src_a, src_b, src_c, src_d] = src_plane_model
        print(f"Source Plane equation: {src_a:.2f}x + {src_b:.2f}y + {src_c:.2f}z + {src_d:.2f} = 0")

        src_inlier_cloud = source_temp.select_by_index(src_inliers)
        src_inlier_cloud.paint_uniform_color([1.0, 0, 0])

        tg_plane_model, tg_inliers = target_temp.segment_plane(distance_threshold=0.001, ransac_n=3, num_iterations=1000)
        [tg_a, tg_b, tg_c, tg_d] = tg_plane_model
        print(f"Target Plane equation: {tg_a:.2f}x + {tg_b:.2f}y + {tg_c:.2f}z + {tg_d:.2f} = 0")

        tg_inlier_cloud = target_temp.select_by_index(tg_inliers)
        tg_inlier_cloud.paint_uniform_color([0, 1.0, 0])

        #outlier_cloud = self.source.select_by_index(inliers, invert=True)
        #o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])
        o3d.visualization.draw_geometries([src_inlier_cloud, tg_inlier_cloud])

        src_n = np.array([src_a, src_b, src_c])
        tg_n = np.array([tg_a, tg_b, tg_c])
        dot_res = src_n.dot(tg_n)
        crs_res = np.cross(src_n, tg_n)
        #norm_vec = np.linalg.norm(src_n) * np.linalg.norm(tg_c)
        ang_btw_planes = np.arccos(dot_res)
        print('Angle between two planes[deg]: ', np.rad2deg(ang_btw_planes))
        print('Rotation vector between two planes: ', crs_res)

        T_two_planes = np.eye(4)
        cv2.Rodrigues(crs_res, T_two_planes[:3, :3])
        print('Rotate to parallel two planes')
        src_inlier_cloud.transform(T_two_planes)
        o3d.visualization.draw_geometries([src_inlier_cloud, tg_inlier_cloud])

if __name__ == '__main__':
    args = setArgs()

    voxel_size = 0.01
    print(args.ply_file)
    pc = PointSetRegistration()
    pc.prepare_dataset(args.ply_file, voxel_size)

    #pc.fit_plane()

    filename = 'config/trans01.xml'
    s = cv2.FileStorage()
    s.open(filename, cv2.FileStorage_READ)
    T_btw_cameras = np.eye(4)
    T_btw_cameras[:3, :3] = s.getNode('R').mat()
    T_btw_cameras[:3, 3] = s.getNode('tvec').mat().reshape((3))
    print(filename)
    print(T_btw_cameras)
    src_1, tg_1 = pc.transform_points(T_btw_cameras)
    pc.fit_plane(src_1, tg_1)

    filename = 'config/new_trans01.xml'
    s = cv2.FileStorage()
    s.open(filename, cv2.FileStorage_READ)
    T_btw_cameras_ = np.eye(4)
    T_btw_cameras_[:3, :3] = s.getNode('R').mat()
    T_btw_cameras_[:3, 3] = s.getNode('tvec').mat().reshape((3))
    print(filename)
    print(T_btw_cameras_)
    src_2, tg_2 = pc.transform_points(T_btw_cameras_)
    pc.fit_plane(src_2, tg_2)

    print('between two matrix')
    print(T_btw_cameras.dot(np.linalg.inv(T_btw_cameras_)))




#voxel_size = 0.005  # means 5mm for this dataset
#source, target, source_down, target_down, source_fpfh, target_fpfh = prepare_dataset(voxel_size)
#transform_points(T_btw_cameras)
#
#
#filename = 'config/new_trans01.xml'
#s = cv2.FileStorage()
#s.open(filename, cv2.FileStorage_READ)
#T_btw_cameras = np.eye(4)
#T_btw_cameras[:3, :3] = s.getNode('R').mat()
#T_btw_cameras[:3, 3] = s.getNode('tvec').mat().reshape((3))
#voxel_size = 0.005  # means 5mm for this dataset
#source, target, source_down, target_down, source_fpfh, target_fpfh = prepare_dataset(voxel_size)
#transform_points(T_btw_cameras)
#
#
#
#result_ransac = execute_global_registration(source_down, target_down,
#                                            source_fpfh, target_fpfh,
#                                            voxel_size)
#print(result_ransac)
#draw_registration_result(source_down, target_down, result_ransac.transformation)
#
#
#pcd = source
#plane_model, inliers = pcd.segment_plane(distance_threshold=0.01,
#                                         ransac_n=3,
#                                         num_iterations=1000)
#[a, b, c, d] = plane_model
#print(f"Plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")
#
#inlier_cloud = pcd.select_by_index(inliers)
#inlier_cloud.paint_uniform_color([1.0, 0, 0])
#outlier_cloud = pcd.select_by_index(inliers, invert=True)
#o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])
#
#
#pcd = target
#plane_model, inliers = pcd.segment_plane(distance_threshold=0.01,
#                                         ransac_n=3,
#                                         num_iterations=1000)
#[a, b, c, d] = plane_model
#print(f"Plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")
#
#inlier_cloud = pcd.select_by_index(inliers)
#inlier_cloud.paint_uniform_color([1.0, 0, 0])
#outlier_cloud = pcd.select_by_index(inliers, invert=True)
#o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])
#
#
#
#
#
#def draw_registration_result_original_color(source, target, transformation):
#    source_temp = copy.deepcopy(source)
#    source_temp.transform(transformation)
#    o3d.visualization.draw_geometries([source_temp, target],
#                                      zoom=0.5,
#                                      front=[-0.2458, -0.8088, 0.5342],
#                                      lookat=[1.7745, 2.2305, 0.9787],
#                                      up=[0.3109, -0.5878, -0.7468])
#
#
#
## colored pointcloud registration
## This is implementation of following paper
## J. Park, Q.-Y. Zhou, V. Koltun,
## Colored Point Cloud Registration Revisited, ICCV 2017
##voxel_radius = [0.04, 0.02, 0.01]
##max_iter = [50, 30, 14]
##current_transformation = np.identity(4)
##print("3. Colored point cloud registration")
##for scale in range(3):
##    iter = max_iter[scale]
##    radius = voxel_radius[scale]
##    print([iter, radius, scale])
##
##    print("3-1. Downsample with a voxel size %.2f" % radius)
##    source_down = source.voxel_down_sample(radius)
##    target_down = target.voxel_down_sample(radius)
##    pu.db
##
##    print("3-2. Estimate normal.")
##    source_down.estimate_normals(
##        o3d.geometry.KDTreeSearchParamHybrid(radius=radius * 2, max_nn=30))
##    target_down.estimate_normals(
##        o3d.geometry.KDTreeSearchParamHybrid(radius=radius * 2, max_nn=30))
##
##    print("3-3. Applying colored point cloud registration")
##    result_icp = o3d.pipelines.registration.registration_colored_icp(
##        source_down, target_down, radius, current_transformation,
##        o3d.pipelines.registration.TransformationEstimationForColoredICP(),
##        o3d.pipelines.registration.ICPConvergenceCriteria(relative_fitness=1e-6,
##                                                          relative_rmse=1e-6,
##                                                          max_iteration=iter))
##    current_transformation = result_icp.transformation
##    print(result_icp)
##draw_registration_result_original_color(source, target,
##                                        result_icp.transformation)
#
#
##
##print("Apply point-to-point ICP")
##reg_p2p = o3d.pipelines.registration.registration_icp(
##    source, target, threshold, trans_init,
##    o3d.pipelines.registration.TransformationEstimationPointToPoint())
##print(reg_p2p)
##print("Transformation is:")
##print(reg_p2p.transformation)
##draw_registration_result(source, target, reg_p2p.transformation)
##
##reg_p2p = o3d.pipelines.registration.registration_icp(
##    source, target, threshold, trans_init,
##    o3d.pipelines.registration.TransformationEstimationPointToPoint(),
##    o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=2000))
##print(reg_p2p)
##print("Transformation is:")
##print(reg_p2p.transformation)
##draw_registration_result(source, target, reg_p2p.transformation)
##
##
##
##
##print("Apply point-to-plane ICP")
##reg_p2l = o3d.pipelines.registration.registration_icp(
##    source, target, threshold, trans_init,
##    o3d.pipelines.registration.TransformationEstimationPointToPlane())
##print(reg_p2l)
##print("Transformation is:")
##print(reg_p2l.transformation)
##draw_registration_result(source, target, reg_p2l.transformation)
##
##
##
