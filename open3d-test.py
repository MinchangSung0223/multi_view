#!/usr/bin/env python3

import open3d as o3d
import numpy as np
import copy
import pudb
import cv2
import argparse
import time
import scipy.optimize
from pytransform3d.rotations import *


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
        print(f"1. Source Plane equation: {src_a:.2f}x + {src_b:.2f}y + {src_c:.2f}z + {src_d:.2f} = 0")

        src_inlier_cloud = source_temp.select_by_index(src_inliers)
        src_inlier_cloud.paint_uniform_color([1.0, 0, 0])

        tg_plane_model, tg_inliers = target_temp.segment_plane(distance_threshold=0.001, ransac_n=3, num_iterations=1000)
        [tg_a, tg_b, tg_c, tg_d] = tg_plane_model
        print(f"2. Target Plane equation: {tg_a:.2f}x + {tg_b:.2f}y + {tg_c:.2f}z + {tg_d:.2f} = 0")

        tg_inlier_cloud = target_temp.select_by_index(tg_inliers)
        tg_inlier_cloud.paint_uniform_color([0, 1.0, 0])

        #outlier_cloud = self.source.select_by_index(inliers, invert=True)
        #o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])



        #o3d.visualization.draw_geometries([src_inlier_cloud, tg_inlier_cloud])

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
        print('Rotation matrix:', T_two_planes[:3, :3])
        src_inlier_cloud.transform(T_two_planes)
        source_temp.transform(T_two_planes)


        #o3d.visualization.draw_geometries([src_inlier_cloud, tg_inlier_cloud])


        # o3d.visualization.draw_geometries([source_temp, target_temp])

#        transformed_src_plane_model, transformed_src_inliers = src_inlier_cloud.segment_plane(distance_threshold=0.001, ransac_n=3, num_iterations=1000)
#        [t_src_a, t_src_b, t_src_c, t_src_d] = transformed_src_plane_model
#        print(f"3. Source(Transformed) Plane equation: {t_src_a:.2f}x + {t_src_b:.2f}y + {t_src_c:.2f}z + {t_src_d:.2f} = 0")
#
#        dist_btw_two_planes = abs(t_src_d - tg_d) / np.sqrt(np.power(tg_a, 2) + np.power(tg_b, 2) + np.power(tg_c, 2))
#        print('Distance between two planes: ', dist_btw_two_planes)
#
#        T_two_planes_trans = np.eye(4)
#        T_two_planes_trans[:3, 3] = np.array([t_src_a, t_src_b, t_src_c]) * - dist_btw_two_planes
#        src_inlier_cloud.transform(T_two_planes_trans)
#        print('Translate to align two planes.')
#        o3d.visualization.draw_geometries([src_inlier_cloud, tg_inlier_cloud])


#        print('Principal Component Analysis(src)')
#        src_mean, src_cov = o3d.geometry.PointCloud.compute_mean_and_covariance(src_inlier_cloud)
#        src_pca = np.linalg.eig(src_cov)
#        src_evalue = src_pca[0]
#        src_evector = src_pca[1].T
#        src_points = [ src_mean, src_mean + src_evector[0], src_mean + src_evector[1], src_mean + src_evector[2], ]
#        src_lines = [ [0, 1], [0, 2], [0, 3], ]
#        line_set = o3d.geometry.LineSet( points=o3d.utility.Vector3dVector(src_points), lines=o3d.utility.Vector2iVector(src_lines),)
#        o3d.visualization.draw_geometries([line_set, src_inlier_cloud])

#        num_ply_file = ['cloud0_corners.ply', 'cloud1_corners.ply']
#        ply = []
#        for i in num_ply_file:
#            ply.append(o3d.io.read_point_cloud(i))
#            
#        ply[1].transform(T_btw_cameras_)
#        src_corners = np.asarray(ply[0].points)
#        tg_corners = np.asarray(ply[1].points)
#        src_center = np.mean(src_corners, axis=0)
#        tg_center = np.mean(tg_corners, axis=0)
#        tvec_btw_centers = - tg_center + src_center
#        T_two_planes_trans = np.eye(4)
#        T_two_planes_trans[:3, 3] = tvec_btw_centers
#        src_inlier_cloud.transform(T_two_planes_trans)
#        source_temp.transform(T_two_planes_trans)
#        print('Translate to align two planes.')
#        #o3d.visualization.draw_geometries([src_inlier_cloud, tg_inlier_cloud])
#        o3d.visualization.draw_geometries([source_temp, target_temp])


if __name__ == '__main__':
    args = setArgs()

    voxel_size = 0.01
    print(args.ply_file)
    pc = PointSetRegistration()
    pc.prepare_dataset(args.ply_file, voxel_size)

    filename = 'config/trans01.xml'
    s = cv2.FileStorage()
    s.open(filename, cv2.FileStorage_READ)
    T_btw_cameras = np.eye(4)
    T_btw_cameras[:3, :3] = s.getNode('R').mat()
    T_btw_cameras[:3, 3] = s.getNode('tvec').mat().reshape((3))
    print(filename)
    print(T_btw_cameras)
    src_1, tg_1 = pc.transform_points(T_btw_cameras)
##    pc.fit_plane(src_1, tg_1)

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

#    print('between two matrix')
#    print(T_btw_cameras.dot(np.linalg.inv(T_btw_cameras_)))

    num_ply_file = ['cloud0_fiducial.ply', 'cloud1_fiducial.ply']
    ply = []
    for i in num_ply_file:
        ply.append(o3d.io.read_point_cloud(i))
        
    ply[0].paint_uniform_color([0, 1.0, 0])
    ply[1].paint_uniform_color([1.0, 0, 0])

    o3d.visualization.draw_geometries([ply[0], ply[1]])
    ply[1].transform(T_btw_cameras)
    o3d.visualization.draw_geometries([ply[0], ply[1]])

    pts_ply_0 = np.array(ply[0].points)
    pts_ply_1 = np.array(ply[1].points)

    #error = np.linalg.norm(pts_ply_0 - pts_ply_1)
    #print(error)
    err = np.linalg.norm(pts_ply_0 - pts_ply_1, axis=1)
    print(np.mean(err))

    X0 = np.eye(4)
    tic = time.perf_counter()

    def lossfn(x):
        cost = []
        X = np.eye(4)
        R = matrix_from_quaternion(x[0:4])
        tvec = x[4:7]
        X[:3, :3] = R
        X[:3, 3] = tvec
        cost = []

        #source_temp = copy.deepcopy(ply[0])
        #target_temp = copy.deepcopy(ply[1])
        source_temp = ply[0]
        target_temp = ply[1]

        src_plane_model, src_inliers = source_temp.segment_plane(distance_threshold=0.001, ransac_n=3, num_iterations=1000)
        [src_a, src_b, src_c, src_d] = src_plane_model

        tg_plane_model, tg_inliers = target_temp.segment_plane(distance_threshold=0.001, ransac_n=3, num_iterations=1000)
        [tg_a, tg_b, tg_c, tg_d] = tg_plane_model

        src_n = np.array([src_a, src_b, src_c])
        tg_n = np.array([tg_a, tg_b, tg_c])
        dot_res = src_n.dot(tg_n)
        crs_res = np.cross(src_n, tg_n)
        #norm_vec = np.linalg.norm(src_n) * np.linalg.norm(tg_c)
        ang_btw_planes = np.arccos(dot_res)

        for pt0, pt1 in zip(pts_ply_0, pts_ply_1):
            dummy = np.ones(1)
            pt1 = np.append(pt1, dummy, axis=0)
            pt_1 = X.dot(pt1)
            cost_ = np.linalg.norm(pt0 - pt_1[:3]) #+ 1000 * ang_btw_planes
            cost.append(cost_)
        return cost

    Rx0 = X0[:3, :3]
    tx0 = X0[:3, 3]
    x0 = np.concatenate((quaternion_from_matrix(Rx0), tx0.reshape(3)), axis=0)
    ''' lm method
    - jac: only '2-point'
    - ftol: Tolerance for termination by the change of the cost function. Default is 1e-8.
      xtol: Tolerance for termination by the change of the independent variables. Default is 1e-8. 
      gtol: Tolerance for termination by the norm of the gradient. Default is 1e-8. 
    If None, the termination by this condition is disabled.
    - loss: only 'linear' option, so 'f_scale' also cannot control.
    - max_nfec
    '''
    cost_fn_tol = 0.00001
    res = scipy.optimize.least_squares(fun=lossfn, x0=x0, method='lm', ftol=cost_fn_tol)
    T = np.eye(4)
    T[:3, :3] = matrix_from_quaternion(res.x[0:4])
    T[:3, 3] = res.x[4:7]
    print(T)
    toc = time.perf_counter()
    elapsed_time = toc - tic
    print('Time(Tabb_zc1):', round(elapsed_time, 2))
    ply[1].transform(T)
    o3d.visualization.draw_geometries([ply[0], ply[1]])

    pts_ply_0 = np.array(ply[0].points)
    pts_ply_1 = np.array(ply[1].points)

    err = np.linalg.norm(pts_ply_0 - pts_ply_1, axis=1)
    print(np.mean(err))
    #error = np.linalg.norm(pts_ply_0 - pts_ply_1)
    #print(error)

#    num_ply_file = ['cloud0_corners.ply', 'cloud1_corners.ply']
#    ply = []
#    for i in num_ply_file:
#        ply.append(o3d.io.read_point_cloud(i))
#        
#    src_corners = np.asarray(ply[0].points)
#    tg_corners = np.asarray(ply[1].points)
#    src_center = np.mean(src_corners, axis=0)
#    tg_center = np.mean(tg_corners, axis=0)
#    tvec_btw_centers = tg_center - src_center
#
#    T_two_planes_trans = np.eye(4)
#    T_two_planes_trans[:3, 3] = tvec_btw_centers
