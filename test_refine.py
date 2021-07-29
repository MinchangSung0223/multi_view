#!/usr/bin/env python3
import numpy as np
import kinematics as kin


def absoluteOrientation(A, B):
    centroid_a = np.mean(A, axis=0)
    centroid_b = np.mean(B, axis=0)
    A_wrt_centroid = A - centroid_a
    B_wrt_centroid = B - centroid_b
    M = 0
    for a_, b_ in zip(A_wrt_centroid, B_wrt_centroid):
        M_ = np.dot(np.expand_dims(a_, axis=1), np.expand_dims(b_, axis=0))
        M = M + M_
    Sxx, Sxy, Sxz = M[0, 0], M[0, 1], M[0, 2]
    Syx, Syy, Syz = M[1, 0], M[1, 1], M[1, 2]
    Szx, Szy, Szz = M[2, 0], M[2, 1], M[2, 2]
    N = np.array([[Sxx + Syy + Szz, Syz - Szy, Szx - Sxz, Sxy - Syx],
                  [Syz - Szy, Sxx - Syy - Szz, Sxy + Syx, Szx + Sxz],
                  [Szx - Sxz, Sxy + Syx, -Sxx + Syy - Szz, Syz + Szy],
                  [Sxy - Syx, Szx + Sxz, Syz + Szy, -Sxx - Syy + Szz]])

    eig_val, eig_vec = np.linalg.eig(N)
    q = eig_vec[:, np.argmax(eig_val)]
    print('---------------')
    print('Eigenvalues', eig_val)
    print('Max. Eigenvalue', max(eig_val))
    print('Eigenvectors', eig_vec)
    print('Eigenvector corresponding to max. eigenvalue', eig_vec[:, np.argmax(eig_val)])
    print('Norm ', np.linalg.norm(q))
    print('---------------')
    qw, qx, qy, qz = q[0], q[1], q[2], q[3]
#    rot1 = np.array([[1 - 2 * qy ** 2 - 2 * qz ** 2, 2 * qx * qy - 2 * qz * qw, 2 * qx * qz + 2 * qy * qw],
#                     [2 * qx * qy + 2 * qz * qw, 1 - 2 * qx ** 2 - 2 * qz ** 2, 2 * qy * qz - 2 * qx * qw],
#                     [2 * qx * qz - 2 * qy * qw, 2 * qy * qz + 2 * qx * qw, 1 - 2 * qx ** 2 - 2 * qy ** 2]])

    v = q[1:4]
    v_out = np.outer(v, v)
    Z = np.array([[qw, -qz, qy],
                  [qz, qw, -qx],
                  [-qy, qx, qw]])
    rot2 = v_out + np.dot(Z, Z)

    rot = rot2
    t = centroid_b - np.dot(rot, centroid_a)
    return rot, t


if __name__ == '__main__':
    A = kin.generatePointSet(4, 3, 0, 10)
    T_ab = kin.homogeneousMatrix(0, 50, 30, 0, 90, 0, True)
    B = kin.transform(A, T_ab)
    print('A', A)
    print('B', B)
    R_est, t_est = absoluteOrientation(A, B)
    print('Ground truth')
    R_original = T_ab[:3, :3]
    t_original = T_ab[:3, 3]
    print('Orientation', R_original)
    print('Position', t_original)
    e_ori = R_original - R_est
    e_pos = t_original - t_est
    print('---------------')
    print('Estimation')
    print('Orientation', R_est)
    print('Position', t_est)
    print('---------------')
    print('Error')
    print('e_ori', np.linalg.norm(e_ori))
    print('e_pos', np.linalg.norm(e_pos))
