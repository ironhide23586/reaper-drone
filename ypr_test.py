

import numpy as np


def get_force_normal_plane_angle(r, R, t, v_l, v_w, k):
    jx = (r * (t[3] - t[0]) + R * (t[2] - t[1]))
    jy = (r * (t[3] + t[0]) - R * (t[2] + t[1]))
    jz = ((k[1] * t[1] + k[3] * t[3]) - (k[0] * t[0] + k[2] * t[2]))
    j = np.array([jx, jy, jz])
    e = np.array([0, 0, 1])
    n = np.cross(j, e)
    angle = 0
    tr = np.sum(t)
    fx = -n[0] / v_w
    fy = -n[1] / v_l
    fz = np.sqrt(tr**2 - np.square(fx) - np.square(fy))
    if np.max(np.abs(n)) > 0:
        angle = np.arctan2(n[0], n[1])
        angle = -np.rad2deg(angle)
    return angle, [fx, fy, fz], jz


def torques2forces(j, v_l, v_w):
    e = np.array([0, 0, 1])
    n = np.cross(j[:3], e)
    fx = -n[0] / v_w
    fy = -n[1] / v_l
    fs = np.square(fx) + np.square(fy)
    fa = j[-1]**2
    fz = np.sign(fa - fs) * np.sqrt(np.abs(fa - fs))
    return np.array([fx, fy, fz])


def forces2torques(f, v_l, v_w):
    fx, fy, fz = f
    n = np.zeros(3)
    n[0] = -(fx * v_w)
    n[1] = -(fy * v_l)
    e = np.array([0, 0, -1])
    jx, jy, _ = np.cross(n, e)
    return np.array([jx, jy])


def get_tmat(r, R, k):
    return np.array([[-r, -R, R, r],
                     [r, -R, -R, r],
                     [-k[0], k[1], -k[2], k[3]],
                     [1, 1, 1, 1]])


def thrust2torques(t, A):
    B = np.expand_dims(t, 0).T
    C = np.dot(A, B)
    return C


def thrust2forces_torques(t, A, v_l, v_w):
    torques_netthrust = thrust2torques(thrusts, t_mat)
    f_xyz = torques2forces(torques_netthrust.T[0], v_l, v_w)
    assert abs(torques_netthrust[-1] - np.linalg.norm(f_xyz)) < .005
    return f_xyz, torques_netthrust[:3].T[0]


def forces_yaw_torque_to_thrusts(f_xyz, yaw_torque, t_mat, v_l, v_w):
    torques_xy = forces2torques(f_xyz, v_l, v_w)
    torques_and_netthrust = list(torques_xy) + [yaw_torque, np.linalg.norm(f_xyz)]
    t_pred = np.dot(np.linalg.pinv(t_mat), torques_and_netthrust)
    return t_pred


if __name__ == '__main__':
    thrusts = np.array([.64, .64, .1, .9])
    thrust_torque_coeffs = np.array([.3, .3, .3, .3])
    radius_front = 1
    radius_rear = 1
    angle_front_deg = 90

    thrust_torque_coeffs[[0, 3]] *= radius_front
    thrust_torque_coeffs[[1, 2]] *= radius_rear
    angle_side_deg = 180 - angle_front_deg
    length = (radius_front * np.cos(np.deg2rad(angle_front_deg / 2))
              + radius_rear * np.cos(np.deg2rad(angle_front_deg / 2)))
    width_front = 2 * radius_front * np.sin(np.deg2rad(angle_front_deg / 2))
    width_rear = 2 * radius_rear * np.sin(np.deg2rad(angle_front_deg / 2))
    width = (width_rear + width_front) / 2.

    t_mat = get_tmat(radius_front, radius_rear, thrust_torque_coeffs)
    f_xyz, torques_xyz = thrust2forces_torques(thrusts, t_mat, length, width)
    t_pred = forces_yaw_torque_to_thrusts(f_xyz, torques_xyz[-1], t_mat, length, width)

    print(t_pred, f_xyz)
    print(np.max(np.abs(t_pred - thrusts)))

    k = 0