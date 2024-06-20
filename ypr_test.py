

import numpy as np


def torques2forces(j, v_l, v_w):
    e = np.array([0, 0, 1])
    n = np.cross(j[:3], e)
    fx = -n[0] / v_w
    fy = -n[1] / v_l
    fs = np.square(fx) + np.square(fy)
    fa = j[-1]**2
    fz = np.sign(fa - fs) * np.sqrt(np.abs(fa - fs))
    return np.array([fx, fy, fz])


def cross_prod(p, q):
    a, b, c = p
    x, y, z = q
    r = [((b * z) - (c * y)), ((c * x) - (a * z)), ((a * y) - (b * x))]
    return np.array(r)


def forces2torques(f, v_l, v_w):
    fx, fy, fz = f
    n = np.zeros(3)
    n[0] = -(fx * v_w)
    n[1] = -(fy * v_l)
    e = np.array([0, 0, -1])
    jx, jy, _ = np.cross(n, e)
    # jx_, jy_, _ = cross_prod(n, e)
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
    tm = np.linalg.pinv(t_mat)
    t_pred = np.dot(tm, torques_and_netthrust) / 5.
    return t_pred


if __name__ == '__main__':

    # 1024 -> 25g, 21g | 23g ; 51.0204081632653 g ;  +28.0204081632653 g
    # 1048 -> 122g, 113g | 117.5g ; 102.0408163265306 g ; -15.459183673469397 g
    # 1075 -> 183g, 190g | 186.5 ; 153.0612244897959 g ; -33.438775510204096 g
    # 1100 -> 264g, 256g | 260 ; 204.0816326530612 g ; -55.918367346938794 g


    thrusts = np.array([0.27091882, 0.20020814,  60.51046997,  60.4397593])
    thrust_torque_coeffs = np.array([.3, .3, .3, .3])

    radius_front = .12
    radius_rear = .12
    angle_front_deg = 90
    angle_rear_deg = 90

    # radius_front = .2556
    # radius_rear = .2556
    # angle_front_deg = 112.5520925919324
    # angle_rear_deg = 112.5520925919324

    thrust_torque_coeffs[[0, 3]] *= radius_front
    thrust_torque_coeffs[[1, 2]] *= radius_rear

    length = (radius_front * np.cos(np.deg2rad(angle_front_deg / 2))
              + radius_rear * np.cos(np.deg2rad(angle_rear_deg / 2)))
    width_front = 2 * radius_front * np.sin(np.deg2rad(angle_front_deg / 2))
    width_rear = 2 * radius_rear * np.sin(np.deg2rad(angle_rear_deg / 2))
    width = (width_rear + width_front) / 2.

    t_mat = get_tmat(radius_front, radius_rear, thrust_torque_coeffs)
    f_xyz, torques_xyz = thrust2forces_torques(thrusts, t_mat, length, width)
    # t_pred = forces_yaw_torque_to_thrusts(f_xyz, torques_xyz[-1], t_mat, length, width)

    f = [0, 0, 2.5]
    tz = .05
    t_pred = forces_yaw_torque_to_thrusts(f, tz, t_mat, length, width)

    print(t_pred, f_xyz)
    print(np.max(np.abs(t_pred - thrusts)))


    f_xyz = [0, 0, 1.5 * 9.8]
    tq = 0
    t_pred = forces_yaw_torque_to_thrusts(f_xyz, tq, t_mat, length, width)
    t_am = t_pred * 175
    print(t_pred, 'N,', t_am, 'rad/s')
    k = 0