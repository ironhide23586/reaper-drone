import numpy as np


def quaternion_to_euler_ned(q):
    q_w, q_x, q_y, q_z = q
    yaw = np.arctan2(2 * (q_w * q_z + q_x * q_y), 1 - 2 * (q_y * q_y + q_z * q_z))
    sin_pitch = 2 * (q_w * q_y - q_z * q_x)
    if abs(sin_pitch) >= 1:
        pitch = np.sign(np.pi / 2, sin_pitch)  # Use 90 degrees if out of range
    else:
        pitch = np.arcsin(sin_pitch)
    roll = np.arctan2(2 * (q_w * q_x + q_y * q_z), 1 - 2 * (q_x * q_x + q_y * q_y))
    return np.rad2deg(yaw), np.rad2deg(pitch), np.rad2deg(roll)

def vector_to_quaternion(v):
    v_norm = v / np.linalg.norm(v)
    r = np.array([1, 0, 0])
    axis = np.cross(r, v_norm)
    theta = np.arccos(np.dot(r, v_norm))
    if np.linalg.norm(axis) != 0:
        axis = axis / np.linalg.norm(axis)
    q_w = np.cos(theta / 2)
    q_x = np.sin(theta / 2) * axis[0]
    q_y = np.sin(theta / 2) * axis[1]
    q_z = np.sin(theta / 2) * axis[2]
    q = np.array([q_w, q_x, q_y, q_z])
    q /= np.linalg.norm(q)
    return q

def quaternion_from_angular_velocity(q, omega):
    q_w, q_x, q_y, q_z = q
    omega_x, omega_y, omega_z = omega
    q_dot = 0.5 * np.array([
        -q_x * omega_x - q_y * omega_y - q_z * omega_z,
        q_w * omega_x + q_y * omega_z - q_z * omega_y,
        q_w * omega_y + q_z * omega_x - q_x * omega_z,
        q_w * omega_z + q_x * omega_y - q_y * omega_x
    ])
    q_dot /= np.linalg.norm(q_dot)
    return q_dot


def quaternion_multiply(q1, q2, normalize=False):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    q = np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    ])
    if normalize:
        q /= np.linalg.norm(q)
    return q


def quaternion_jacobian(q, omega):
    q_w, q_x, q_y, q_z = q
    omega_x, omega_y, omega_z = omega
    return 0.5 * np.array([
        [-q_x, -q_y, -q_z],
        [q_w, -q_z, q_y],
        [q_z, q_w, -q_x],
        [-q_y, q_x, q_w]
    ])


def euler_to_quaternion(yaw_deg, pitch_deg, roll_deg):
    yaw = np.deg2rad(yaw_deg)
    pitch = np.deg2rad(pitch_deg)
    roll = np.deg2rad(roll_deg)
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)
    q = np.zeros(4)
    q[0] = cy * cp * cr + sy * sp * sr
    q[1] = cy * cp * sr - sy * sp * cr
    q[2] = sy * cp * sr + cy * sp * cr
    q[3] = sy * cp * cr - cy * sp * sr
    return q


def transform_to_quaternion_frame(v_body_frame, q_pose):
    vx, vy, vz = v_body_frame
    qw, qx, qy, qz = q_pose
    vx_frame = (qw ** 2 + qx ** 2 - qy ** 2 - qz ** 2) * vx + 2 * (qx * qy - qw * qz) * vy + 2 * (qx * qz + qw * qy) * vz
    vy_frame = 2 * (qx * qy + qw * qz) * vx + (qw ** 2 - qx ** 2 + qy ** 2 - qz ** 2) * vy + 2 * (qy * qz - qw * qx) * vz
    vz_frame = 2 * (qx * qz - qw * qy) * vx + 2 * (qy * qz + qw * qx) * vy + (qw ** 2 - qx ** 2 - qy ** 2 + qz ** 2) * vz
    v_frame = np.array([vx_frame, vy_frame, vz_frame])
    return v_frame


def vector_jacobian_wrt_quaternion(v_body_frame, q_pose):
    vx, vy, vz = v_body_frame
    qw, qx, qy, qz = q_pose

    dvx_dqw = 2 * (qw * vx - qz * vy + qy * vz)
    dvx_dqx = 2 * (qx * vx + qy * vy + qz * vz)
    dvx_dqy = 2 * (-qy * vx + qx * vy - qw * vz)
    dvx_dqz = 2 * (-qz * vx + qw * vy + qx * vz)

    dvy_dqw = 2 * (qz * vx + qw * vy - qx * vz)
    dvy_dqx = 2 * (qy * vx - qx * vy - qw * vz)
    dvy_dqy = 2 * (qw * vx + qw * vy + qz * vz)
    dvy_dqz = 2 * (qw * vx + qz * vy - qy * vz)

    dvz_dqw = 2 * (-qy * vx + qx * vy + qw * vz)
    dvz_dqx = 2 * (qz * vx - qw * vy - qy * vz)
    dvz_dqy = 2 * (-qw * vx + qz * vy + qx * vz)
    dvz_dqz = 2 * (qx * vx + qy * vy + qw * vz)

    derivative = np.array([[dvx_dqw, dvx_dqx, dvx_dqy, dvx_dqz],
                           [dvy_dqw, dvy_dqx, dvy_dqy, dvy_dqz],
                           [dvz_dqw, dvz_dqx, dvz_dqy, dvz_dqz]])
    return derivative



class ExtendedKalmanFilter:

    def __init__(self):

        # State vector: [position, velocity, quaternion]
        self.state = [np.zeros(3), np.zeros(3), np.zeros(4)]
        self.state[-1][0] = 1.

        # Covariance matrix
        self.P = np.eye(16) * 0.1

        # Process noise covariance
        self.Q = np.eye(16) * 0.01

        # Measurement noise covariance
        self.R = np.eye(6) * 0.1

        # Identity matrix
        self.I = np.eye(16)
        self.pitch_offset_quaternion = np.array([0.7071067811865476, 0, 0.7071067811865476, 0])

    def acc_to_pitch_roll(self, acc_s):
        pitch = np.rad2deg(np.arctan2(acc_s[0], np.sqrt(acc_s[1] ** 2 + acc_s[2] ** 2)))
        roll = np.rad2deg(np.arctan2(acc_s[1], acc_s[2]))
        return pitch, roll

    def process_measurement(self, sensor_readings, quat, delta_t):
        acc_s, omega_s = sensor_readings
        acc_pitch, acc_roll = self.acc_to_pitch_roll(acc_s)
        quat_ypr = quaternion_to_euler_ned(quat)
        acc_ypr = np.array([quat_ypr[0], acc_pitch, acc_roll])
        acc_s_quaternion = euler_to_quaternion(*acc_ypr)
        omega_w_world_x, omega_w_world_y, omega_w_world_z = transform_to_quaternion_frame(omega_s, quat)
        omega_s_quaternion = quaternion_multiply(quat, np.array([1, omega_w_world_x * delta_t * .5,
                                                                 omega_w_world_y * delta_t * .5,
                                                                 omega_w_world_z * delta_t * .5]), normalize=True)
        s_quaternion = .99 * omega_s_quaternion + .01 * acc_s_quaternion
        s_quaternion /= np.linalg.norm(s_quaternion)
        acc_s_world_frame = transform_to_quaternion_frame(acc_s, s_quaternion)
        omega_s_world_frame = np.array([omega_w_world_x, omega_w_world_y, omega_w_world_z])
        return acc_s_world_frame, omega_s_world_frame, s_quaternion

    def state_transition_function(self, delta_t, sensor_readings):
        pos, vel, quat = self.state
        acc_s_world_frame, omega_s_world_frame, s_quaternion = self.process_measurement(sensor_readings, quat, delta_t)
        acc_s_world = (acc_s_world_frame - [0, 0, 1]) * 9.81  #######
        vel_s_world = vel + acc_s_world * delta_t
        pos_s_world = pos + vel_s_world * delta_t + .5 * acc_s_world * delta_t ** 2
        new_state = [pos_s_world, vel_s_world, s_quaternion]
        return new_state

    def state_transition_function_jacobian(self, delta_t, sensor_readings):
        # implement the differentials of every state transition function with respect to every state variable
        # [position, velocity, quaternion, acceleration, angular velocity]
        acc_s, omega_s = sensor_readings
        quat, vel, pos = self.state[6:10], self.state[3:6], self.state[0:3]

        d_pos_s_world_d_pos = 1
        d_pos_s_world_d_vel = delta_t



    def predict(self, delta_t, sensor_readings):
        new_state = self.state_transition_function(delta_t, sensor_readings)



        self.state_transition_matrix = np.eye(16)
        self.state_transition_matrix[0:3, 3:6] = np.eye(3) * delta_t
        self.state_transition_matrix[3:6, -3:] = np.eye(3) * delta_t
        self.state_transition_matrix[6:10, 10:13] = np.array([[-0.5 * delta_t * wx,    -0.5 * delta_t * wy,   -0.5 * delta_t * wz],
                                                              [0.5 * delta_t * wx,     0.5 * delta_t * wz,   -0.5 * delta_t * wy],
                                                              [-0.5 * delta_t * wz,     0.5 * delta_t * wx,    0.5 * delta_t * wy],
                                                              [0.5 * delta_t * wy,     -0.5 * delta_t * wx,   0.5 * delta_t * wz]])
        next_state = np.dot(self.state_transition_matrix, self.state)

        next_pos = self.state[0:3] + self.state[3:6] * delta_t + 0.5 * accel * delta_t ** 2
        next_vel = self.state[3:6] + accel * delta_t
        next_quaternion = self.state[6:10] + self.quaternion_from_angular_velocity(self.state[6:10], sensor_gyro * delta_t)


        state_transition_matrix_jacobian = np.eye(16)
        state_transition_matrix_jacobian[0:3, 3:6] = np.eye(3) * delta_t
        state_transition_matrix_jacobian[3:6, -3:] = np.eye(3) * delta_t
        state_transition_matrix_jacobian[6:10, 10:13] = np.array([[-0.5 * delta_t * wx, -0.5 * delta_t * wy, -0.5 * delta_t * wz],
                                                                  [0.5 * delta_t * qw,  0.5 * delta_t * qz,  -0.5 * delta_t * qy],
                                                                  [-0.5 * delta_t * qz,  0.5 * delta_t * qw,  0.5 * delta_t * qx],
                                                                  [0.5 * delta_t * qy,  -0.5 * delta_t * qx,  0.5 * delta_t * qw]])
        state_transition_matrix_jacobian[7:10, 6:10] = np.array([[0.5 * delta_t * wx, 1,    0.5 * delta_t * wz, -0.5 * delta_t * wy],
                                                                 [0.5 * delta_t * wy, -0.5 * delta_t * wz, 1, 0.5 * delta_t * wx],
                                                                 [0.5 * delta_t * wz, 0.5 * delta_t * wy, -0.5 * delta_t * wx, 1]])

        new_P = state_transition_matrix_jacobian @ self.P @ state_transition_matrix_jacobian.T + self.Q

        self.state = next_state

        q = self.state[6:10]
        vel = self.state[3:6]

        # Predict position
        self.state[0:3] += vel * self.dt + 0.5 * accel * self.dt ** 2

        # Predict velocity
        self.state[3:6] += accel * self.dt

        # Predict quaternion
        omega = gyro * self.dt
        delta_q = self.quaternion_from_angular_velocity(q, omega)
        self.state[6:10] = self.quaternion_multiply(q, delta_q)

        # Normalize quaternion
        self.state[6:10] /= np.linalg.norm(self.state[6:10])

        # State transition matrix
        F = np.eye(16)
        F[0:3, 3:6] = np.eye(3) * self.dt
        F[3:6, 6:10] = self.quaternion_jacobian(q, accel * self.dt)

        # Predict covariance
        self.P = F @ self.P @ F.T + self.Q

    def update(self, accel, gyro):
        # Measurement vector: [accel, gyro]
        z = np.hstack((accel, gyro))

        # Measurement prediction
        h = np.hstack((self.state[3:6], self.state[6:10]))

        # Measurement matrix
        H = np.zeros((6, 10))
        H[0:3, 3:6] = np.eye(3)
        H[3:6, 6:10] = np.eye(3, 4)

        # Kalman gain
        K = self.P @ H.T @ np.linalg.inv(H @ self.P @ H.T + self.R)

        # Update state
        self.state += K @ (z - h)

        # Update covariance
        self.P = (self.I - K @ H) @ self.P
