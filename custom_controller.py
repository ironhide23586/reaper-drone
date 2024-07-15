import carb

# Imports from the Pegasus library
from pegasus.simulator.logic.state import State
from pegasus.simulator.logic.backends import Backend

# Auxiliary scipy and numpy modules
import numpy as np
from scipy.spatial.transform import Rotation



def get_tmat(r, R, k):
    return np.array([[-r, -R, R, r],
                     [r, -R, -R, r],
                     [-k[0], k[1], -k[2], k[3]],
                     [1, 1, 1, 1]])


def forces2torques(f, v_l, v_w):
    fx, fy, fz = f
    n = np.zeros(3)
    n[0] = -(fx * v_w)
    n[1] = -(fy * v_l)
    e = np.array([0, 0, -1])
    jx, jy, _ = np.cross(n, e)
    return np.array([jx, jy])


def forces_yaw_torque_to_thrusts(f_xyz, yaw_torque, t_mat_inv, v_l, v_w):
    torques_xy = forces2torques(f_xyz, v_l, v_w)
    torques_and_netthrust = list(torques_xy) + [yaw_torque, np.linalg.norm(f_xyz)]
    t_pred = np.dot(t_mat_inv, torques_and_netthrust)
    return t_pred


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


class CustomController(Backend):

    def __init__(self):
        self.m = 0.8  # kg
        self.g = 9.81  # m/s^2
        # front-right, rear-left, front-left, rear-right
        self.input_ref = np.array([0., 0., 0., 0.])
        # self.input_ref = np.array([600., 600., 600., 600.])
        self.sensor_data = {}

        # The current state of the vehicle expressed in the inertial frame (in ENU)
        self.p = np.zeros((3,))                   # The vehicle position
        self.q = np.array([1, 0, 0, 0])           # The vehicle attitude
        self.alpha = np.zeros((3,))                # The angular acceleration of the vehicle
        self.w = np.zeros((3,))                   # The angular velocity of the vehicle
        self.w_prev = np.zeros((3,))              # The previous angular velocity of the vehicle
        self.v = np.zeros((3,))                   # The linear velocity of the vehicle in the inertial frame
        self.a = np.zeros((3,))                   # The linear acceleration of the vehicle in the inertial frame

        self.yaw = 0.
        self.pitch = 0.
        self.roll = 0.
        self.actuation_d = .2

        self.yaw_prev = 0.
        self.pitch_prev = 0.
        self.roll_prev = 0.

    # def update_sensor(self, sensor_type, data):
    #     self.sensor_data[sensor_type] = data
    #     print(data)
    #     exit()

    def update_state(self, state: State):
        """
        Method that updates the current state of the vehicle. This is a callback that is called at every physics step

        Args:
            state (State): The current state of the vehicle.
        """
        self.yaw_prev = self.yaw
        self.pitch_prev = self.pitch
        self.roll_prev = self.roll

        self.p = state.get_position_ned()
        qx, qy, qz, qw = state.attitude #.get_attitude_ned_frd()
        self.q = np.array([qw, qx, qy, qz])
        self.yaw, self.pitch, self.roll = quaternion_to_euler_ned(self.q)
        self.pitch *= -1
        self.yaw *= -1
        self.q = euler_to_quaternion(self.yaw, self.pitch, self.roll)

        self.w_prev = self.w
        self.w = state.get_angular_velocity_frd()
        self.v = state.get_linear_velocity_ned()
        self.a = state.get_linear_acceleration_ned()
        # print('Yaw:', self.yaw, '| Pitch:', self.pitch, '| Roll:', self.roll)

        # print(self.p, self.q, np.linalg.norm(self.q), self.w, self.v, self.a)

    def input_reference(self):
        # print(self.input_ref, -self.p[-1])
        return self.input_ref

    def actuate_pp(self, v):
        # v = np.clip(v, 0., 65535.)
        return v

    def actuate_yaw(self, m):
        v = m * np.array([self.actuation_d, self.actuation_d, -self.actuation_d, -self.actuation_d])
        return self.actuate_pp(v)

    def actuate_pitch(self, m):
        v = m * np.array([self.actuation_d, -self.actuation_d, self.actuation_d, -self.actuation_d])
        return self.actuate_pp(v)

    def actuate_roll(self, m):
        v = m * np.array([-self.actuation_d, self.actuation_d, self.actuation_d, -self.actuation_d])
        return self.actuate_pp(v)

    def actuate_thrust(self, m):
        v = m * self.actuation_d * np.array([1, 1, 1, 1])
        return self.actuate_pp(v)

    def update(self, dt):
        k = 100
        k_pitch = 3

        self.alpha = (self.w - self.w_prev) / dt

        resultant_delta_derivative = (
                    self.actuate_roll(-k * np.sign(self.roll) * abs(np.sin(self.alpha[0])))
                    + self.actuate_pitch(-k_pitch * k * np.sign(self.pitch) * abs(np.sin(self.alpha[1])))
                    + np.sign(self.p[-1] + .5) * 1000 * self.actuate_thrust(abs(self.a[-1]))
                    + np.sign(self.v[0]) * 200 * self.actuate_pitch(abs(self.a[0]))
                    + np.sign(-self.v[1]) * 200 * self.actuate_roll(abs(self.a[1]))
                    + np.sign(-self.w[-1]) * 200 * self.actuate_yaw(abs(self.alpha[-1])))
        resultant_delta_integral = (self.actuate_roll(-k * np.sin(np.deg2rad(self.roll))**2)
                                    + self.actuate_pitch(-k_pitch * k * np.sign(self.pitch)
                                                         * np.sin(np.deg2rad(self.pitch))**2)
                                    + self.actuate_thrust((self.p[-1] + .5) * 50)
                                    + self.actuate_pitch(self.p[0]) * 50
                                    + self.actuate_roll(-self.p[1]) * 50
                                    + self.actuate_yaw(np.sin(-np.deg2rad(self.yaw))) * 100)
        resultant_delta_proportional = (
                    self.actuate_roll(-10 * k * np.sign(self.roll) * abs(np.sin(self.w[0])))
                    + self.actuate_pitch(-200 * k_pitch * k * np.sign(self.pitch) * np.sin(self.w[1])**2)
                    + self.actuate_thrust(np.sign(self.p[-1] + .5) * abs(self.v[-1]) * 10)
                    + self.actuate_pitch(self.v[0] * 10)
                    + self.actuate_roll(-self.v[1] * 10)
                    + self.actuate_yaw(np.sin(-self.w[-1])) * 10)

        resultant_delta_num = np.vstack([resultant_delta_proportional, resultant_delta_integral, resultant_delta_derivative])
        pid_const = np.array([1.021, .156, .0018])
        resultant_delta = .97 * (np.dot(pid_const, resultant_delta_num) + self.actuate_pitch(-85) + self.actuate_roll(9))
        # front-right, rear-left, front-left, rear-right

        self.input_ref = np.clip(self.actuate_thrust(3430) + resultant_delta,
                                 0, 65535)

        # print(resultant_delta, self.input_ref, self.actuate_thrust((self.p[-1] + .5) * 10))
        print(resultant_delta_proportional, resultant_delta_integral, resultant_delta_derivative, self.actuate_thrust(np.sign(self.p[-1] + .5) * abs(self.v[-1]) * 10),
              self.actuate_thrust((self.p[-1] + .5) * 50))
        # self.input_ref = self.actuate_thrust(6800) + self.actuate_yaw(-200)



        # self.input_ref += self.actuate_yaw(-k * np.sin(np.deg2rad(self.yaw)))

        # h = 1
        # curr_h = -self.p[-1]
        # self.input_ref += self.actuate_thrust(k * (h - curr_h))

        # orientation = sensor_data['orientation']
        # angular_velocity = sensor_data['angular_velocity']
        # if self.input_ref[0] < 655:


        #     self.input_ref += .01

