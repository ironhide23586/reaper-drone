



def get_init_state():
    init_state = {
                    'raw_system_state': {
                        'battery': 100.,
                        'attitude': {
                            'pose': {
                                'euler': {
                                    'yaw_degrees': 0.,
                                    'pitch_degrees': 0.,
                                    'roll_degrees': 0.
                                },
                                'quaternion': [0., 0., 0., 0.]
                            },
                            'translation': {
                                'altitude_meters': 0.,
                                'gps_location_degrees': [0., 0.],
                                'velocity_meters_s': [0., 0., 0.],
                                'acceleration_meters_s2': [0., 0., 0.]
                            },
                            'rotation': {
                                'velocity_degrees_s': [0., 0., 0.],
                                'acceleration_degrees_s2': [0., 0., 0.]
                            }
                        },
                        'sensor': {
                            'altitude_meters': 0.,
                            'barometric_pressure': 0.,
                            'gps': [],
                            'magnetometer': [],
                            'accelerometer': [],
                            'gyroscope': [],
                            'lidar': {}
                        },
                        'actuator': {
                            'propeller_power_fraction': [0., 0., 0., 0.]
                            'temperature_celsius': 0.
                        },
                    },
                    'processed_system_state': {
                        'flight_time_seconds': 0.,
                        'attitude': {
                            'pose': {
                                'euler': {
                                    'yaw_degrees': 0.,
                                    'pitch_degrees': 0.,
                                    'roll_degrees': 0.
                                },
                                'quaternion': [0., 0., 0., 0.]
                            },
                            'translation': {
                                'altitude_meters': 0.,
                                'gps_location_degrees': [0., 0.],
                                'velocity_meters_s': [0., 0., 0.],
                                'acceleration_meters_s2': [0., 0., 0.],
                                'world_coords_enu_meters': [0., 0., 0.]
                            },
                            'rotation': {
                                'velocity_degrees_s': [0., 0., 0.],
                                'acceleration_degrees_s2': [0., 0., 0.]
                            }
                        }
                    },
                    'static_config': {
                        'alarms': {
                            'battery': {
                                'min_return_to_home_percentage': 20,
                                'min_emergency_landing_percentage': 5
                            },
                            'attitude': {
                                'pose': {
                                    'euler': {
                                        'max_pitch_degrees': 30,
                                        'max_roll_degrees': 20
                                    }
                                },
                                'translation': {
                                    'max_altitude_meters': 100.,
                                    'max_velocity_meters_s': 30.,
                                    'max_acceleration_meters_s2': 5.
                                },
                                'rotation': {
                                    'max_velocity_degrees_s': 40,
                                    'max_acceleration_degrees_s2': 10.
                                }
                            },
                            'actuator': {
                                'max_propeller_power_fraction': .6,
                                'max_temperature_celsius': 40,
                            },
                            'derived': {
                                'max_takeoff_point_geofence_radius_meters': 100,
                                'max_flight_time_seconds': 600
                            }
                        }
                    }
                }
    return init_state


class StateConfig:

    def __init__(self, cfg=None):
        if cfg is None:
            self.cfg = get_init_state()

    def copy(self):
        return self
