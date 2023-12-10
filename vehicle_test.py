
from vehicle.control import StateMachine, ActuatorInterface


if __name__ == '__main__':

    controller = ActuatorInterface()
    drone_state = StateMachine(controller)


