import time

import vehicle.utils


class ActuatorInterface:

    def __init__(self):
        pass

    def actuate(self, curr_state, targ_state):
        curr_state = self.get_current_state()
        return curr_state

    def get_current_state(self):
        return vehicle.utils.StateConfig()


class StateMachine:

    def __init__(self, controller, init_state=None, loop_delay_ms=10):
        self.state_update_received = False
        self.new_state = None
        self.controller = controller
        if init_state is not None:
            self.state = self.get_state(init_state)
        else:
            self.state = self.controller.get_current_state()
        self.curr_state = self.state.copy()
        self.loop_delay_ms = loop_delay_ms

    def get_state(self, input_state):
        if type(input_state) == dict:
            state = vehicle.utils.StateConfig(input_state)
        else:
            state = input_state
        return state

    def update_state(self, targ_state):
        self.new_state = targ_state
        self.state_update_received = True

    def live(self):
        """
        Active Control Loop for vehicle
        :return: None
        """
        while True:
            if not self.state_update_received:
                self.new_state = self.state
            else:
                self.state_update_received = False
                self.state = self.new_state
            self.curr_state = self.step(self.new_state)
            time.sleep(self.loop_delay_ms)

    def step(self, targ_state):
        """
        Called at every control loop execution. If no target state (targ_state) is provided, checks for alarms and
        maintains identity configuration.
        :return: Current state
        """
        curr_state = self.actuate(targ_state)
        # TODO: check alarms
        return curr_state

    def actuate(self, targ_state):
        curr_state = self.controller.actuate(self.curr_state, targ_state)
        return curr_state
