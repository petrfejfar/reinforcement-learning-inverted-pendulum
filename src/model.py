from math import cos, sin, floor, pi
import matplotlib.pyplot as plt


class Model:
    def __init__(self, initial_state, AREA_SIZE, SAFE_ANGLE_RAD):
        self.state = initial_state
        self.AREA_SIZE = AREA_SIZE
        self.SAFE_ANGLE_RAD = SAFE_ANGLE_RAD

    def reset(self, state=(0.0, 0.0, 0.0, 0.0)):
        self.state = state

    def get_state(self):
        return self.state

    def simulate(self, force, timespan):
        """
            Compute the next states given the force and the current states
            Update the four state variables, using Euler's method.
        """
        x = self.state[0]
        x_dot = self.state[1]
        theta = self.state[2]
        theta_dot = self.state[3]

        GRAVITY = 9.8
        MASSCART = 1.0
        MASSPOLE = 0.1
        TOTAL_MASS = MASSPOLE + MASSCART
        LENGTH = 0.5
        POLEMASS_LENGTH = MASSPOLE * LENGTH
        FOURTHIRDS = 4.0 / 3

        costheta = cos(theta)
        sintheta = sin(theta)

        temp = (force + POLEMASS_LENGTH * theta_dot * theta_dot * sintheta) / TOTAL_MASS

        thetaacc = (GRAVITY * sintheta - costheta * temp) / (LENGTH * (FOURTHIRDS - MASSPOLE * costheta * costheta / TOTAL_MASS))

        xacc = temp - POLEMASS_LENGTH * thetaacc * costheta / TOTAL_MASS

        self.state = (x + timespan * x_dot, x_dot + timespan * xacc, theta + timespan * theta_dot, theta_dot + timespan * thetaacc)

    def reward(self):
        return self.r_time()

    def system_safe(self):
        """Is the system in stable?"""
        theta = self.state[2] - (2*pi) * floor((self.state[2] + pi) / (2*pi))
        return abs(self.state[0]) < self.AREA_SIZE and abs(theta) < self.SAFE_ANGLE_RAD

    def r_time(self):
        """Return 2 (reward for two milliseconds in safe state) or 0"""
        if self.system_safe():
            return 1
        else:
            return 0

    def r_safe(self):
        """Return 0 if safe, -1 otherwise"""
        if self.system_safe():
            return 1
        else:
            return -1

    def r_theta(self):
        """Return reward based on pole angle"""
        if abs(self.state[0]) >= self.AREA_SIZE:
            return -pi ** 2

        return - (self.state[2]) ** 2

    def draw_state(self, force, filename, qTable):
        plt.clf()
        plt.xlim([-self.AREA_SIZE, self.AREA_SIZE])
        plt.ylim([-1, 3.8])

        plt.title('state: %+.2f %+.2f %+.2f %+.2f' % self.state)
        if self.system_safe():
            color = "green"
        else:
            color = "red"

        qVals = qTable.get_q_vals(self.state)
        plt.annotate("%09.2f" % qVals[0], (-2, 3), color="red" if force > 0 else "green")
        plt.annotate("%09.2f" % qVals[1], (1.5, 3), color="green" if force > 0 else "red")
        # draw state
        plt.plot([self.state[0] + 0, self.state[0] + sin(self.state[2])], [0, cos(self.state[2])], color=color, aa=True)

        # draw safezone
        plt.plot([self.state[0] + 0, self.state[0] + sin(self.SAFE_ANGLE_RAD)], [0, cos(self.SAFE_ANGLE_RAD)], color="grey", aa=True)
        plt.plot([self.state[0] + 0, self.state[0] + sin(-self.SAFE_ANGLE_RAD)], [0, cos(-self.SAFE_ANGLE_RAD)], color="grey", aa=True)

        # draw force
        if force > 0:
            plt.plot([0, 1], [0, 0], color="black", aa=True)
        else:
            plt.plot([0, -1], [0, 0], color="black", aa=True)

        plt.savefig(filename)
