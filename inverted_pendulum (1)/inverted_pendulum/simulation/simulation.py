from math import sin, cos, pi
import random

class Simulator:
    def simulate_step(self, state, F, dt):
        x, x_dot, theta, theta_dot = state

        m = 0.2
        M = 3
        L = 0.3
        K = 0.006
        A = x
        B = x_dot
        C = theta
        D = theta_dot
        g = 9.81
        b = 0.1

        A = L * sin(theta)
        B = L * cos(theta)

        C = (B**2 + K + B**2 * m + A**2 * m) * (M + m)
        D = F * B * m + B * m**2 * A * theta_dot**2 - b * x_dot * B * m - A * g * (M + m)

        theta_dot_dot = D / C
        x_dot_dot = ( (F + m * A * theta_dot**2 - b * x_dot) - D / C * B * m ) / (M + m)

        x_dot = state[1] + x_dot_dot * dt
        x = state[0] + x_dot * dt + x_dot_dot * dt * dt / 2

        theta_dot = state[3] + theta_dot_dot * dt
        theta = state[2] + theta_dot * dt + theta_dot_dot * dt * dt / 2

        return [ x, x_dot, theta, theta_dot ]

    def random_state(self, state):
        state[0] = 10
        state[1] = 0
        state[2] = pi - 60 * pi / 180
        state[3] = 0

        return state